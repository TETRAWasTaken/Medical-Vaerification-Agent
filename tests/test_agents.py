"""
Unit tests for multi-agent evaluation logic, safety database, and end-to-end workflow.
"""
import json
import os
import pytest

from src.agents.planner import PlannerAgent
from src.agents.research import ResearchAgent
from src.agents.executor import ExecutorAgent
from src.agents.critic import CriticAgent
from src.utils.drug_db import check_drug_interactions, detect_drug_classes
from src.utils.pdf_exporter import generate_audit_pdf, export_audit_json
from src.workflow import run_audit_workflow


@pytest.fixture
def protocols():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "protocols.json")
    with open(config_path, "r") as f:
        return json.load(f)


@pytest.fixture
def mock_ehr():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "mock_ehr.json")
    with open(data_path, "r") as f:
        return json.load(f)


def test_drug_db_interactions():
    """Verify deterministic detection of known contraindications."""
    # 1. Warfarin + Aspirin
    alerts1 = check_drug_interactions(["Warfarin 5mg daily", "Aspirin 81mg"])
    assert len(alerts1) >= 1
    assert any("Warfarin and Aspirin" in a for a in alerts1)

    # 2. Metformin + Contrast Agent
    alerts2 = check_drug_interactions(["Metformin 500mg", "Iodinated Contrast media"])
    assert len(alerts2) >= 1
    assert any("Metformin" in a and "Contrast" in a for a in alerts2)

    # 3. ACE Inhibitor + Potassium Supplement
    alerts3 = check_drug_interactions(["Lisinopril 10mg", "Potassium Chloride 20mEq"])
    assert len(alerts3) >= 1
    assert any("Hyperkalemia" in a or "ACE Inhibitor" in a for a in alerts3)

    # 4. Safe medications (No interaction)
    alerts4 = check_drug_interactions(["Atorvastatin 20mg", "Levothyroxine 50mcg"])
    assert len(alerts4) == 0


def test_planner_agent(protocols):
    """Test protocol parsing into machine-evaluable rules."""
    oncology_proto = protocols["NCT05912381"]
    planner = PlannerAgent()
    state = {"protocol": oncology_proto, "audit_logs": []}
    result_state = planner.run(state)

    rules = result_state.get("parsed_rules", [])
    assert len(rules) == 7  # 4 inclusions + 3 exclusions
    fields = [r["field"] for r in rules]
    assert "age" in fields
    assert "egfr" in fields
    assert "platelets" in fields
    assert "medications" in fields
    assert "brain_metastases" in fields


def test_research_agent(mock_ehr):
    """Test data extraction, type coercion, and standardizations."""
    patient = mock_ehr["PAT-101"]
    researcher = ResearchAgent()
    state = {"patient_raw_data": patient, "patient_id": patient["patient_id"], "audit_logs": []}
    result_state = researcher.run(state)

    norm = result_state.get("normalized_ehr", {})
    assert norm["age"] == 54
    assert norm["egfr"] == 78.5
    assert norm["platelets"] == 245000.0
    assert norm["brain_metastases"] is False
    assert "METFORMIN 500MG BID" in norm["medications"]


def test_executor_agent(protocols, mock_ehr):
    """Test rule evaluation and eligibility scoring."""
    oncology_proto = protocols["NCT05912381"]
    planner = PlannerAgent()
    p_state = planner.run({"protocol": oncology_proto, "audit_logs": []})

    researcher = ResearchAgent()
    r_state = researcher.run({
        "patient_raw_data": mock_ehr["PAT-101"],
        "patient_id": "PAT-101",
        "audit_logs": []
    })

    combined_state = {
        **p_state,
        "normalized_ehr": r_state["normalized_ehr"]
    }

    executor = ExecutorAgent()
    exec_state = executor.run(combined_state)

    assert exec_state["eligibility_score"] == 100.0
    assert len(exec_state["match_results"]) == 7
    assert all(r["passed"] for r in exec_state["match_results"])


def test_workflow_clean_patient_eligible(protocols, mock_ehr):
    """End-to-end: PAT-101 must achieve 100% score and be APPROVED."""
    final_state = run_audit_workflow(mock_ehr["PAT-101"], protocols["NCT05912381"])
    assert final_state["eligibility_score"] == 100.0
    assert final_state["critic_approved"] is True
    assert len(final_state["safety_alerts"]) == 0
    assert len(final_state["deviation_warnings"]) == 0


def test_workflow_borderline_renal_failure(protocols, mock_ehr):
    """End-to-end: PAT-202 must fail eGFR and Platelets, and be REJECTED."""
    final_state = run_audit_workflow(mock_ehr["PAT-202"], protocols["NCT05912381"])
    assert final_state["critic_approved"] is False
    assert final_state["eligibility_score"] < 100.0

    # Must contain renal and platelet warnings
    dev_str = " ".join(final_state["deviation_warnings"])
    assert "egfr" in dev_str.lower() or "renal" in dev_str.lower()


def test_workflow_critical_safety_hazard(protocols, mock_ehr):
    """End-to-end: PAT-303 must trigger critical drug interaction and be REJECTED."""
    final_state = run_audit_workflow(mock_ehr["PAT-303"], protocols["NCT05912381"])
    assert final_state["critic_approved"] is False

    # Check drug alert for Warfarin + Aspirin
    safety_str = " ".join(final_state["safety_alerts"])
    assert "Warfarin" in safety_str and "Aspirin" in safety_str


def test_pdf_and_json_export(protocols, mock_ehr):
    """Ensure report exporter outputs valid PDF bytes and JSON string."""
    state = run_audit_workflow(mock_ehr["PAT-101"], protocols["NCT05912381"])

    pdf_bytes = generate_audit_pdf(state)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")

    json_str = export_audit_json(state)
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed["verdict"]["critic_approved"] is True
    assert parsed["verdict"]["eligibility_score"] == 100.0


def test_gemini_service_resolution(monkeypatch):
    """Verify Gemini API key resolution and safe graceful fallback."""
    from src.utils.gemini_service import is_gemini_configured, generate_gemini_clinical_review

    # When env keys are cleared and argument is empty, is_gemini_configured must be False
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    assert is_gemini_configured(None) is False
    assert is_gemini_configured("") is False

    # When key is not provided, returns helpful message instead of crashing
    narrative = generate_gemini_clinical_review(
        protocol={"title": "Test"},
        ehr={"patient_id": "P1"},
        match_results=[],
        safety_alerts=[],
        deviation_warnings=[],
        api_key=None
    )
    assert "Gemini LLM narrative not active" in narrative

