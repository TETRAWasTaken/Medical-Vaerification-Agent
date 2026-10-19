# Autonomous Clinical Trial Matching & Protocol Deviation Auditor

An enterprise-grade, stateful multi-agent healthcare AI system built with **Python**, **Streamlit**, and **LangGraph/LangChain**. The platform automates patient eligibility screening against complex trial protocols, audits safety contraindications and drug-drug interactions, and flags protocol deviations with full regulatory audit trails.

---

## 🌟 Key Capabilities

- **Deterministic Protocol Deconstruction (Planner Agent)**: Converts unstructured clinical trial inclusion and exclusion criteria into structured, machine-evaluable Boolean rules with typed operators and clinical rationales.
- **EHR Standardization & Mapping (Research Agent)**: Normalizes unstructured patient EHR inputs to standard clinical ontologies (LOINC, SNOMED CT) and standardizes pharmaceutical nomenclature.
- **Deterministic Rule Evaluation (Executor Agent)**: Evaluates patient criteria against protocol thresholds and computes deterministic eligibility scores.
- **Gemini LLM Clinical Safety Medical Monitor**:
  - Leverages Google Gemini (`gemini-1.5-flash`, `gemini-2.0-flash`, or `gemini-1.5-pro`) via `langchain-google-genai`.
  - Analyzes the patient profile, protocol matching outcomes, organ system toxicities, and pharmacodynamic interactions.
  - Synthesizes an executive **Clinical Safety Review Narrative** with Principal Investigator (PI) recommendations and risk mitigation strategies.
- **Safety Gatekeeping & Contraindication Detection (Critic Agent)**:
  - Audits failed inclusion criteria and exclusion violations.
  - Cross-references current patient medications against high-risk drug-drug contraindication profiles (e.g., Warfarin + Aspirin, Metformin + Radiocontrast, ACE Inhibitors + Potassium).
  - Performs physiological safety boundary checks (e.g., eGFR < 45 and < 30 mL/min/1.73m² renal safety boundaries, thrombocytopenia thresholds, active CNS metastases).
- **Executive & Regulatory Audit Reporting**: Generates downloadable, publication-grade **PDF audit reports** (via ReportLab) and structured **JSON audit payloads**, including Gemini's clinical review narrative.

---

## 🏛️ Multi-Agent Architecture

The stateful workflow is orchestrated using **LangGraph**:

```
[Start]
   │
   ▼
[PlannerAgent] ───► Deconstructs protocol into JSON Boolean rules
   │
   ▼
[ResearchAgent] ──► Normalizes EHR parameters (LOINC/SNOMED)
   │
   ▼
[ExecutorAgent] ──► Evaluates criteria & computes Eligibility Score (%)
   │
   ▼
[CriticAgent] ────► Audits contraindications, physiological boundaries & issues Verdict
   │
   ▼
 [END] ───────────► State exported to Streamlit Dashboard & PDF Exporter
```

### State Structure (`AgentState`)
- `patient_id`: Unique identifier of the evaluated patient.
- `patient_raw_data`: Raw EHR payload from electronic health records.
- `protocol`: Selected clinical trial protocol specification.
- `parsed_rules`: List of extracted machine-evaluable rules with operators.
- `normalized_ehr`: Cleaned, type-cast, and canonicalized patient metrics.
- `match_results`: Deterministic pass/fail outcomes per criterion.
- `eligibility_score`: Percentage of protocol criteria satisfied.
- `critic_approved`: Boolean gatekeeper verdict (`True` only if zero critical hazards or violations).
- `deviation_warnings`: Identified protocol deviations and threshold deficits.
- `safety_alerts`: Severe drug interactions or organ safety hazard alarms.
- `audit_logs`: Timestamped system traceability logs.

---

## 📂 Project Structure

```
Medical Vaerification Agent/
├── app.py                      # Main Streamlit Dashboard UI
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive setup & run instructions
├── config/
│   └── protocols.json          # Synthetic clinical trial protocols (NCT05912381, NCT04192831)
├── data/
│   └── mock_ehr.json           # Synthetic patient EHR records (Eligible, Borderline, Hazard)
├── src/
│   ├── __init__.py
│   ├── state.py                # TypedDict state definition for multi-agent graph
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── planner.py          # Deconstructs protocols into Boolean evaluation rules
│   │   ├── research.py         # Normalizes patient EHR metrics (LOINC/SNOMED)
│   │   ├── executor.py         # Deterministic matching & eligibility scoring
│   │   └── critic.py           # Safety gatekeeper, drug interactions & deviations
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── drug_db.py          # Local safety database for drug-drug contraindications
│   │   └── pdf_exporter.py     # Generates downloadable PDF/JSON audit reports
│   └── workflow.py             # LangGraph state machine orchestrator
└── tests/
    ├── __init__.py
    └── test_agents.py          # Unit tests for multi-agent evaluation logic
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Virtual environment tool (`venv` or `uv`)

### 2. Environment Setup
```bash
# Clone or navigate to the repository
cd "/Users/anshumaansoni/PycharmProjects/Medical Vaerification Agent"

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch Streamlit Application
```bash
streamlit run app.py
```
Open your browser to the local address displayed (typically `http://localhost:8501`).

### 4. Running the Test Suite
```bash
pytest -v tests/
```
All unit tests validate:
- Planner rule extraction
- Research data normalization
- Executor mathematical scoring
- Critic safety gatekeeping and drug interaction detection
- PDF and JSON report generation

---

## 🧪 Pre-configured Synthetic Datasets

### Trial Protocols (`config/protocols.json`)
1. **NCT05912381 (Phase III Pembrolizumab + Metformin Oncology Study)**:
   - Evaluates immune checkpoint inhibition + metabolic modulation in solid tumors.
   - Inclusion: Age >= 18, eGFR >= 50 mL/min/1.73m², Platelets >= 100,000 /µL, ECOG <= 2.
   - Exclusion: Warfarin therapy, HbA1c > 10.0%, Active Brain Metastases.
2. **NCT04192831 (Phase II Diabetic Nephropathy SGLT2 Inhibitor Study)**:
   - Evaluates renal protection and albuminuria reduction in diabetic kidney disease.
   - Inclusion: Age >= 21, HbA1c <= 9.0%, UACR >= 300 mg/g.
   - Exclusion: eGFR < 30 mL/min/1.73m², Type 1 Diabetes, History of DKA.

### Patient Cohorts (`data/mock_ehr.json`)
- **PAT-101 (Eleanor Vance - Clean Eligible)**: Meets all inclusion criteria, 0 exclusion criteria, safe medications. Receives **APPROVED** verdict with 100% eligibility score.
- **PAT-202 (Marcus Holloway - Borderline Renal Failure)**: eGFR 42 mL/min (< 50) and low platelet count (82,000 /µL). Receives **REJECTED** verdict with renal boundary warnings.
- **PAT-303 (Arthur Pendelton - Critical Safety Hazard)**: Active Warfarin prescription and concurrent Aspirin causing high-risk bleeding contraindication, plus active CNS metastases. Receives **REJECTED** verdict with severe safety hazard alarms.
