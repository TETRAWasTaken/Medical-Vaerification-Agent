"""
Autonomous Clinical Trial Matching & Protocol Deviation Auditor
Main Streamlit Application
"""
import json
import os
import sys
import pandas as pd
import streamlit as st

# Ensure repository root is in Python module search path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.workflow import run_audit_workflow_streaming, run_audit_workflow
from src.utils.pdf_exporter import generate_audit_pdf, export_audit_json

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Clinical Trial Auditor | Multi-Agent Safety Platform",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling (Enterprise Medical Dark/Light Adaptive Aesthetic) ---
st.markdown("""
<style>
    /* Main container refinements */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }
    
    /* Header card */
    .header-card {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: #F8FAFC;
        padding: 24px 30px;
        border-radius: 12px;
        margin-bottom: 24px;
        border: 1px solid #334155;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.15);
    }
    .header-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 6px;
        letter-spacing: -0.02em;
        color: #FFFFFF;
    }
    .header-subtitle {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-bottom: 12px;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-right: 8px;
        letter-spacing: 0.05em;
    }
    .badge-blue { background-color: #1E3A8A; color: #93C5FD; border: 1px solid #2563EB; }
    .badge-emerald { background-color: #064E3B; color: #6EE7B7; border: 1px solid #059669; }
    .badge-purple { background-color: #4C1D95; color: #D8B4FE; border: 1px solid #7C3AED; }

    /* Metric cards */
    .metric-container {
        display: flex;
        gap: 16px;
        margin-bottom: 20px;
    }
    .metric-card {
        flex: 1;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748B;
        margin-bottom: 4px;
        letter-spacing: 0.04em;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0F172A;
    }
    .verdict-approved {
        color: #15803D !important;
        background: #DCFCE7;
        padding: 4px 12px;
        border-radius: 6px;
        display: inline-block;
        border: 1px solid #86EFAC;
    }
    .verdict-rejected {
        color: #B91C1C !important;
        background: #FEE2E2;
        padding: 4px 12px;
        border-radius: 6px;
        display: inline-block;
        border: 1px solid #FCA5A5;
    }

    /* Safety Alert Container */
    .alert-box {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 12px;
        color: #991B1B;
    }
    .alert-box h4 {
        margin: 0 0 6px 0;
        font-size: 0.95rem;
        font-weight: 700;
    }
    .alert-box p {
        margin: 0;
        font-size: 0.85rem;
        line-height: 1.4;
    }

    /* Warning Alert Container */
    .warning-box {
        background-color: #FFFBEB;
        border-left: 5px solid #F59E0B;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 12px;
        color: #92400E;
    }

    /* Clean Success Box */
    .success-box {
        background-color: #F0FDF4;
        border-left: 5px solid #22C55E;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 12px;
        color: #166534;
    }

    /* Gemini Card */
    .gemini-card {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 100%);
        border: 1px solid #6366F1;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 24px;
        color: #F8FAFC;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.15);
    }
    .gemini-card h3 {
        color: #A5B4FC;
        margin-top: 0;
        font-size: 1.15rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Protocol Info Panel */
    .protocol-panel {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 16px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# --- Helper Data Loaders ---
@st.cache_data
def load_protocols() -> dict:
    config_path = os.path.join(CURRENT_DIR, "config", "protocols.json")
    with open(config_path, "r") as f:
        return json.load(f)


@st.cache_data
def load_mock_ehr() -> dict:
    data_path = os.path.join(CURRENT_DIR, "data", "mock_ehr.json")
    with open(data_path, "r") as f:
        return json.load(f)


protocols_data = load_protocols()
mock_ehr_data = load_mock_ehr()

# --- Header Banner ---
st.markdown("""
<div class="header-card">
    <div class="header-title">🩺 Autonomous Clinical Trial Matching & Protocol Deviation Auditor</div>
    <div class="header-subtitle">Multi-Agent Stateful Graph Workflow: Protocol Rule Deconstruction • EHR Normalization • Deterministic Matching • Safety & Contraindication Gatekeeper</div>
    <div>
        <span class="badge-pill badge-blue">LangGraph Orchestrated</span>
        <span class="badge-pill badge-emerald">Deterministic Rule Engine</span>
        <span class="badge-pill badge-purple">Safety Critic Gatekeeper</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Configuration & Inputs ---
with st.sidebar:
    st.header("📋 Clinical Trial Protocol")
    protocol_options = list(protocols_data.keys())
    selected_proto_id = st.selectbox(
        "Select Protocol",
        protocol_options,
        format_func=lambda pid: f"{pid} - {protocols_data[pid].get('short_name', pid)}"
    )
    current_protocol = protocols_data[selected_proto_id]

    # Protocol summary card
    st.markdown(f"""
    <div class="protocol-panel">
        <b>Condition:</b> {current_protocol.get('condition')}<br/>
        <b>Phase:</b> {current_protocol.get('phase')}<br/>
        <b>Intervention:</b> <i>{current_protocol.get('investigational_agent')}</i><br/>
        <b>Rules:</b> {len(current_protocol.get('inclusion_criteria', []))} Inclusions, {len(current_protocol.get('exclusion_criteria', []))} Exclusions
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.header("👤 Patient EHR Record")

    preset_map = {
        "PAT-101": "PAT-101: Eleanor Vance (Clean Eligible)",
        "PAT-202": "PAT-202: Marcus Holloway (Borderline Renal Failure)",
        "PAT-303": "PAT-303: Arthur Pendelton (Critical Safety Hazard)"
    }

    selected_preset = st.selectbox(
        "Load Patient Preset Profile",
        list(preset_map.keys()),
        format_func=lambda k: preset_map[k]
    )

    base_patient = mock_ehr_data[selected_preset]

    st.caption("Interactive EHR Fine-Tuning:")

    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        age_input = st.number_input("Age (yrs)", min_value=1, max_value=110, value=int(base_patient.get("age", 50)))
        egfr_input = st.number_input("eGFR (mL/min)", min_value=5.0, max_value=150.0, value=float(base_patient.get("labs", {}).get("egfr", 60.0)), step=1.0)
        platelets_input = st.number_input("Platelets (/µL)", min_value=10000, max_value=800000, value=int(base_patient.get("labs", {}).get("platelets", 200000)), step=5000)

    with col_sb2:
        hba1c_input = st.number_input("HbA1c (%)", min_value=3.5, max_value=16.0, value=float(base_patient.get("labs", {}).get("hba1c", 6.5)), step=0.1)
        uacr_input = st.number_input("UACR (mg/g)", min_value=0.0, max_value=2500.0, value=float(base_patient.get("labs", {}).get("uacr", 100.0)), step=10.0)
        ecog_input = st.selectbox("ECOG Status", [0, 1, 2, 3, 4], index=int(base_patient.get("ecog", 1)))

    st.markdown("<b>Clinical Comorbidities & Risk Factors:</b>", unsafe_allow_html=True)
    brain_met_input = st.checkbox("Active Brain Metastases", value=bool(base_patient.get("history", {}).get("brain_metastases", False)))
    t1d_input = st.checkbox("Type 1 Diabetes Mellitus", value=bool(base_patient.get("history", {}).get("type_1_diabetes", False)))
    dka_input = st.checkbox("History of Ketoacidosis (DKA)", value=bool(base_patient.get("history", {}).get("history_ketoacidosis", False)))

    meds_default = "\n".join(base_patient.get("medications", []))
    meds_text = st.text_area("Medications (One per line)", value=meds_default, height=100)
    meds_list = [m.strip() for m in meds_text.strip().split("\n") if m.strip()]

    st.markdown("---")
    st.header("🤖 Gemini LLM Integration")
    gemini_default_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    gemini_key_input = st.text_input(
        "Gemini API Key",
        value=gemini_default_key,
        type="password",
        help="Paste your Google Gemini API key here to activate AI-powered Clinical Safety Medical Monitor reviews."
    )
    gemini_model_choice = st.selectbox(
        "Gemini Model",
        ["gemini-flash-latest", "gemini-3.8-flash", "gemini-pro-latest"],
        index=0
    )

    if gemini_key_input and len(gemini_key_input.strip()) > 5:
        st.markdown('<span class="badge-pill badge-emerald">🟢 Gemini LLM Active</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-pill badge-blue">⚪ Deterministic (Key Optional)</span>', unsafe_allow_html=True)

    # Construct active patient payload
    active_patient = {
        "patient_id": base_patient.get("patient_id", "CUSTOM-PAT"),
        "name": base_patient.get("name", "Custom Patient"),
        "gender": base_patient.get("gender", "Unknown"),
        "age": age_input,
        "ecog": ecog_input,
        "labs": {
            "egfr": egfr_input,
            "platelets": platelets_input,
            "hba1c": hba1c_input,
            "uacr": uacr_input
        },
        "history": {
            "brain_metastases": brain_met_input,
            "type_1_diabetes": t1d_input,
            "history_ketoacidosis": dka_input
        },
        "medications": meds_list
    }

    run_audit_clicked = st.button("Run Multi-Agent Audit", type="primary", use_container_width=True)

# --- Execution Management in Session State ---
state_key = f"audit_state_{selected_proto_id}_{selected_preset}_{age_input}_{egfr_input}_{platelets_input}_{hba1c_input}_{brain_met_input}_{len(meds_list)}"

if run_audit_clicked or "current_audit_state" not in st.session_state:
    try:
        with st.status("Executing Multi-Agent Clinical Trial Workflow...", expanded=True) as status_box:
            stream_gen = run_audit_workflow_streaming(
                active_patient,
                current_protocol,
                gemini_api_key=gemini_key_input.strip() if gemini_key_input else None
            )
            final_state = None
            for agent_id, desc, curr_state in stream_gen:
                st.write(f"**[{agent_id.upper()}]** {desc}")
                final_state = curr_state
            if status_box is not None:
                status_box.update(label="Multi-Agent Execution Completed", state="complete", expanded=False)
    except Exception:
        final_state = run_audit_workflow(
            active_patient,
            current_protocol,
            gemini_api_key=gemini_key_input.strip() if gemini_key_input else None
        )

    st.session_state["current_audit_state"] = final_state
else:
    final_state = st.session_state["current_audit_state"]

# --- Top Key Metrics Row ---
score = final_state.get("eligibility_score", 0.0)
approved = final_state.get("critic_approved", False)
match_results = final_state.get("match_results", [])
passed_rules = sum(1 for r in match_results if r.get("passed"))
total_rules = len(match_results)
safety_alerts = final_state.get("safety_alerts", [])
deviation_warnings = final_state.get("deviation_warnings", [])

col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    st.metric(
        label="Protocol Eligibility Score",
        value=f"{score:.1f}%",
        delta=f"{passed_rules} of {total_rules} Rules Met"
    )

with col_m2:
    if approved:
        badge_html = '<div class="verdict-approved">APPROVED (ELIGIBLE)</div>'
    else:
        badge_html = '<div class="verdict-rejected">REJECTED (INELIGIBLE)</div>'

    st.markdown("**Critic Gatekeeper Verdict**")
    st.markdown(badge_html, unsafe_allow_html=True)

with col_m3:
    st.metric(
        label="Evaluated Criteria",
        value=f"{passed_rules}/{total_rules}",
        delta=f"{total_rules - passed_rules} Failed Rules" if total_rules - passed_rules > 0 else "All Passed",
        delta_color="inverse" if total_rules - passed_rules > 0 else "normal"
    )

with col_m4:
    st.metric(
        label="Safety & Contraindications",
        value=f"{len(safety_alerts)} Hazard(s)",
        delta=f"{len(deviation_warnings)} Deviations",
        delta_color="inverse" if len(safety_alerts) > 0 else "off"
    )

st.markdown("<br/>", unsafe_allow_html=True)

# --- Critic Safety & Deviation Alerts Section ---
if safety_alerts or deviation_warnings:
    st.subheader("Safety Hazards & Protocol Deviations Detected")
    
    for alert in safety_alerts:
        st.markdown(f"""
        <div class="alert-box">
            <h4>CRITICAL SAFETY CONTRAINDICATION</h4>
            <p>{alert}</p>
        </div>
        """, unsafe_allow_html=True)

    for dev in deviation_warnings:
        st.markdown(f"""
        <div class="warning-box">
            <h4>⚡ PROTOCOL DEVIATION DETECTED</h4>
            <p>{dev}</p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="success-box">
        <h4>PROTOCOL COMPLIANCE VERIFIED</h4>
        <p>The Critic Agent verified that zero contraindications, drug interactions, or physiological safety boundary breaches were detected. Candidate is fully compliant.</p>
    </div>
    """, unsafe_allow_html=True)

# --- Gemini LLM Clinical Safety Review Section ---
st.subheader("Gemini Medical Monitor Analysis & Clinical Narrative")
llm_narrative = final_state.get("llm_safety_narrative", "").strip()

if gemini_key_input and len(gemini_key_input.strip()) > 5:
    if llm_narrative and "Deterministic safety audit completed" not in llm_narrative:
        st.markdown("""
        <div class="gemini-card">
            <h3>Gemini Senior Clinical Safety Monitor Narrative</h3>
            <div style="font-size: 0.92rem; line-height: 1.6; color: #E2E8F0;">
        """, unsafe_allow_html=True)
        st.markdown(llm_narrative)
        st.markdown("</div></div>", unsafe_allow_html=True)
    else:
        st.info("Gemini is connected. Click **'🚀 Run Multi-Agent Audit'** in the sidebar to generate real-time AI clinical analysis.")
else:
    st.markdown("""
    <div style="background-color: #F8FAFC; border: 1px dashed #94A3B8; border-radius: 8px; padding: 16px 20px; margin-bottom: 20px;">
        <h4 style="margin: 0 0 6px 0; color: #1E293B; font-size: 0.95rem;">🔑 Activate Gemini Clinical Intelligence</h4>
        <p style="margin: 0; font-size: 0.88rem; color: #64748B; line-height: 1.4;">
            The multi-agent deterministic rules and local contraindication database are actively screening this patient. 
            Enter your <b>Gemini API Key</b> in the sidebar to activate the <b>Senior Clinical Safety Medical Monitor LLM</b> 
            for automated clinical pharmacology risk analysis, organ toxicity reviews, and PI regulatory recommendations.
        </p>
    </div>
    """, unsafe_allow_html=True)

# --- Detailed Rule Audit Matrix ---
st.subheader("Deterministic Protocol Rule Audit Matrix")

if match_results:
    table_rows = []
    for r in match_results:
        target_display = f"{r.get('operator')} {r.get('target_value')} {r.get('unit', '')}".strip()
        actual_display = str(r.get("actual_value", "N/A"))
        if isinstance(r.get("actual_value"), float):
            actual_display = f"{r.get('actual_value'):.1f}"

        status_icon = "🟢 PASS" if r.get("passed") else "🔴 FAIL"

        table_rows.append({
            "Status": status_icon,
            "Rule ID": r.get("rule_id"),
            "Type": r.get("criterion_type"),
            "Description": r.get("description"),
            "Target Condition": target_display,
            "Observed Value": actual_display,
            "Clinical Explanation": r.get("clinical_explanation")
        })

    df_rules = pd.DataFrame(table_rows)

    st.dataframe(
        df_rules,
        column_config={
            "Status": st.column_config.TextColumn("Status", width="small"),
            "Rule ID": st.column_config.TextColumn("Rule ID", width="small"),
            "Type": st.column_config.TextColumn("Type", width="small"),
            "Description": st.column_config.TextColumn("Criterion Description", width="large"),
            "Target Condition": st.column_config.TextColumn("Target Threshold", width="medium"),
            "Observed Value": st.column_config.TextColumn("Observed EHR Value", width="medium"),
            "Clinical Explanation": st.column_config.TextColumn("Agent Assessment", width="large"),
        },
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No rule match results available. Click 'Run Multi-Agent Audit' to evaluate.")

# --- Actionable Report Exporter Bar ---
st.subheader("Export & Regulatory Audit Artifacts")

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    pdf_bytes = generate_audit_pdf(final_state)
    st.download_button(
        label="Download Regulatory Audit PDF Report",
        data=pdf_bytes,
        file_name=f"clinical_trial_audit_{final_state.get('patient_id')}_{selected_proto_id}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

with col_exp2:
    json_export_str = export_audit_json(final_state)
    st.download_button(
        label="Download Structured JSON Audit Package",
        data=json_export_str,
        file_name=f"clinical_trial_audit_{final_state.get('patient_id')}_{selected_proto_id}.json",
        mime="application/json",
        use_container_width=True
    )

# --- Traceability Log Expander ---
with st.expander("🔍 Complete Multi-Agent Traceability & Audit Logs", expanded=False):
    logs = final_state.get("audit_logs", [])
    if logs:
        for log_line in logs:
            if "[CriticAgent]" in log_line and "ALARM" in log_line:
                st.markdown(f":red[`{log_line}`]")
            elif "[ExecutorAgent]" in log_line and "=> FAIL" in log_line:
                st.markdown(f":orange[`{log_line}`]")
            elif "[CriticAgent]" in log_line and "APPROVED" in log_line:
                st.markdown(f":green[`{log_line}`]")
            else:
                st.text(log_line)
    else:
        st.write("No execution logs recorded.")
