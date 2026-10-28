"""
Critic Agent: Serves as clinical trial compliance gatekeeper, drug interaction auditor & safety deviation detector.
"""
from typing import Dict, Any, List
from datetime import datetime, timezone
from src.state import AgentState
from src.utils.drug_db import check_drug_interactions


class CriticAgent:
    """
    Safety gatekeeper and protocol deviation auditor:
    1. Audits all failed rules (both inclusion failures and exclusion violations).
    2. Cross-references patient medications with the safety database for drug-drug contraindications.
    3. Performs physiological safety boundary checks (e.g., eGFR renal reserve thresholds).
    4. Issues final binary verdict (`critic_approved`) and populates deviation warnings and safety alerts.
    """

    def __init__(self):
        self.name = "CriticAgent"

    def run(self, state: AgentState) -> AgentState:
        """
        Executes critical audit and gatekeeping within the multi-agent graph.
        """
        match_results = state.get("match_results", [])
        ehr = state.get("normalized_ehr", {})
        protocol = state.get("protocol", {})
        audit_logs = list(state.get("audit_logs", []))

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        audit_logs.append(f"[{now_str}] [CriticAgent] Commencing comprehensive safety and protocol deviation audit.")

        deviation_warnings: List[str] = []
        safety_alerts: List[str] = []

        # 1. Audit Failed Rules
        failed_exclusions: List[Dict[str, Any]] = []
        failed_inclusions: List[Dict[str, Any]] = []

        for item in match_results:
            if not item["passed"]:
                rule_desc = f"Rule {item['rule_id']} [{item['criterion_type']}]: {item['description']} (Observed: {item['actual_value']})"
                if item["criterion_type"] == "EXCLUSION":
                    failed_exclusions.append(item)
                    deviation_warnings.append(f"CRITICAL EXCLUSION VIOLATION: {rule_desc}")
                else:
                    failed_inclusions.append(item)
                    deviation_warnings.append(f"INCLUSION CRITERIA DEFICIT: {rule_desc}")

        if failed_exclusions:
            audit_logs.append(
                f"[{now_str}] [CriticAgent] Flagged {len(failed_exclusions)} mandatory protocol exclusion violation(s)."
            )
        if failed_inclusions:
            audit_logs.append(
                f"[{now_str}] [CriticAgent] Flagged {len(failed_inclusions)} unmet protocol inclusion criterion/criteria."
            )

        # 2. Medication Safety & Drug-Drug Interactions
        raw_meds = ehr.get("medications_display", ehr.get("medications", []))
        drug_alerts = check_drug_interactions(raw_meds)
        for da in drug_alerts:
            safety_alerts.append(da)
            audit_logs.append(f"[{now_str}] [CriticAgent] [DRUG_INTERACTION_ALARM] {da}")

        # 3. Physiological & Organ Safety Boundary Checks
        egfr_val = ehr.get("egfr", 0.0)
        platelet_val = ehr.get("platelets", 0.0)

        # Renal boundary: eGFR < 45 mL/min warning, eGFR < 30 severe warning
        if egfr_val < 30.0:
            severe_renal_alert = (
                f"SEVERE RENAL RISK: Baseline eGFR is {egfr_val:.1f} mL/min/1.73m² (< 30.0), indicating "
                f"Stage 4-5 chronic kidney failure. Severe toxic accumulation risk."
            )
            safety_alerts.append(severe_renal_alert)
            deviation_warnings.append(f"RENAL SAFETY HAZARD: eGFR {egfr_val:.1f} < 30.0 mL/min/1.73m².")
            audit_logs.append(f"[{now_str}] [CriticAgent] {severe_renal_alert}")
        elif egfr_val < 45.0:
            moderate_renal_alert = (
                f"RENAL BOUNDARY WARNING: Baseline eGFR is {egfr_val:.1f} mL/min/1.73m² (< 45.0). "
                f"Requires nephrology clearance and dose reduction protocol monitoring."
            )
            safety_alerts.append(moderate_renal_alert)
            deviation_warnings.append(f"RENAL BORDERLINE DEFICIT: eGFR {egfr_val:.1f} < 45.0 mL/min/1.73m².")
            audit_logs.append(f"[{now_str}] [CriticAgent] {moderate_renal_alert}")

        # Thrombocytopenia boundary
        if platelet_val < 100000.0:
            plt_alert = (
                f"HEMATOLOGIC BOUNDARY WARNING: Platelet count is {platelet_val:,.0f} /µL (< 100,000 /µL), "
                f"posing an elevated bleeding and myelosuppression liability."
            )
            safety_alerts.append(plt_alert)
            audit_logs.append(f"[{now_str}] [CriticAgent] {plt_alert}")

        # Active Brain Metastases check
        if ehr.get("brain_metastases", False):
            cns_alert = "NEUROLOGIC SAFETY HAZARD: Active CNS brain metastases recorded. Uncontrolled intracranial disease liability."
            safety_alerts.append(cns_alert)
            audit_logs.append(f"[{now_str}] [CriticAgent] {cns_alert}")

        # 4. Final Gatekeeping Decision
        # Approved ONLY if ZERO critical safety alerts, ZERO failed exclusions, and ZERO failed inclusions
        has_critical_safety = (len(safety_alerts) > 0)
        has_exclusion_failures = (len(failed_exclusions) > 0)
        has_inclusion_failures = (len(failed_inclusions) > 0)

        if not has_critical_safety and not has_exclusion_failures and not has_inclusion_failures:
            critic_approved = True
            decision_msg = "GATEWAY APPROVAL: Patient successfully cleared all safety checks, zero deviations detected."
        else:
            critic_approved = False
            reasons = []
            if has_critical_safety:
                reasons.append(f"{len(safety_alerts)} safety hazard(s)")
            if has_exclusion_failures:
                reasons.append(f"{len(failed_exclusions)} exclusion violation(s)")
            if has_inclusion_failures:
                reasons.append(f"{len(failed_inclusions)} unmet inclusion criteria")
            decision_msg = f"GATEWAY REJECTION: Protocol deviations detected ({', '.join(reasons)})."

        audit_logs.append(f"[{now_str}] [CriticAgent] Final Verdict: {'APPROVED' if critic_approved else 'REJECTED'}. {decision_msg}")

        # 5. Gemini LLM Medical Monitor Review
        from src.utils.gemini_service import is_gemini_configured, generate_gemini_clinical_review

        gemini_key = state.get("gemini_api_key")
        gemini_model = state.get("gemini_model", "gemini-2.5-flash")
        if is_gemini_configured(gemini_key):
            audit_logs.append(f"[{now_str}] [CriticAgent] Engaging Gemini LLM as Senior Clinical Safety Medical Monitor ({gemini_model})...")
            llm_narrative = generate_gemini_clinical_review(
                protocol=protocol,
                ehr=ehr,
                match_results=match_results,
                safety_alerts=safety_alerts,
                deviation_warnings=deviation_warnings,
                api_key=gemini_key,
                model_name=gemini_model
            )
            audit_logs.append(f"[{now_str}] [CriticAgent] Gemini LLM clinical safety narrative successfully generated.")
        else:
            llm_narrative = (
                "Deterministic safety audit completed. Connect a Gemini API key in the sidebar "
                "to activate real-time Gemini LLM clinical pharmacology analysis and Medical Monitor narrative synthesis."
            )

        return {
            **state,
            "critic_approved": critic_approved,
            "deviation_warnings": deviation_warnings,
            "safety_alerts": safety_alerts,
            "llm_safety_narrative": llm_narrative,
            "audit_logs": audit_logs
        }
