"""
Research Agent: Cleans, type-casts, and normalizes unstructured patient EHR inputs to standards.
"""
from typing import Dict, Any, List
from datetime import datetime, timezone
from src.state import AgentState


# Standard clinical terminology aliases mapped to canonical keys
CANONICAL_FIELD_MAP = {
    # Renal & Labs (LOINC mappings)
    "egfr": "egfr",
    "estimated_gfr": "egfr",
    "gfr": "egfr",
    "estimated_glomerular_filtration_rate": "egfr",
    "loinc_33914_3": "egfr",

    "platelets": "platelets",
    "platelet_count": "platelets",
    "plt": "platelets",
    "loinc_777_3": "platelets",

    "hba1c": "hba1c",
    "glycated_hemoglobin": "hba1c",
    "a1c": "hba1c",
    "hemoglobin_a1c": "hba1c",
    "loinc_4548_4": "hba1c",

    "uacr": "uacr",
    "urine_albumin_creatinine_ratio": "uacr",
    "albumin_creatinine_ratio": "uacr",
    "loinc_14959_1": "uacr",

    # Demographics & Clinical Performance
    "age": "age",
    "patient_age": "age",
    "ecog": "ecog",
    "ecog_performance_status": "ecog",
    "ecog_ps": "ecog",

    # Clinical Conditions & History
    "brain_metastases": "brain_metastases",
    "active_brain_metastases": "brain_metastases",
    "cns_metastases": "brain_metastases",

    "type_1_diabetes": "type_1_diabetes",
    "t1d": "type_1_diabetes",
    "diabetes_type_1": "type_1_diabetes",

    "history_ketoacidosis": "history_ketoacidosis",
    "dka_history": "history_ketoacidosis",
    "diabetic_ketoacidosis": "history_ketoacidosis",

    "medications": "medications",
    "meds": "medications",
    "current_medications": "medications",
    "prescriptions": "medications"
}


class ResearchAgent:
    """
    Cleans, type-casts, and normalizes EHR inputs to standardized LOINC/SNOMED canonical data model.
    """

    def __init__(self):
        self.name = "ResearchAgent"

    @staticmethod
    def _to_float(val: Any, default: float = 0.0) -> float:
        if val is None:
            return default
        if isinstance(val, (int, float)):
            return float(val)
        try:
            # Strip strings with units like "78 mL/min"
            cleaned = str(val).split()[0].replace(",", "")
            return float(cleaned)
        except (ValueError, IndexError):
            return default

    @staticmethod
    def _to_int(val: Any, default: int = 0) -> int:
        if val is None:
            return default
        if isinstance(val, int):
            return val
        try:
            cleaned = str(val).split()[0].replace(",", "")
            return int(float(cleaned))
        except (ValueError, IndexError):
            return default

    @staticmethod
    def _to_bool(val: Any) -> bool:
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        if isinstance(val, str):
            lower = val.strip().lower()
            return lower in ("true", "1", "yes", "positive", "present", "y")
        return False

    def normalize(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts and normalizes all fields into a flat, standardized dictionary.
        """
        extracted: Dict[str, Any] = {}

        # 1. Flatten top-level and nested dictionaries (labs, vitals, history)
        flat_sources = [raw_data]
        if isinstance(raw_data.get("labs"), dict):
            flat_sources.append(raw_data["labs"])
        if isinstance(raw_data.get("vitals"), dict):
            flat_sources.append(raw_data["vitals"])
        if isinstance(raw_data.get("history"), dict):
            flat_sources.append(raw_data["history"])

        for src in flat_sources:
            for k, v in src.items():
                k_clean = k.strip().lower()
                canonical_k = CANONICAL_FIELD_MAP.get(k_clean, k_clean)
                if canonical_k not in extracted or extracted[canonical_k] is None:
                    extracted[canonical_k] = v

        # 2. Normalize types
        normalized: Dict[str, Any] = {}

        # Demographics
        normalized["age"] = self._to_int(extracted.get("age", 0))
        normalized["gender"] = str(extracted.get("gender", "Unknown")).strip()
        normalized["ecog"] = self._to_int(extracted.get("ecog", 0))

        # Labs
        normalized["egfr"] = self._to_float(extracted.get("egfr", 0.0))
        normalized["platelets"] = self._to_float(extracted.get("platelets", 0.0))
        normalized["hba1c"] = self._to_float(extracted.get("hba1c", 0.0))
        normalized["uacr"] = self._to_float(extracted.get("uacr", 0.0))

        # Boolean Flags
        normalized["brain_metastases"] = self._to_bool(extracted.get("brain_metastases", False))
        normalized["type_1_diabetes"] = self._to_bool(extracted.get("type_1_diabetes", False))
        normalized["history_ketoacidosis"] = self._to_bool(extracted.get("history_ketoacidosis", False))

        # Medications: Standardize to uppercase strings and keep raw list
        raw_meds = extracted.get("medications", [])
        if isinstance(raw_meds, str):
            med_list = [m.strip() for m in raw_meds.split(",") if m.strip()]
        elif isinstance(raw_meds, list):
            med_list = [str(m).strip() for m in raw_meds if str(m).strip()]
        else:
            med_list = []

        normalized["medications"] = [m.upper() for m in med_list]
        normalized["medications_display"] = med_list

        # Preserve meta info
        normalized["patient_id"] = str(raw_data.get("patient_id", extracted.get("patient_id", "UNKNOWN_PATIENT")))
        normalized["name"] = str(raw_data.get("name", extracted.get("name", "Unknown Patient")))

        return normalized

    def run(self, state: AgentState) -> AgentState:
        """
        Executes EHR research and data normalization within the multi-agent graph.
        """
        raw_data = state.get("patient_raw_data", {})
        patient_id = state.get("patient_id", raw_data.get("patient_id", "UNKNOWN"))
        audit_logs = list(state.get("audit_logs", []))

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        audit_logs.append(f"[{now_str}] [ResearchAgent] Normalizing EHR data for patient {patient_id} against LOINC/SNOMED schemas.")

        normalized_ehr = self.normalize(raw_data)

        audit_logs.append(
            f"[{now_str}] [ResearchAgent] Standardized parameters: Age={normalized_ehr['age']} yrs, "
            f"eGFR={normalized_ehr['egfr']:.1f} mL/min/1.73m², Platelets={normalized_ehr['platelets']:,.0f} /µL, "
            f"HbA1c={normalized_ehr['hba1c']:.1f}%, UACR={normalized_ehr['uacr']:.1f} mg/g, "
            f"ECOG={normalized_ehr['ecog']}, BrainMetastases={normalized_ehr['brain_metastases']}."
        )
        audit_logs.append(
            f"[{now_str}] [ResearchAgent] Normalized {len(normalized_ehr['medications'])} medications to uppercase taxonomy: "
            f"{', '.join(normalized_ehr['medications']) or 'None'}."
        )

        return {
            **state,
            "normalized_ehr": normalized_ehr,
            "audit_logs": audit_logs
        }
