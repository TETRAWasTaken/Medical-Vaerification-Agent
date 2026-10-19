"""
Local safety database and deterministic rules for drug-drug contraindications.
"""
from typing import List, Dict, Set


# Drug classes and matching substrings for standard identification
DRUG_CLASSES: Dict[str, List[str]] = {
    "WARFARIN": ["warfarin", "coumadin", "jantoven"],
    "ASPIRIN": ["aspirin", "acetylsalicylic", "asa", "ecotrin", "bayer"],
    "METFORMIN": ["metformin", "glucophage", "fortamet", "glumetza", "riomet"],
    "CONTRAST_AGENT": ["contrast", "iodinated contrast", "gadolinium", "radiocontrast", "omnipaque", "visipaque"],
    "ACE_INHIBITOR": [
        "lisinopril", "enalapril", "ramipril", "captopril", "benazepril",
        "quinapril", "fosinopril", "perindopril", "trandolapril", "ace inhibitor", "acei"
    ],
    "POTASSIUM_SUPPLEMENT": [
        "potassium", "potassium chloride", "kcl", "k-dur", "klor-con", "micro-k", "potassium citrate"
    ],
}

# High-risk contraindication pairs and clinical descriptions
CONTRAINDICATION_RULES = [
    {
        "classes": ("WARFARIN", "ASPIRIN"),
        "severity": "CRITICAL",
        "title": "Severe Hemorrhagic Risk (Warfarin + Aspirin)",
        "message": "CRITICAL CONTRAINDICATION: Concurrent use of Warfarin and Aspirin synergistically impairs hemostasis and dramatically elevates major gastrointestinal and intracranial bleeding risk."
    },
    {
        "classes": ("METFORMIN", "CONTRAST_AGENT"),
        "severity": "CRITICAL",
        "title": "Lactic Acidosis Risk (Metformin + Contrast Agent)",
        "message": "CRITICAL CONTRAINDICATION: Administration of Iodinated Contrast Agents with Metformin may induce contrast-induced nephropathy leading to systemic metformin accumulation and potentially fatal lactic acidosis."
    },
    {
        "classes": ("ACE_INHIBITOR", "POTASSIUM_SUPPLEMENT"),
        "severity": "HIGH",
        "title": "Hyperkalemia Hazard (ACE Inhibitor + Potassium Supplement)",
        "message": "HIGH-RISK CONTRAINDICATION: Co-administration of ACE Inhibitors with exogenous Potassium supplements restricts renal potassium excretion, risking life-threatening hyperkalemia and cardiac dysrhythmias."
    }
]


def detect_drug_classes(med_list: List[str]) -> Set[str]:
    """
    Scans a list of medication strings and identifies recognized pharmacological classes.
    """
    detected: Set[str] = set()
    cleaned_meds = [m.lower().strip() for m in med_list if isinstance(m, str) and m.strip()]

    for med in cleaned_meds:
        for drug_class, patterns in DRUG_CLASSES.items():
            if any(pattern in med for pattern in patterns):
                detected.add(drug_class)

    return detected


def check_drug_interactions(med_list: List[str]) -> List[str]:
    """
    Deterministic safety function checking for known high-risk drug combinations:
    - Warfarin + Aspirin (Bleeding Hazard)
    - Metformin + Contrast Agent (Lactic Acidosis Risk)
    - ACE Inhibitor + Potassium Supplement (Hyperkalemia Hazard)

    Returns a list of human-readable alert strings.
    """
    if not med_list:
        return []

    detected_classes = detect_drug_classes(med_list)
    alerts: List[str] = []

    for rule in CONTRAINDICATION_RULES:
        class_a, class_b = rule["classes"]
        if class_a in detected_classes and class_b in detected_classes:
            alerts.append(rule["message"])

    return alerts
