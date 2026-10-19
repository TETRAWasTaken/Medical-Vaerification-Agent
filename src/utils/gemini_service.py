"""
Gemini LLM Service Integration for Clinical Trial Protocol Intelligence & Safety Review.
"""
import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import SystemMessage, HumanMessage
    GEMINI_LIB_AVAILABLE = True
except ImportError:
    GEMINI_LIB_AVAILABLE = False

# Default models in order of priority
SUPPORTED_GEMINI_MODELS = [
    "gemini-flash-latest",
    "gemini-3.8-flash",
    "gemini-pro-latest",
    "gemini-2.5-flash",
    "gemini-1.5-flash"
]


def resolve_api_key(api_key: Optional[str] = None) -> Optional[str]:
    """
    Resolves Gemini API key from direct argument or environment variables.
    """
    if api_key and api_key.strip():
        return api_key.strip()
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or None


def is_gemini_configured(api_key: Optional[str] = None) -> bool:
    """
    Returns True if Gemini libraries and a valid API key string are present.
    """
    resolved = resolve_api_key(api_key)
    return bool(GEMINI_LIB_AVAILABLE and resolved and len(resolved) > 5)


def extract_text_content(content: Any) -> str:
    """
    Extracts plain markdown text whether the LLM returns a string or a list of content blocks.
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            elif hasattr(part, "text"):
                parts.append(part.text)
        return "\n".join(parts).strip()
    return str(content).strip()


def get_gemini_llm(
    api_key: Optional[str] = None,
    model_name: str = "gemini-flash-latest",
    temperature: float = 0.2
):
    """
    Instantiates ChatGoogleGenerativeAI client with specified model and API key.
    """
    if not GEMINI_LIB_AVAILABLE:
        return None

    key = resolve_api_key(api_key)
    if not key:
        return None

    try:
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=key,
            temperature=temperature,
            max_output_tokens=1024
        )
    except Exception:
        return None


def generate_gemini_clinical_review(
    protocol: Dict[str, Any],
    ehr: Dict[str, Any],
    match_results: List[Dict[str, Any]],
    safety_alerts: List[str],
    deviation_warnings: List[str],
    api_key: Optional[str] = None,
    model_name: str = "gemini-flash-latest"
) -> str:
    """
    Uses Gemini LLM to synthesize an expert Clinical Safety Medical Monitor review,
    analyzing patient physiology, drug contraindications, and protocol deviations.
    Automatically tries fallback models if the requested model returns NOT_FOUND.
    """
    resolved_key = resolve_api_key(api_key)
    if not resolved_key:
        return (
            "Gemini LLM narrative not active. Provide a valid Gemini API key to enable "
            "automated AI-grounded Medical Monitor reviews and narrative synthesis."
        )

    # Format clinical context
    protocol_title = protocol.get("title", "Clinical Protocol")
    protocol_id = protocol.get("protocol_id", "N/A")
    investigational_drug = protocol.get("investigational_agent", "Investigational Agent")

    passed_count = sum(1 for r in match_results if r.get("passed"))
    total_count = len(match_results)

    failed_criteria = [
        f"- {r.get('rule_id')} [{r.get('criterion_type')}]: {r.get('description')} "
        f"(Target: {r.get('operator')} {r.get('target_value')}, Observed: {r.get('actual_value')})"
        for r in match_results if not r.get("passed")
    ]
    failed_summary = "\n".join(failed_criteria) if failed_criteria else "Zero failed criteria."

    alerts_summary = "\n".join(f"- {a}" for a in safety_alerts) if safety_alerts else "Zero drug contraindications."
    deviations_summary = "\n".join(f"- {d}" for d in deviation_warnings) if deviation_warnings else "Zero deviations."

    patient_summary = (
        f"Patient ID: {ehr.get('patient_id', 'UNKNOWN')}, Age: {ehr.get('age')}, "
        f"Gender: {ehr.get('gender')}, eGFR: {ehr.get('egfr')} mL/min/1.73m², "
        f"Platelets: {ehr.get('platelets'):,.0f} /µL, HbA1c: {ehr.get('hba1c')}%, "
        f"ECOG: {ehr.get('ecog')}, Brain Metastases: {ehr.get('brain_metastases')}, "
        f"Medications: {', '.join(ehr.get('medications_display', [])) or 'None'}"
    )

    system_prompt = (
        "You are a Senior Clinical Safety Medical Monitor and Clinical Pharmacology Specialist "
        "at a global Contract Research Organization (CRO). Your role is to provide authoritative, "
        "medically sound, and concise clinical trial eligibility and safety audits. "
        "Ground your reasoning in clinical pharmacology, FDA GCP guidelines, and organ system safety boundaries."
    )

    user_prompt = f"""
Please perform a rigorous Clinical Safety and Protocol Deviation Audit for the following candidate:

### TRIAL PROTOCOL:
- ID: {protocol_id}
- Title: {protocol_title}
- Intervention: {investigational_drug}

### PATIENT EHR PROFILE:
{patient_summary}

### DETERMINISTIC MATCHING RESULTS:
- Criteria Satisfied: {passed_count}/{total_count} ({round(passed_count/max(1, total_count)*100, 1)}%)
- Failed Protocol Rules:
{failed_summary}

### DETECTED SAFETY ALERTS & DRUG CONTRAINDICATIONS:
{alerts_summary}

### PROTOCOL DEVIATION WARNINGS:
{deviations_summary}

---
### REQUIRED OUTPUT:
Synthesize a concise, structured Clinical Safety Review containing:
1. **Clinical Assessment & Eligibility Rationale**: Summary of candidate status.
2. **Pharmacological & Organ Safety Review**: Analysis of drug interactions, renal/hepatic clearance, or toxicity liabilities.
3. **Regulatory Recommendation & Action Plan**: Final trial disposition (Enrolled, Screen Failure, or Dose-Modification Run-in) and mandatory safety monitoring.
Keep the commentary professional, high-density, and actionable for the Principal Investigator (PI).
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    # Models to attempt, putting requested model first
    models_to_try = [model_name] + [m for m in SUPPORTED_GEMINI_MODELS if m != model_name]

    last_error = None
    for target_model in models_to_try:
        try:
            llm = get_gemini_llm(api_key=resolved_key, model_name=target_model)
            if not llm:
                continue
            response = llm.invoke(messages)
            text_result = extract_text_content(response.content)
            if text_result:
                return text_result
        except Exception as e:
            last_error = e
            # If model is not found, continue to next supported candidate
            err_str = str(e)
            if "NOT_FOUND" in err_str or "404" in err_str:
                continue
            # For other errors (e.g. invalid key quota), return error immediately
            return f"Gemini API execution error ({target_model}): {err_str}"

    return f"Gemini API error: {str(last_error)}. Please verify your Gemini API key and model permissions."
