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

try:
    from google import genai
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False

# Default models in order of priority (Fastest and officially supported first)
SUPPORTED_GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash-002",
    "gemini-1.5-pro",
    "gemini-1.5-pro-latest",
    "gemini-1.5-pro-002"
]

_MODEL_CACHE: Dict[str, List[str]] = {}


def get_available_gemini_models(api_key: Optional[str] = None) -> Tuple[List[str], Optional[str]]:
    """
    Dynamically queries Google Generative AI API (ModelService.ListModels) for models authorized
    for this API key that support generateContent.
    Returns (models_list, error_message).
    """
    resolved_key = resolve_api_key(api_key)
    if not resolved_key or len(resolved_key) < 5:
        return [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro"
        ], None

    if resolved_key in _MODEL_CACHE:
        return _MODEL_CACHE[resolved_key], None

    if not GOOGLE_GENAI_AVAILABLE:
        return SUPPORTED_GEMINI_MODELS, None

    try:
        client = genai.Client(api_key=resolved_key)
        discovered = []
        for m in client.models.list():
            actions = getattr(m, 'supported_actions', []) or []
            name = m.name.replace('models/', '') if m.name else ''
            # Filter for generative models (excluding embeddings, imagen, etc.)
            if 'gemini' in name.lower() and (not actions or 'generateContent' in actions):
                discovered.append(name)

        if discovered:
            # Sort prioritized: flash models first, then pro
            def sort_key(name: str):
                name_l = name.lower()
                if "2.5-flash" in name_l:
                    return 0
                if "2.0-flash" in name_l:
                    return 1
                if "1.5-flash" in name_l:
                    return 2
                if "flash" in name_l:
                    return 3
                if "pro" in name_l:
                    return 4
                return 5

            discovered.sort(key=sort_key)
            _MODEL_CACHE[resolved_key] = discovered
            return discovered, None

        return SUPPORTED_GEMINI_MODELS, None
    except Exception as e:
        return [], str(e)


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
    model_name: str = "gemini-2.5-flash",
    temperature: float = 0.2,
    timeout: float = 15.0,
    max_retries: int = 1,
    api_version: Optional[str] = None
):
    """
    Instantiates ChatGoogleGenerativeAI client with specified model, low latency timeouts, and API key.
    """
    if not GEMINI_LIB_AVAILABLE:
        return None

    key = resolve_api_key(api_key)
    if not key:
        return None

    # Sanitize model name (strip descriptions like "(Fastest)")
    clean_model = model_name.split()[0].strip() if model_name else "gemini-2.5-flash"

    try:
        kwargs: Dict[str, Any] = {
            "model": clean_model,
            "google_api_key": key,
            "temperature": temperature,
            "max_output_tokens": 768,
            "timeout": timeout,
            "max_retries": max_retries
        }
        if api_version:
            kwargs["api_version"] = api_version
        return ChatGoogleGenerativeAI(**kwargs)
    except Exception:
        return None


def generate_gemini_clinical_review(
    protocol: Dict[str, Any],
    ehr: Dict[str, Any],
    match_results: List[Dict[str, Any]],
    safety_alerts: List[str],
    deviation_warnings: List[str],
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.5-flash"
) -> str:
    """
    Uses Gemini LLM to synthesize an expert Clinical Safety Medical Monitor review,
    analyzing patient physiology, drug contraindications, and protocol deviations.
    Automatically queries authorized models and tries fallback models if the requested model returns NOT_FOUND.
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

    # Sanitize requested model name
    clean_requested = model_name.split()[0].strip() if model_name else "gemini-2.5-flash"

    # Query dynamic available models for this specific API key
    discovered_models, list_err = get_available_gemini_models(resolved_key)

    if discovered_models:
        candidate_pool = [clean_requested] + [m for m in discovered_models if m != clean_requested]
    else:
        candidate_pool = [
            clean_requested,
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash-002",
            "gemini-1.5-pro",
            "gemini-1.5-pro-latest"
        ]

    # Deduplicate candidate pool
    models_to_try: List[str] = []
    for m in candidate_pool:
        if m not in models_to_try:
            models_to_try.append(m)

    last_error = None
    for target_model in models_to_try:
        # First attempt with default API version
        for api_ver in [None, "v1"]:
            try:
                llm = get_gemini_llm(
                    api_key=resolved_key,
                    model_name=target_model,
                    timeout=12.0,
                    max_retries=1,
                    api_version=api_ver
                )
                if not llm:
                    continue
                response = llm.invoke(messages)
                text_result = extract_text_content(response.content)
                if text_result:
                    return text_result
            except Exception as e:
                last_error = e
                err_str = str(e)
                # If model is not found or timed out, proceed to next variant or candidate immediately
                if any(indicator in err_str.upper() for indicator in ["NOT_FOUND", "404", "TIMEOUT", "DEADLINE"]):
                    continue
                # For non-404 errors (e.g., quota limits or invalid keys), return early with diagnosis
                if "INVALID_ARGUMENT" in err_str or "API_KEY_INVALID" in err_str:
                    return "Gemini API Error: The provided Gemini API key is invalid. Please verify and re-enter your key."
                if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                    return "Gemini API Quota Exceeded (429): Your free tier request rate limit was reached. Please wait a minute and re-try."
                return f"Gemini API execution error ({target_model}): {err_str}"

    if last_error and ("404" in str(last_error) or "NOT_FOUND" in str(last_error)):
        return (
            "Gemini API Error (404 NOT_FOUND): None of the requested Gemini models were found for this API key. "
            "This usually occurs when the Google Cloud Project does not have the 'Generative Language API' enabled, "
            "or when using an API key from an enterprise/restricted GCP project. "
            "To fix this, please create a free API key directly at Google AI Studio (https://aistudio.google.com/app/apikey) "
            "where Generative Language API model endpoints are enabled by default."
        )

    return f"Gemini API error: {str(last_error)}. Please verify your Gemini API key and model permissions."
