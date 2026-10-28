"""
State definition for the Clinical Trial Auditor multi-agent workflow.
"""
from typing import TypedDict, List, Dict, Any


class AgentState(TypedDict, total=False):
    """
    Central state container passed between agents in the LangGraph workflow.
    """
    patient_id: str
    patient_raw_data: Dict[str, Any]
    protocol: Dict[str, Any]
    parsed_rules: List[Dict[str, Any]]
    normalized_ehr: Dict[str, Any]
    match_results: List[Dict[str, Any]]
    eligibility_score: float
    critic_approved: bool
    deviation_warnings: List[str]
    safety_alerts: List[str]
    llm_safety_narrative: str
    gemini_api_key: str
    gemini_model: str
    audit_logs: List[str]
