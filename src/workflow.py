"""
Multi-Agent Workflow Orchestrator using LangGraph.
Connects Planner -> Research -> Executor -> Critic -> END in a stateful pipeline.
"""
import os
from typing import Dict, Any, Generator, Tuple, Optional
from src.state import AgentState
from src.agents.planner import PlannerAgent
from src.agents.research import ResearchAgent
from src.agents.executor import ExecutorAgent
from src.agents.critic import CriticAgent

# Try importing LangGraph components
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    END = "__end__"


def _build_langgraph_workflow():
    """
    Builds and compiles the LangGraph StateGraph connecting all 4 clinical trial agents.
    """
    planner_agent = PlannerAgent()
    research_agent = ResearchAgent()
    executor_agent = ExecutorAgent()
    critic_agent = CriticAgent()

    workflow = StateGraph(AgentState)

    # Add agent nodes
    workflow.add_node("planner", planner_agent.run)
    workflow.add_node("research", research_agent.run)
    workflow.add_node("executor", executor_agent.run)
    workflow.add_node("critic", critic_agent.run)

    # Establish linear deterministic flow
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "research")
    workflow.add_edge("research", "executor")
    workflow.add_edge("executor", "critic")
    workflow.add_edge("critic", END)

    return workflow.compile()


# Compile once at module level if LangGraph is available
_COMPILED_APP = _build_langgraph_workflow() if LANGGRAPH_AVAILABLE else None


def run_audit_workflow(
    patient_raw_data: Dict[str, Any],
    protocol: Dict[str, Any],
    gemini_api_key: Optional[str] = None
) -> AgentState:
    """
    Synchronously runs the complete multi-agent clinical trial auditor workflow.
    """
    resolved_key = (
        gemini_api_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or ""
    )

    initial_state: AgentState = {
        "patient_id": patient_raw_data.get("patient_id", "UNKNOWN"),
        "patient_raw_data": patient_raw_data,
        "protocol": protocol,
        "parsed_rules": [],
        "normalized_ehr": {},
        "match_results": [],
        "eligibility_score": 0.0,
        "critic_approved": False,
        "deviation_warnings": [],
        "safety_alerts": [],
        "llm_safety_narrative": "",
        "gemini_api_key": resolved_key,
        "audit_logs": []
    }

    if LANGGRAPH_AVAILABLE and _COMPILED_APP is not None:
        try:
            final_state = _COMPILED_APP.invoke(initial_state)
            return final_state
        except Exception:
            # Fallback to pure Python state machine if invoke fails
            pass

    # Pure Python state machine fallback
    return _run_pure_python_pipeline(initial_state)


def _run_pure_python_pipeline(state: AgentState) -> AgentState:
    """
    Reliable pure Python sequential fallback executing the 4 agents in order.
    """
    state = PlannerAgent().run(state)
    state = ResearchAgent().run(state)
    state = ExecutorAgent().run(state)
    state = CriticAgent().run(state)
    return state


def run_audit_workflow_streaming(
    patient_raw_data: Dict[str, Any],
    protocol: Dict[str, Any],
    gemini_api_key: Optional[str] = None
) -> Generator[Tuple[str, str, AgentState], None, AgentState]:
    """
    Generator that yields (agent_name, step_description, current_state) at each node execution,
    enabling real-time animated step-by-step progress tracking in UI environments like Streamlit.
    """
    resolved_key = (
        gemini_api_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or ""
    )

    state: AgentState = {
        "patient_id": patient_raw_data.get("patient_id", "UNKNOWN"),
        "patient_raw_data": patient_raw_data,
        "protocol": protocol,
        "parsed_rules": [],
        "normalized_ehr": {},
        "match_results": [],
        "eligibility_score": 0.0,
        "critic_approved": False,
        "deviation_warnings": [],
        "safety_alerts": [],
        "llm_safety_narrative": "",
        "gemini_api_key": resolved_key,
        "audit_logs": []
    }

    # Step 1: Planner
    yield ("planner", "Deconstructing clinical protocol into evaluable Boolean rules...", state)
    state = PlannerAgent().run(state)
    yield ("planner_done", f"Deconstructed {len(state.get('parsed_rules', []))} inclusion & exclusion rules.", state)

    # Step 2: Research
    yield ("research", "Normalizing unstructured EHR parameters & medication taxonomy...", state)
    state = ResearchAgent().run(state)
    yield ("research_done", "EHR data cleaned and mapped to LOINC/SNOMED canonical schemas.", state)

    # Step 3: Executor
    yield ("executor", "Evaluating criteria deterministically and calculating eligibility score...", state)
    state = ExecutorAgent().run(state)
    yield ("executor_done", f"Criteria evaluation complete. Score: {state.get('eligibility_score', 0.0)}%.", state)

    # Step 4: Critic
    yield ("critic", "Auditing safety boundaries, drug contraindications & protocol deviations...", state)
    state = CriticAgent().run(state)
    status_str = "APPROVED" if state.get("critic_approved") else "REJECTED"
    yield ("critic_done", f"Safety audit complete. Critic Verdict: {status_str}.", state)

    return state
