"""
Planner Agent: Deconstructs clinical trial protocols into machine-evaluable Boolean rules.
"""
from typing import Dict, Any, List
from datetime import datetime, timezone
from src.state import AgentState


class PlannerAgent:
    """
    Deconstructs clinical trial inclusion and exclusion criteria into structured,
    deterministic machine-evaluable JSON rules.
    """

    def __init__(self):
        self.name = "PlannerAgent"

    def run(self, state: AgentState) -> AgentState:
        """
        Executes protocol parsing step within the multi-agent graph.
        """
        protocol = state.get("protocol", {})
        protocol_id = protocol.get("protocol_id", "UNKNOWN_PROTOCOL")
        audit_logs = list(state.get("audit_logs", []))

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        audit_logs.append(f"[{now_str}] [PlannerAgent] Initiating protocol rule deconstruction for {protocol_id}.")

        parsed_rules: List[Dict[str, Any]] = []

        # Parse Inclusion Criteria
        inclusions = protocol.get("inclusion_criteria", [])
        for item in inclusions:
            rule = {
                "rule_id": item.get("id", f"INC-{len(parsed_rules)+1:02d}"),
                "criterion_type": "INCLUSION",
                "field": item.get("field", "").strip().lower(),
                "operator": item.get("operator", "==").strip(),
                "target_value": item.get("target_value"),
                "unit": item.get("unit", ""),
                "description": item.get("description", ""),
                "rationale": item.get("rationale", "Mandatory inclusion criterion for cohort definition.")
            }
            parsed_rules.append(rule)
            audit_logs.append(
                f"[{now_str}] [PlannerAgent] Synthesized Rule {rule['rule_id']} (INCLUSION): "
                f"{rule['field']} {rule['operator']} {rule['target_value']} ({rule['unit']})"
            )

        # Parse Exclusion Criteria
        exclusions = protocol.get("exclusion_criteria", [])
        for item in exclusions:
            rule = {
                "rule_id": item.get("id", f"EXC-{len(parsed_rules)+1:02d}"),
                "criterion_type": "EXCLUSION",
                "field": item.get("field", "").strip().lower(),
                "operator": item.get("operator", "==").strip(),
                "target_value": item.get("target_value"),
                "unit": item.get("unit", ""),
                "description": item.get("description", ""),
                "rationale": item.get("rationale", "Mandatory exclusion criterion for safety and risk avoidance.")
            }
            parsed_rules.append(rule)
            audit_logs.append(
                f"[{now_str}] [PlannerAgent] Synthesized Rule {rule['rule_id']} (EXCLUSION): "
                f"{rule['field']} {rule['operator']} {rule['target_value']} ({rule['unit']})"
            )

        audit_logs.append(
            f"[{now_str}] [PlannerAgent] Completed rule deconstruction: {len(parsed_rules)} total evaluable criteria generated."
        )

        return {
            **state,
            "parsed_rules": parsed_rules,
            "audit_logs": audit_logs
        }
