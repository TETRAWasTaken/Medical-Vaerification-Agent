"""
Executor Agent: Deterministically evaluates trial rules against normalized EHR data.
"""
from typing import Dict, Any, List, Tuple
from datetime import datetime, timezone
from src.state import AgentState


class ExecutorAgent:
    """
    Evaluates every parsed protocol criterion against normalized EHR values,
    computes deterministic rule outcomes, and calculates overall eligibility score.
    """

    def __init__(self):
        self.name = "ExecutorAgent"

    @staticmethod
    def _evaluate_condition(actual: Any, operator: str, target: Any) -> Tuple[bool, str]:
        """
        Evaluates actual EHR value against target value with operator.
        Returns (passed: bool, reason: str).
        """
        op = operator.strip().upper()

        # Handle list fields (e.g. medications)
        if isinstance(actual, list):
            target_str = str(target).upper().strip()
            # Check if any list element contains target substring
            contains_target = any(target_str in str(item).upper() for item in actual)

            if op in ("NOT_CONTAINS", "NOT_IN"):
                passed = not contains_target
                reason = f"Target '{target}' {'was NOT detected' if passed else 'WAS DETECTED'} in medication regimen."
                return passed, reason
            elif op in ("CONTAINS", "IN"):
                passed = contains_target
                reason = f"Target '{target}' {'was found' if passed else 'was NOT found'} in medication regimen."
                return passed, reason

        # Handle boolean fields
        if isinstance(actual, bool) or isinstance(target, bool):
            actual_bool = bool(actual)
            target_bool = bool(target)
            if op in ("==", "="):
                passed = (actual_bool == target_bool)
            elif op in ("!=", "<>"):
                passed = (actual_bool != target_bool)
            else:
                passed = (actual_bool == target_bool)
            reason = f"Actual flag is {actual_bool}, required target is {target_bool}."
            return passed, reason

        # Handle numeric fields
        try:
            num_actual = float(actual)
            num_target = float(target)

            if op == ">=":
                passed = (num_actual >= num_target)
            elif op == "<=":
                passed = (num_actual <= num_target)
            elif op == ">":
                passed = (num_actual > num_target)
            elif op == "<":
                passed = (num_actual < num_target)
            elif op in ("==", "="):
                passed = abs(num_actual - num_target) < 1e-6
            elif op in ("!=", "<>"):
                passed = abs(num_actual - num_target) >= 1e-6
            else:
                passed = False
                return False, f"Unsupported operator '{operator}' for numeric comparison."

            status_str = "meets" if passed else "fails"
            reason = f"Observed value {num_actual} {status_str} required threshold ({operator} {num_target})."
            return passed, reason
        except (ValueError, TypeError):
            pass

        # String fallback
        str_actual = str(actual).strip().lower()
        str_target = str(target).strip().lower()

        if op in ("==", "="):
            passed = (str_actual == str_target)
        elif op in ("!=", "<>"):
            passed = (str_actual != str_target)
        elif op == "CONTAINS":
            passed = (str_target in str_actual)
        elif op == "NOT_CONTAINS":
            passed = (str_target not in str_actual)
        else:
            passed = False
            return False, f"Could not evaluate operator '{operator}' between '{actual}' and '{target}'."

        reason = f"Value '{actual}' {'satisfies' if passed else 'does not satisfy'} criterion."
        return passed, reason

    def run(self, state: AgentState) -> AgentState:
        """
        Executes deterministic matching of rules against normalized EHR data.
        """
        rules = state.get("parsed_rules", [])
        ehr = state.get("normalized_ehr", {})
        audit_logs = list(state.get("audit_logs", []))

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        audit_logs.append(f"[{now_str}] [ExecutorAgent] Starting rule evaluation across {len(rules)} criteria.")

        match_results: List[Dict[str, Any]] = []
        passed_count = 0

        for rule in rules:
            field = rule["field"]
            operator = rule["operator"]
            target_value = rule["target_value"]
            criterion_type = rule["criterion_type"]

            # Lookup actual value in normalized EHR
            actual_value = ehr.get(field)

            passed, explanation = self._evaluate_condition(actual_value, operator, target_value)

            if passed:
                passed_count += 1
                status_badge = "PASS"
            else:
                status_badge = "FAIL"

            result_entry = {
                "rule_id": rule["rule_id"],
                "criterion_type": criterion_type,
                "description": rule["description"],
                "field": field,
                "operator": operator,
                "target_value": target_value,
                "unit": rule.get("unit", ""),
                "actual_value": actual_value,
                "passed": passed,
                "status": status_badge,
                "rationale": rule.get("rationale", ""),
                "clinical_explanation": explanation
            }
            match_results.append(result_entry)

            audit_logs.append(
                f"[{now_str}] [ExecutorAgent] Rule {rule['rule_id']} [{criterion_type}] => {status_badge}: "
                f"{field} (Actual: {actual_value}) {operator} {target_value}. Explanation: {explanation}"
            )

        total_rules = len(rules)
        if total_rules > 0:
            eligibility_score = round((passed_count / total_rules) * 100.0, 1)
        else:
            eligibility_score = 0.0

        audit_logs.append(
            f"[{now_str}] [ExecutorAgent] Evaluation Complete: {passed_count}/{total_rules} rules satisfied. "
            f"Eligibility Score = {eligibility_score}%."
        )

        return {
            **state,
            "match_results": match_results,
            "eligibility_score": eligibility_score,
            "audit_logs": audit_logs
        }
