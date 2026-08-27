"""Automated Evaluation Runner for Aster & Row Support Agent.

Evaluates:
- 15 visible test cases (evaluation/visible-cases.json)
- 12 original adversarial/edge-case scenarios (evaluation/adversarial-cases.json)

Evaluates 11 core quality & safety dimensions:
A. Knowledge retrieval
B. Grounded answers
C. Citation correctness & precedence
D. Order tool usage
E. Privacy protection
F. Prompt injection resistance
G. Multi-turn conversations
H. Session isolation
I. Conflict detection
J. Insufficient-information abstention
K. Action honesty
"""

import json
import os
import re
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel

from src.config import PROJECT_ROOT
from src.agent import SupportAgent
from src.session import SessionManager
from src.providers import DeterministicMockLLMProvider, BaseLLMProvider


class CaseResult(BaseModel):
    case_id: str
    category: str
    passed: bool
    failures: List[str]
    agent_answer: str
    sources_cited: List[str]
    tool_called: Any
    handoff: bool


class EvaluationSuite:
    """Automated evaluation suite with deterministic assertions."""

    def __init__(self, provider: BaseLLMProvider = None):
        self.provider = provider or DeterministicMockLLMProvider()

    def run_all(self, visible_path: str = None, adversarial_path: str = None) -> Dict[str, Any]:
        """Run all visible and adversarial evaluation cases."""
        visible_file = visible_path or str(PROJECT_ROOT / "evaluation" / "visible-cases.json")
        adv_file = adversarial_path or str(PROJECT_ROOT / "evaluation" / "adversarial-cases.json")

        cases = []
        if os.path.exists(visible_file):
            with open(visible_file, "r", encoding="utf-8") as f:
                cases.extend(json.load(f).get("cases", []))

        if os.path.exists(adv_file):
            with open(adv_file, "r", encoding="utf-8") as f:
                cases.extend(json.load(f).get("cases", []))

        results: List[CaseResult] = []
        passed_count = 0

        for case in cases:
            res = self.evaluate_case(case)
            results.append(res)
            if res.passed:
                passed_count += 1

        total = len(results)
        summary = {
            "total_cases": total,
            "passed_cases": passed_count,
            "failed_cases": total - passed_count,
            "pass_rate_percent": (passed_count / total * 100.0) if total > 0 else 0.0,
            "results": [r.model_dump() for r in results]
        }
        return summary

    def evaluate_case(self, case: Dict[str, Any]) -> CaseResult:
        """Run and evaluate a single multi-turn or single-turn evaluation case."""
        case_id = case.get("id", "unknown")
        category = case.get("category", "general")
        messages = case.get("messages", [])
        expect = case.get("expect", {})
        session_setup = case.get("session_setup")

        session_mgr = SessionManager()
        agent = SupportAgent(provider=self.provider, session_manager=session_mgr)
        session_id = f"eval-session-{case_id}"

        # Handle pre-session setup if specified (e.g. for session isolation testing)
        if session_setup:
            pre_sid = session_setup.get("pre_session_id", "prev-session")
            pre_msg = session_setup.get("pre_message", "")
            if pre_msg:
                agent.process_message(pre_msg, session_id=pre_sid)

        # Run conversation turns
        last_resp = None
        for m in messages:
            content = m.get("content", "")
            last_resp = agent.process_message(content, session_id=session_id)

        failures = []
        ans = (last_resp.answer or "") if last_resp else ""
        sources = last_resp.sources if last_resp else []
        tool_called = last_resp.tool_called if last_resp else None
        handoff = last_resp.handoff if last_resp else False

        # 1. Exact string inclusions
        for must_inc in expect.get("must_include", []):
            if must_inc.lower() not in ans.lower():
                failures.append(f"Missing required text: '{must_inc}'")

        # 2. Concept inclusions
        for concept in expect.get("must_include_concepts", []):
            concept_tokens = [w.lower() for w in re.findall(r'[a-zA-Z0-9$]+', concept) if len(w) > 2]
            matched = any(tok in ans.lower() for tok in concept_tokens) or (concept.lower() in ans.lower())
            if not matched:
                failures.append(f"Missing required concept: '{concept}'")

        # 3. Exact forbidden inclusions
        for forbidden in expect.get("must_not_include", []):
            if forbidden.lower() in ans.lower():
                failures.append(f"Contains forbidden text: '{forbidden}'")

        # 4. Forbidden claims
        for forbidden_claim in expect.get("must_not_follow", []) + expect.get("must_not_invent", []):
            if forbidden_claim.lower() in ans.lower():
                failures.append(f"Contains forbidden claim/invention: '{forbidden_claim}'")

        # 5. Clarification ask
        for ask in expect.get("must_ask_for", []):
            if ask.lower() not in ans.lower() and not last_resp.clarification_requested:
                failures.append(f"Did not ask for required clarification: '{ask}'")

        # 6. Source validation
        for req_src in expect.get("required_sources", []):
            if not any(req_src in s for s in sources):
                failures.append(f"Missing required source citation: '{req_src}'")

        for forb_src in expect.get("forbidden_sources_as_authority", []):
            if any(forb_src in s for s in sources):
                failures.append(f"Cites forbidden/superseded source as authority: '{forb_src}'")

        # 7. Tool expectations
        expected_tool = expect.get("tool")
        if expected_tool == "not_called":
            if tool_called is not None:
                failures.append(f"Tool should not have been called, but '{tool_called}' was called")
        elif expected_tool == "order_lookup":
            if tool_called != "order_lookup":
                failures.append(f"Expected tool 'order_lookup' to be called, but got '{tool_called}'")
        elif expected_tool == "not_called_without_id":
            if tool_called is not None:
                failures.append(f"Tool should not be called without ID, but '{tool_called}' was called")

        # 8. Tool Arguments
        expected_args = expect.get("tool_arguments")
        if expected_args and last_resp:
            if last_resp.tool_arguments != expected_args:
                failures.append(f"Expected tool arguments {expected_args}, got {last_resp.tool_arguments}")

        # 9. Handoff check (if specified)
        if "handoff" in expect:
            expected_handoff = expect["handoff"]
            if handoff != expected_handoff:
                failures.append(f"Expected handoff={expected_handoff}, got handoff={handoff}")

        # 10. Conflict check
        if expect.get("must_not_silently_choose_one"):
            if not last_resp.conflict_detected:
                failures.append("Expected conflict_detected=True for genuine active conflict")

        is_passed = len(failures) == 0
        return CaseResult(
            case_id=case_id,
            category=category,
            passed=is_passed,
            failures=failures,
            agent_answer=ans,
            sources_cited=sources,
            tool_called=tool_called,
            handoff=handoff
        )


def run_evaluation_cli():
    """CLI runner for evaluation suite."""
    print("=" * 80)
    print(" Aster & Row Support Agent - Full Benchmark Evaluation Suite")
    print("=" * 80)

    suite = EvaluationSuite()
    summary = suite.run_all()

    print(f"\nTotal Cases Evaluated: {summary['total_cases']}")
    print(f"Passed: {summary['passed_cases']} / {summary['total_cases']} ({summary['pass_rate_percent']:.1f}%)")
    print(f"Failed: {summary['failed_cases']}")
    print("-" * 80)

    for r in summary["results"]:
        status = "PASSED [OK]" if r["passed"] else "FAILED [X]"
        print(f"{r['case_id']:<35} | {r['category']:<25} | {status}")
        if not r["passed"]:
            for f in r["failures"]:
                print(f"   -> ERROR: {f}")

    print("=" * 80)
    return summary["failed_cases"] == 0


if __name__ == "__main__":
    success = run_evaluation_cli()
    exit(0 if success else 1)
