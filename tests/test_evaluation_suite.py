"""Pytest integration for the automated evaluation benchmark suite."""

import pytest
from src.evaluation import EvaluationSuite


@pytest.fixture
def eval_suite():
    return EvaluationSuite()


def test_full_evaluation_benchmark_all_cases_pass(eval_suite):
    """Run all 27 visible and adversarial cases and assert 100% pass rate."""
    summary = eval_suite.run_all()
    assert summary["total_cases"] == 27
    assert summary["failed_cases"] == 0, f"Evaluation failures: {[r for r in summary['results'] if not r['passed']]}"
    assert summary["pass_rate_percent"] == 100.0
