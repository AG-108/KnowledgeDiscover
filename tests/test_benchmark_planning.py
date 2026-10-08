import pytest

from kd.benchmark_execution import execution_policy
from kd.benchmark_planning import plan_chunks


def test_chunks_cover_each_case_once_keep_llm_exclusive_and_bound_all_phases():
    cases = [{"name": f"case_{i}", "model": model, "task": task}
             for i, (model, task) in enumerate([("gplearn", "sr"), ("sindy", "ode"),
                                               ("dso", "sr"), ("llmsr", "ode")])]
    counts = {case["name"]: 2 if case["task"] == "ode" else 1 for case in cases}
    policy = execution_policy(300, setup_timeout=20, score_timeout=30)
    plan = plan_chunks(cases, counts, policy, max_hours=1, overhead_hours=0.1)
    assert sorted(name for chunk in plan["chunks"] for name in chunk["cases"]) == sorted(counts)
    assert all(chunk["planned_hours"] <= 1 for chunk in plan["chunks"])
    llm = next(chunk for chunk in plan["chunks"] if chunk["queue"] == "gpu_llmsr")
    assert llm["cases"] == ["case_3"]
    assert llm["phase_budget_hours"] * 3600 == 20 + 2 * (20 + 300 + 5 + 30) + 20 + 30
    assert plan == plan_chunks(list(reversed(cases)), counts, policy, max_hours=1, overhead_hours=0.1)


def test_unbounded_or_oversized_case_cannot_be_assigned_a_false_40h_bound():
    cases = [{"name": "long", "model": "gplearn", "task": "sr"}]
    with pytest.raises(ValueError, match="Finite T"):
        plan_chunks(cases, {"long": 1}, execution_policy())
    with pytest.raises(ValueError, match="One case exceeds"):
        plan_chunks(cases, {"long": 100}, execution_policy(3600))
    with pytest.raises(ValueError, match="40 hours"):
        plan_chunks(cases, {"long": 1}, execution_policy(60), max_hours=48)
