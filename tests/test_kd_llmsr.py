"""Tests for the generic LLM-SR adapter without requiring an external LLM."""

import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kd.model.kd_llmsr import (
    DEFAULT_BACKBONE,
    KD_LLMSR,
    _extract_body,
    _inline_body,
    _Island,
    _Program,
    _trim_function_body,
)

ENDPOINT = "http://127.0.0.1:10005/v1/chat/completions"


class _Response:
    def __init__(self, content, status=200):
        self.content = content
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)

    def json(self):
        return {
            "model": DEFAULT_BACKBONE,
            "choices": [{"message": {"content": self.content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }


def _fake_endpoint(monkeypatch, content, status=200):
    calls = []
    lock = threading.Lock()

    def post(url, **kwargs):
        with lock:
            calls.append((url, kwargs))
        return _Response(content, status)

    monkeypatch.setattr("kd.model.kd_llmsr.requests.post", post)
    return calls


def test_llmsr_defaults_follow_llm_srbench_protocol(monkeypatch):
    monkeypatch.delenv("LLMSR_MODEL", raising=False)
    model = KD_LLMSR(endpoint=ENDPOINT)
    assert model._connection() == (ENDPOINT, "meta-llama/Llama-3.1-8B-Instruct")
    assert (model.temperature, model.samples_per_prompt, model.num_evaluators) == (0.8, 4, 4)
    assert (model.evaluation_timeout, model.num_islands, model.functions_per_prompt) == (30, 10, 2)
    assert (model.max_parameters, model.max_llm_calls) == (10, 1000)


def test_llmsr_fits_multiline_program_with_bfgs(monkeypatch):
    content = (
        "Here is an improved version:\n```python\n"
        "def equation_v2(x1: np.ndarray, params: np.ndarray) -> np.ndarray:\n"
        '    """Quadratic law."""\n'
        "    a, b = params[0], params[1]\n"
        "    output = a * x1 ** 2\n"
        "    output += b\n"
        "    return output\n```\nThis adds a quadratic term."
    )
    calls = _fake_endpoint(monkeypatch, content)
    X = np.linspace(-2, 2, 41).reshape(-1, 1)
    y = 3.5 * X[:, 0] ** 2 + 0.5
    model = KD_LLMSR(endpoint=ENDPOINT, max_llm_calls=4, random_state=7)
    model.fit(X, y, variable_names=["x1"], target_name="y")

    np.testing.assert_allclose(model.predict(X), y, rtol=1e-5, atol=1e-6)
    assert "x1 ** 2" in model.best_expression_
    assert "output += b" in model.best_program_
    stats = model.search_stats_
    assert (stats["llm_calls"], stats["prompts"], stats["accepted"]) == (4, 1, 4)
    assert stats["served_models"] == [DEFAULT_BACKBONE]
    assert stats["completion_tokens"] == 20
    payloads = [kwargs["json"] for _, kwargs in calls]
    assert all(url == ENDPOINT for url, _ in calls)
    assert {payload["n"] for payload in payloads} == {1}
    assert {payload["temperature"] for payload in payloads} == {0.8}
    assert {payload["max_tokens"] for payload in payloads} == {1024}
    assert {payload["model"] for payload in payloads} == {DEFAULT_BACKBONE}
    assert len({payload["seed"] for payload in payloads}) == 4
    prompt = payloads[0]["messages"][0]["content"]
    assert prompt.startswith("You are a helpful assistant")
    assert "MAX_NPARAMS = 10" in prompt
    assert "def equation_v0(x1: np.ndarray, params: np.ndarray) -> np.ndarray:" in prompt
    assert "output = params[0] * x1 + params[1]" in prompt
    assert prompt.rstrip().endswith('"""Improved version of `equation_v0`."""')
    assert "training observations" not in prompt


def test_llmsr_counts_every_call_against_the_budget(monkeypatch):
    calls = _fake_endpoint(monkeypatch, "def equation(x1, params):\n    return params[0] * x1")
    X = np.linspace(-2, 2, 21).reshape(-1, 1)
    model = KD_LLMSR(endpoint=ENDPOINT, max_llm_calls=10, num_islands=2, random_state=0)
    model.fit(X, X[:, 0] ** 2)
    assert len(calls) == 10
    assert (model.search_stats_["llm_calls"], model.search_stats_["prompts"]) == (10, 3)
    assert model.search_stats_["llm_call_budget"] == 10
    assert model.stop_reason_ == "budget_exhausted"


def test_llmsr_stops_when_initial_skeleton_is_exact(monkeypatch):
    calls = _fake_endpoint(monkeypatch, "unused")
    X = np.linspace(-1, 1, 11).reshape(-1, 1)
    model = KD_LLMSR(endpoint=ENDPOINT, max_llm_calls=8).fit(X, X[:, 0] + 1)
    assert calls == []
    assert model.search_stats_["stopped_on_zero_loss"] is True
    assert model.stop_reason_ == "early_stop"
    assert model.search_stats_["best_call_index"] is None


def test_llmsr_does_not_retry_client_errors(monkeypatch):
    calls = _fake_endpoint(monkeypatch, "", status=404)
    X = np.linspace(-1, 1, 11).reshape(-1, 1)
    with pytest.raises(RuntimeError, match="after 1 attempt"):
        KD_LLMSR(endpoint=ENDPOINT, max_llm_calls=1, samples_per_prompt=1).fit(X, X[:, 0] ** 3)
    assert len(calls) == 1


def test_llmsr_requires_endpoint_before_search(monkeypatch):
    monkeypatch.delenv("LLMSR_ENDPOINT", raising=False)
    with pytest.raises(RuntimeError, match="LLMSR_ENDPOINT"):
        KD_LLMSR(max_llm_calls=1).fit([[0], [1]], [0, 1])


def test_llmsr_inlines_restricted_program_bodies():
    sample = (
        "def equation_v1(x, v, params):\n"
        "    import numpy as np\n"
        "    k, c = params[0:2]\n"
        "    force: float = -k * x\n"
        "    force -= c * v\n"
        "    return force + np.sin(params[2] * x)\n"
        "print('trailing text is trimmed')"
    )
    body = _trim_function_body(_extract_body(sample))
    assert "print" not in body
    expression = _inline_body(body)
    model = KD_LLMSR(endpoint=ENDPOINT)
    compiled, _, count = model._compile_expression(expression, ["x", "v"])
    assert count == 3
    assert compiled == "-p0 * x - p1 * v + np.sin(p2 * x)"


@pytest.mark.parametrize(
    "body",
    [
        "    return params[10] * x1",
        "    for i in range(3):\n        x1 = x1 * 2\n    return x1",
        "    return np.load('secret.npy')",
        "    return __import__('os').getcwd()",
        "    a = x1\n" + "    a = a * a\n" * 40 + "    return a",
    ],
)
def test_llmsr_rejects_programs_outside_the_protocol(body):
    model = KD_LLMSR(endpoint=ENDPOINT)
    with pytest.raises(ValueError):
        model._compile_expression(_inline_body(body), ["x1"])


def test_llmsr_evaluator_enforces_timeout():
    model = KD_LLMSR(endpoint=ENDPOINT, evaluation_timeout=0.05, num_evaluators=2)
    model.variable_names_ = ["x1"]
    X = np.linspace(0.1, 1, 200_000).reshape(-1, 1)
    slow = " + ".join(f"p{i} * np.exp(np.sin(p{i} * x1))" for i in range(10))
    results = model._evaluate_parallel([(slow, 10), (slow, 10)], X, np.cos(X[:, 0]))
    assert [status for status, _, _ in results] == ["timeout", "timeout"]


def test_llmsr_island_prompt_orders_examples_from_worst_to_best():
    island = _Island()
    for mse in (4.0, 1.0, 9.0):
        island.register(_Program("    return params[0]", "p0", np.ones(1), mse, None))
    examples = island.examples(np.random.default_rng(0), 2, 0.1, 30000)
    assert len(examples) == 2
    assert [program.mse for program in examples] == sorted(
        (program.mse for program in examples), reverse=True
    )
