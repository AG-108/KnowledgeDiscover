"""Tests for the generic LLM-SR adapter without requiring an external LLM."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kd.model.kd_llmsr import KD_LLMSR


class _Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {"content": ["def equation_v1(x1, p0):\n    return p0 * x1 ** 2"]}


def test_llmsr_fits_sampled_structure_and_constants(monkeypatch):
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return _Response()

    monkeypatch.setattr("kd.model.kd_llmsr.requests.post", post)
    X = np.linspace(-2, 2, 41).reshape(-1, 1)
    y = 3.5 * X[:, 0] ** 2
    model = KD_LLMSR(
        endpoint="http://localhost:5000/completions",
        max_samples=1,
        samples_per_prompt=1,
        optimizer_restarts=1,
        random_state=7,
    ).fit(X, y, variable_names=["x1"])

    np.testing.assert_allclose(model.predict(X), y, rtol=1e-6, atol=1e-8)
    assert "x1 ** 2" in model.best_expression_
    assert model.search_stats_ == {"sampled": 1, "accepted": 1, "rejected": 0, "islands": 2}
    assert calls[0][0] == "http://localhost:5000/completions"
    assert calls[0][1]["timeout"] == 120.0
    assert "Representative training observations" in calls[0][1]["json"]["prompt"]


def test_llmsr_requires_endpoint_before_search(monkeypatch):
    monkeypatch.delenv("LLMSR_ENDPOINT", raising=False)
    with pytest.raises(RuntimeError, match="LLMSR_ENDPOINT"):
        KD_LLMSR(max_samples=1).fit([[0], [1]], [0, 1])


def test_llmsr_rejects_unsafe_generated_code(monkeypatch):
    model = KD_LLMSR(endpoint="http://localhost", max_samples=1)
    with pytest.raises(ValueError, match="unsupported function"):
        model._compile_expression("np.load('secret.npy')", ["x1"])
