"""Offline API contracts only; these tests do not validate an official checkpoint."""

from types import SimpleNamespace
import hashlib
import numpy as np
import pytest
import sympy as sp

from kd.model import kd_e2e


def _node(value, *children):
    return SimpleNamespace(value=value, children=children)


def test_missing_assets_do_not_train_or_download():
    with pytest.raises(kd_e2e.E2EUnavailable, match="explicit official"):
        kd_e2e.E2ETransformerModel().check_available()


def test_checkpoint_requires_explicit_trust_and_hash(tmp_path):
    source = tmp_path / "source"
    module = source / "symbolicregression/model/sklearn_wrapper.py"
    module.parent.mkdir(parents=True)
    module.write_text("# mock source", encoding="utf-8")
    checkpoint = tmp_path / "mock.pt"
    checkpoint.write_bytes(b"not a real checkpoint")
    model = kd_e2e.E2ETransformerModel(checkpoint_path=checkpoint, source_dir=source)
    with pytest.raises(kd_e2e.E2EUnavailable, match="trust_checkpoint"):
        model.check_available()
    model.trust_checkpoint = True
    with pytest.raises(kd_e2e.E2EUnavailable, match="SHA256"):
        model.check_available()
    model.checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert model.check_available() == (checkpoint, source)


def test_tree_conversion_preserves_original_feature_mapping():
    tree = _node("add", _node("pow2", _node("x_2")), _node("sin", _node("x_0")))
    expression = kd_e2e._tree_expression(tree, ["first", "unused", "last"])
    first, last = sp.symbols("first last")
    assert sp.simplify(expression - last**2 - sp.sin(first)) == 0


def test_fit_uses_official_relabeled_tree_and_preserves_seed(monkeypatch, tmp_path):
    module = tmp_path / "symbolicregression/model/sklearn_wrapper.py"
    module.parent.mkdir(parents=True)
    module.write_text("# mock source", encoding="utf-8")
    tree = _node("mul", _node("2"), _node("x_1"))

    class OfficialModel:
        env = SimpleNamespace(simplifier=SimpleNamespace(
            tree_to_numexpr_fn=lambda selected: lambda X: 2 * X[:, 1:2]))

        def __call__(self, inputs):
            return [tree]

    class Estimator:
        def __init__(self, model, **kwargs):
            self.model = model

        def fit(self, X, y, verbose=False):
            self.model(X)

        def retrieve_tree(self, with_infos=False):
            assert with_infos
            return {"relabed_predicted_tree": tree, "refinement_type": "NoRef"}

    monkeypatch.setattr(kd_e2e, "_load_official", lambda *args: (OfficialModel(), Estimator))
    model = kd_e2e.E2ETransformerModel()
    monkeypatch.setattr(model, "check_available", lambda: (tmp_path / "mock.pt", tmp_path))
    np.random.seed(143)
    expected = np.random.rand()
    np.random.seed(143)
    X = np.arange(12).reshape(6, 2)
    model.fit(X, 2 * X[:, 1], variable_names=["a", "b"])
    assert np.random.rand() == expected
    assert model.best_expression_ == "2.0*b"
    np.testing.assert_array_equal(model.predict(X), 2 * X[:, 1])
    assert model.provenance_["pretraining_seconds"] is None
    assert model.provenance_["inference_seconds"] >= 0
