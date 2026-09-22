import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

torch = pytest.importorskip("torch")

from kd.model.dso.core import DeepSymbolicOptimizer, load_default_config
from kd.model.dso.memory import Batch
from kd.model.dso.policy import RNNPolicy, safe_cross_entropy
from kd.model.dso.prior import make_prior
from kd.model.dso.program import Program, from_str_tokens, from_tokens
from kd.model.dso.state_manager import make_state_manager
from kd.model.dso.task.regression.regression import RegressionTask, make_regression_metric

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task(function_set=("add", "sub", "mul", "div"), n=30, slope=2.0, intercept=0.0):
    """Build a RegressionTask on y = slope*x + intercept and bind it to Program."""
    X = np.linspace(-1.0, 1.0, n).reshape(-1, 1)
    y = slope * X[:, 0] + intercept
    task = RegressionTask(
        function_set=list(function_set), dataset=(X, y), metric="inv_nrmse", metric_params=(1.0,)
    )
    Program.clear_cache()
    Program.set_execute(protected=False)
    Program.set_task(task)
    Program.set_const_optimizer("scipy")
    Program.set_complexity("token")
    return task, X, y


def _make_policy(task, max_length=16, **kwargs):
    cfg = load_default_config()
    prior = make_prior(Program.library, cfg["prior"])
    sm = make_state_manager(cfg["state_manager"])
    defaults = dict(
        max_length=max_length,
        cell="lstm",
        num_layers=1,
        num_units=32,
        initializer="zeros",
        learning_rate=0.005,
        entropy_weight=0.03,
        entropy_gamma=0.7,
    )
    defaults.update(kwargs)
    return RNNPolicy(prior, sm, **defaults)


def _batch_from_policy(policy, n=8):
    actions, obs, priors = policy.sample(n)
    programs = [from_tokens(a) for a in actions]
    r = np.array([p.r for p in programs], dtype=np.float32)
    lengths = np.array([min(len(p.traversal), policy.max_length) for p in programs], dtype=np.int32)
    on_policy = np.array([p.originally_on_policy for p in programs], dtype=np.int32)
    B = Batch(
        actions=actions, obs=obs, priors=priors, lengths=lengths, rewards=r, on_policy=on_policy
    )
    return B, programs, r


# ---------------------------------------------------------------------------
# Library / token construction
# ---------------------------------------------------------------------------


def test_library_built_from_function_set():
    task, X, _ = _make_task(function_set=("add", "sub", "mul", "div"))
    lib = task.library
    # 4 operators + 1 input variable
    assert lib.L == 5
    assert "add" in lib.names and "x1" in lib.names
    assert task.X_train.shape[1] == 1


def test_library_grows_with_input_dim():
    X = np.random.randn(20, 3)
    y = X[:, 0] + X[:, 1]
    task = RegressionTask(
        function_set=["add", "mul"], dataset=(X, y), metric="inv_nrmse", metric_params=(1.0,)
    )
    Program.set_task(task)
    # 2 operators + 3 input variables
    assert task.library.L == 5
    assert {"x1", "x2", "x3"}.issubset(set(task.library.names))


# ---------------------------------------------------------------------------
# Program execution -- hand-checkable numerics
# ---------------------------------------------------------------------------


def test_program_execute_matches_analytic():
    task, X, _ = _make_task()
    x = X[:, 0]

    p_sq = from_str_tokens(["mul", "x1", "x1"], skip_cache=True)
    np.testing.assert_allclose(p_sq.execute(X), x**2, atol=1e-12)

    p_dbl = from_str_tokens(["add", "x1", "x1"], skip_cache=True)
    np.testing.assert_allclose(p_dbl.execute(X), 2 * x, atol=1e-12)

    p_zero = from_str_tokens(["sub", "x1", "x1"], skip_cache=True)
    np.testing.assert_allclose(p_zero.execute(X), np.zeros_like(x), atol=1e-12)


def test_reward_is_one_for_exact_expression():
    """The single most important regression test for this port: the correct
    expression must score the metric's maximum. An earlier attempt at wiring DSO
    into this repo produced reward == 0 for every candidate, because the
    execution path being used expected PDE-shaped inputs."""
    task, X, y = _make_task(slope=2.0, intercept=0.0)
    p = from_str_tokens(["add", "x1", "x1"], skip_cache=True)  # This evaluates to 2 * x.
    assert task.reward_function(p) == pytest.approx(1.0, abs=1e-9)
    info = task.evaluate(p)
    # NOTE: `success` is a numpy bool, so `is True` would fail -- coerce it.
    assert bool(info["success"]) is True
    assert info["nmse_test"] == pytest.approx(0.0, abs=1e-12)


def test_reward_is_lower_for_wrong_expression():
    task, X, y = _make_task(slope=2.0)
    exact = from_str_tokens(["add", "x1", "x1"], skip_cache=True)
    wrong = from_str_tokens(["mul", "x1", "x1"], skip_cache=True)
    assert task.reward_function(wrong) < task.reward_function(exact)
    assert bool(task.evaluate(wrong)["success"]) is False


def test_cython_fallback_selects_python_execute():
    """The compiled cyfunc extension is not built in this port; set_execute must
    fall back to python_execute rather than selecting a broken fast path."""
    _make_task()
    assert Program.have_cython is False


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


# The inv_* metrics take one scaling parameter; the neg_* metrics take none.
# DSO asserts on this arity, so the parameter list is part of each case.
@pytest.mark.parametrize(
    "name,args,expected_max",
    [
        ("inv_nrmse", (1.0,), 1.0),
        ("inv_nmse", (1.0,), 1.0),
        ("inv_mse", (1.0,), 1.0),
        ("neg_nrmse", (), 0.0),
        ("neg_nmse", (), 0.0),
        ("neg_mse", (), 0.0),
        ("neg_rmse", (), 0.0),
    ],
)
def test_metric_perfect_prediction_hits_max(name, args, expected_max):
    y = np.array([1.0, 2.0, 3.0, 4.0])
    metric, invalid_reward, max_reward = make_regression_metric(name, y, *args)
    assert metric(y, y) == pytest.approx(expected_max, abs=1e-9)
    assert max_reward == pytest.approx(expected_max, abs=1e-9)


def test_metric_wrong_arity_is_rejected():
    """DSO validates the number of metric parameters; make sure that guard
    survived the port."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(AssertionError):
        make_regression_metric("neg_mse", y, 1.0)  # takes none
    with pytest.raises(AssertionError):
        make_regression_metric("inv_nrmse", y)  # takes one


def test_metric_degrades_with_error():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    metric, _, _ = make_regression_metric("inv_nrmse", y, 1.0)
    good = metric(y, y + 0.01)
    bad = metric(y, y + 1.0)
    assert good > bad


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------


def test_length_constraint_bounds_sampled_traversals():
    task, _, _ = _make_task()
    cfg = load_default_config()
    cfg["prior"] = {
        "length": {"min_": 4, "max_": 10, "on": True},
        "repeat": {"on": False},
        "inverse": {"on": False},
        "trig": {"on": False},
        "const": {"on": False},
        "no_inputs": {"on": False},
        "uniform_arity": {"on": False},
        "soft_length": {"on": False},
        "domain_range": {"on": False},
    }
    prior = make_prior(Program.library, cfg["prior"])
    sm = make_state_manager(cfg["state_manager"])
    pol = RNNPolicy(prior, sm, max_length=12)

    actions, _, _ = pol.sample(16)
    for a in actions:
        p = from_tokens(a)
        assert 4 <= len(p.traversal) <= 10


# ---------------------------------------------------------------------------
# State manager
# ---------------------------------------------------------------------------


def test_state_manager_input_dim_matches_output_width():
    task, _, _ = _make_task()

    class _FakePolicy:
        max_length = 16
        device = torch.device("cpu")

    sm = make_state_manager(
        {
            "observe_parent": True,
            "observe_sibling": True,
            "observe_action": False,
            "observe_dangling": False,
            "embedding": False,
        }
    )
    sm.setup_manager(_FakePolicy())
    lib = Program.library
    obs = np.array(
        [[lib.EMPTY_ACTION, lib.EMPTY_PARENT, lib.EMPTY_SIBLING, 1]] * 4, dtype=np.float32
    )
    out = sm.get_tensor_input(obs)
    assert out.shape == (4, sm.input_dim)
    assert out.dtype == torch.float32


def test_state_manager_embeddings_are_trainable():
    task, _, _ = _make_task()

    class _FakePolicy:
        max_length = 16
        device = torch.device("cpu")

    sm = make_state_manager({"embedding": True, "embedding_size": 8})
    sm.setup_manager(_FakePolicy())
    params = list(sm.parameters())
    assert len(params) == 2  # parent + sibling
    assert all(p.requires_grad for p in params)


# ---------------------------------------------------------------------------
# Policy: sampling, scoring, gradients
# ---------------------------------------------------------------------------


def test_policy_sample_shapes_and_termination():
    task, _, _ = _make_task()
    pol = _make_policy(task, max_length=16)
    n = 8
    actions, obs, priors = pol.sample(n)

    assert actions.shape[0] == n and actions.shape[1] <= 16
    assert obs.shape == (n, task.OBS_DIM, actions.shape[1])
    assert priors.shape == (n, actions.shape[1], pol.n_choices)
    assert actions.min() >= 0 and actions.max() < pol.n_choices


def test_sampled_programs_have_finite_positive_rewards():
    task, _, _ = _make_task()
    pol = _make_policy(task, max_length=16)
    _, _, r = _batch_from_policy(pol, n=16)
    assert np.isfinite(r).all()
    assert (r > 0).any()


def test_neglogp_and_entropy_wellformed():
    task, _, _ = _make_task()
    pol = _make_policy(task, max_length=16)
    B, _, _ = _batch_from_policy(pol, n=8)

    neglogp, entropy = pol.make_neglogp_and_entropy(B)
    assert neglogp.shape == (8,)
    assert entropy.shape == (8,)
    assert torch.isfinite(neglogp).all()
    assert torch.isfinite(entropy).all()
    assert (neglogp >= 0).all()  # -log p >= 0
    assert (entropy >= -1e-6).all()  # entropy >= 0


def test_safe_cross_entropy_guards_zero_times_neg_inf():
    p = torch.tensor([[1.0, 0.0]])
    logq = torch.tensor([[0.0, -float("inf")]])
    out = safe_cross_entropy(p, logq, dim=1)
    assert torch.isfinite(out).all()


def test_train_step_updates_parameters():
    task, _, _ = _make_task()
    pol = _make_policy(task, max_length=16)
    B, _, r = _batch_from_policy(pol, n=16)

    before = [p.detach().clone() for p in pol.parameters()]
    baseline = float(np.quantile(r, 0.95, method="higher"))
    loss = pol.train_step(baseline, B)

    assert np.isfinite(loss)
    changed = sum(1 for a, b in zip(before, pol.parameters()) if not torch.equal(a, b))
    assert changed > 0


def test_train_step_stays_finite_over_several_steps():
    task, _, _ = _make_task()
    pol = _make_policy(task, max_length=16)
    for _ in range(4):
        B, _, r = _batch_from_policy(pol, n=12)
        loss = pol.train_step(float(np.quantile(r, 0.95, method="higher")), B)
        assert np.isfinite(loss)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_train_step_keeps_entropy_decay_on_cuda():
    """Registered buffers created after ``Module.to`` must still use the policy device."""
    task, _, _ = _make_task()
    pol = _make_policy(task, max_length=16, device=torch.device("cuda:0"))
    B, _, r = _batch_from_policy(pol, n=8)

    assert pol.entropy_gamma_decay.device.type == "cuda"
    loss = pol.train_step(float(np.quantile(r, 0.95, method="higher")), B)
    assert np.isfinite(loss)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

_E2E_PRIOR = {
    "length": {"min_": 2, "max_": 16, "on": True},
    "repeat": {"on": False},
    "inverse": {"on": True},
    "trig": {"on": False},
    "const": {"on": False},
    "no_inputs": {"on": True},
    "uniform_arity": {"on": True},
    "soft_length": {"on": False},
    "domain_range": {"on": False},
}


def _e2e_config(X, y, **training):
    train_cfg = {"n_samples": 1000, "batch_size": 100, "verbose": False, "seed": 0, "hof": 5}
    train_cfg.update(training)
    return {
        "task": {
            "dataset": (X, y),
            "function_set": ["add", "sub", "mul", "div"],
            "threshold": 1e-10,
        },
        "training": train_cfg,
        "policy": {"max_length": 16},
        "policy_optimizer": {"learning_rate": 0.005},
        "prior": _E2E_PRIOR,
    }


def test_optimizer_recovers_linear_expression():
    """Full stack: DSO should recover y = 2x + 1 exactly on a tiny budget."""
    X = np.linspace(-1.0, 1.0, 40).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 1.0

    model = DeepSymbolicOptimizer(_e2e_config(X, y))
    result = model.train()

    assert result["r"] == pytest.approx(1.0, abs=1e-6)
    assert bool(result["program"].evaluate.get("success")) is True
    np.testing.assert_allclose(result["program"].execute(X), y, atol=1e-9)
    assert len(result["history"]) == result["iterations"]
    assert len(result["hall_of_fame"]) > 0


def test_pqt_optimizer_runs():
    X = np.linspace(-1.0, 1.0, 40).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 1.0

    cfg = _e2e_config(X, y, n_samples=600)
    cfg["policy_optimizer"] = {
        "policy_optimizer_type": "pqt",
        "learning_rate": 0.005,
        "pqt_k": 10,
        "pqt_batch_size": 10,
        "pqt_use_pg": True,
    }
    model = DeepSymbolicOptimizer(cfg)
    result = model.train()

    assert model.policy.pqt is True
    assert np.isfinite(result["r"])
    assert result["iterations"] > 0


def test_unported_task_types_raise():
    for task_type in ("control", "binding"):
        with pytest.raises(NotImplementedError):
            DeepSymbolicOptimizer({"task": {"task_type": task_type}}).setup()


def test_unported_ppo_optimizer_raises():
    X = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
    y = 2.0 * X[:, 0]
    cfg = _e2e_config(X, y)
    cfg["policy_optimizer"] = {"policy_optimizer_type": "ppo"}
    with pytest.raises(NotImplementedError):
        DeepSymbolicOptimizer(cfg).setup()


# ---------------------------------------------------------------------------
# KD_DSO wrapper
# ---------------------------------------------------------------------------


def test_kd_dso_fit_and_predict():
    from kd.model.kd_dso import KD_DSO

    X = np.linspace(-1.0, 1.0, 40).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 1.0

    model = KD_DSO(
        n_samples=1000,
        batch_size=100,
        function_set=["add", "sub", "mul", "div"],
        max_length=16,
        learning_rate=0.005,
        threshold=1e-10,
        seed=0,
        device="cpu",
        verbose=False,
    )
    result = model.fit(X, y)

    assert result is model
    assert isinstance(model.best_expression_, str)
    assert isinstance(model.best_reward_, float)
    assert model.best_reward_ == pytest.approx(1.0, abs=1e-6)
    assert model.success_ is True

    pred = model.predict(X)
    assert pred.shape == (40,)
    np.testing.assert_allclose(pred, y, atol=1e-9)
    assert model.score(X, y) == pytest.approx(1.0, abs=1e-9)

    front = model.get_pareto_front()
    assert isinstance(front, list) and len(front) > 0
    assert {"expression", "reward", "complexity"} <= set(front[0])


def test_kd_dso_predict_before_fit_raises():
    from kd.model.kd_dso import KD_DSO

    with pytest.raises(RuntimeError):
        KD_DSO().predict(np.zeros((3, 1)))


def test_kd_dso_fit_dataset_sets_test_mse():
    from kd.dataset import SymbolicRegressionDataset
    from kd.model.kd_dso import KD_DSO

    dataset = SymbolicRegressionDataset(name="Keijzer-2")
    model = KD_DSO(
        n_samples=600,
        batch_size=200,
        function_set=["add", "sub", "mul", "div"],
        max_length=16,
        learning_rate=0.005,
        seed=0,
        device="cpu",
        verbose=False,
    )
    model.fit_dataset(dataset)

    assert isinstance(model.best_expression_, str)
    assert hasattr(model, "test_mse_")
    assert np.isfinite(model.test_mse_) or np.isnan(model.test_mse_)


def test_kd_dso_is_registered_in_model_package():
    import kd.model as m

    assert "KD_DSO" in m.__all__
    assert hasattr(m, "KD_DSO")
