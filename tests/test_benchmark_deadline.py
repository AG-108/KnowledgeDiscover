"""Bounded C3 checks on toy data and local fake services, never benchmark matrices."""

import copy
import multiprocessing
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import numpy as np
import pytest

from kd.benchmark_deadline import (
    FitDeadline,
    SoftDeadline,
    _gplearn_fit,
    anytime_call,
    current_deadline,
    deadline_scope,
)
from kd.benchmark_execution import PhaseClock, execution_policy, fit_call


def test_fit_deadline_excludes_setup_and_score_and_keeps_cumulative_budget():
    seen = []
    def observe():
        seen.append(current_deadline().remaining())
        time.sleep(0.02)
    with PhaseClock(policy=execution_policy(0.15)) as clock:
        time.sleep(0.03)
        fit_call(observe)
        time.sleep(0.03)
        fit_call(observe)
    assert 0.10 < seen[0] <= 0.15
    assert seen[1] < seen[0] - 0.015
    assert clock.times()["fit_seconds"] < 0.08
    assert current_deadline() is None


def test_none_budget_preserves_original_call_and_exception_cleanup():
    with PhaseClock(policy=execution_policy()) as clock:
        assert fit_call(lambda: current_deadline()) is None
        with pytest.raises(ValueError):
            fit_call(lambda: (_ for _ in ()).throw(ValueError("ordinary failure")))
        assert clock.phase == "score"
    assert current_deadline() is None
    with deadline_scope(0):
        with pytest.raises(SoftDeadline):
            current_deadline().check()


@pytest.mark.parametrize("seed_kind", ["integer", "random_state", "global"])
def test_gplearn_generational_execution_preserves_evolution(seed_kind):
    from gplearn.genetic import SymbolicRegressor

    seed = 11 if seed_kind == "integer" else np.random.RandomState(11)
    if seed_kind == "global":
        seed = None
        np.random.seed(11)
    params = dict(population_size=32, generations=3, init_depth=(1, 2),
                  stopping_criteria=-1, n_jobs=1, random_state=seed)
    baseline = SymbolicRegressor(**copy.deepcopy(params))
    X = np.linspace(-1, 1, 25).reshape(-1, 1)
    y = np.sin(3 * X[:, 0])
    baseline.fit(X, y)
    if seed_kind == "global":
        np.random.seed(11)
    timed = SymbolicRegressor(**copy.deepcopy(params))
    with deadline_scope(20):
        anytime_call(timed.fit, X, y)
    assert str(timed._program) == str(baseline._program)
    assert timed.run_details_["best_fitness"] == baseline.run_details_["best_fitness"]
    np.testing.assert_allclose(timed.predict(X), baseline.predict(X))
    assert timed.generations == 3 and timed.warm_start is False
    assert timed.stop_reason_ == "budget_exhausted"


def test_gplearn_returns_best_completed_generation_and_restores_configuration():
    from gplearn.genetic import SymbolicRegressor
    model = SymbolicRegressor(population_size=32, generations=10, random_state=2,
                              stopping_criteria=-1, init_depth=(1, 2))
    X = np.linspace(-1, 1, 25).reshape(-1, 1)
    y = np.sin(3 * X[:, 0])
    completed = []
    native = model.fit
    deadline = FitDeadline(20)
    def fit(*args, **kwargs):
        result = native(*args, **kwargs)
        completed.append(model._program)
        if len(completed) == 2:
            deadline.expires = time.perf_counter() - 1
        return result
    _gplearn_fit(fit, model, deadline, (X, y), {})
    assert model._program.raw_fitness_ == min(p.raw_fitness_ for p in completed)
    assert model.stop_reason_ == "time_limit" and len(completed) == 2
    assert (model.generations, model.warm_start, model.random_state) == (10, False, 2)


@pytest.mark.parametrize("module,parameter", [
    ("pysr.sr", "timeout_in_seconds"), ("pyoperon.sklearn", "max_time"),
])
def test_native_budget_dispatch_restores_params_and_reports_deadline(module, parameter, monkeypatch):
    observed = []
    class Native:
        def fit(self):
            observed.append(getattr(self, parameter))
            current_deadline().expires = time.perf_counter() - 1
            return self
    Native.__module__ = module
    if module == "pysr.sr":
        import sys
        monkeypatch.setitem(sys.modules, "pysr", SimpleNamespace(
            jl=SimpleNamespace(seval=lambda expression: 1_000_000_000)))
    model = Native()
    model.early_stop_condition = None
    setattr(model, parameter, None)
    with deadline_scope(0.25):
        assert anytime_call(model.fit) is model
    assert 0 < observed[0] <= (1 if parameter == "max_time" else 0.25)
    assert getattr(model, parameter) is None
    assert model.early_stop_condition is None
    assert model.stop_reason_ == "time_limit"
    assert model.stop_reason_source_ == "fit_deadline"


def test_physo_logger_retains_best_and_does_not_hide_native_errors():
    class Logger:
        def log(self, **kwargs):
            self.saved = True
        def get_pareto_front(self):
            return [], ["worse", "best"], [0.2, 0.8], []
    logger = Logger()
    def SR(**options):
        wrapped = options["get_run_logger"]()
        current_deadline().expires = time.perf_counter() - 1
        wrapped.log()
        raise AssertionError("Unreachable")
    SR.__module__ = "physo.task.sr"
    with deadline_scope(5):
        expression, wrapped = anytime_call(SR, get_run_logger=lambda: logger)
    assert expression == "best" and logger.saved
    assert wrapped.stop_reason_ == "time_limit"
    def SR(**options):
        raise ValueError("native error")
    SR.__module__ = "physo.task.sr"
    with deadline_scope(5), pytest.raises(ValueError, match="native error"):
        anytime_call(SR, get_run_logger=lambda: logger)


def test_physo_stops_before_an_epoch_that_would_overrun_the_deadline():
    class Logger:
        def log(self, **kwargs):
            pass
        def get_pareto_front(self):
            return [], ["best"], [0.9], []
    def SR(**options):
        wrapped = options["get_run_logger"]()
        # A 3 s epoch with only ~2 s left: the next epoch cannot finish in time.
        wrapped._last_log = time.perf_counter() - 3
        current_deadline().expires = time.perf_counter() + 2
        wrapped.log()
        raise AssertionError("Unreachable")
    SR.__module__ = "physo.task.sr"
    with deadline_scope(60):
        expression, wrapped = anytime_call(SR, get_run_logger=Logger)
    assert expression == "best"
    assert wrapped.stop_reason_ == "time_limit"
    assert wrapped.stop_reason_source_ == "epoch_deadline_predicted"

    def SR(**options):
        wrapped = options["get_run_logger"]()
        for _ in range(3):  # fast epochs with plenty of budget keep running
            wrapped.log()
        return "finished", wrapped
    SR.__module__ = "physo.task.sr"
    with deadline_scope(60):
        expression, wrapped = anytime_call(SR, get_run_logger=Logger)
    assert expression == "finished" and wrapped.stop_reason_ is None


def test_llmsr_cancels_slow_http_and_keeps_best_without_live_children(monkeypatch):
    from kd.model.kd_llmsr import KD_LLMSR

    release, received = threading.Event(), threading.Event()
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.set()
            release.wait(15)
            try:
                self.send_response(200)
                self.end_headers()
            except OSError:
                pass
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    model = KD_LLMSR(endpoint=f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                     max_llm_calls=4, request_timeout=30)
    # Supply a completed seed, then exercise the real cancellable HTTP process.
    monkeypatch.setattr(model, "_evaluate_parallel",
                        lambda jobs, X, y: [("ok", np.ones(2), 1.0) for _ in jobs])
    before = {p.pid for p in multiprocessing.active_children()}
    X = np.linspace(-1, 1, 12).reshape(-1, 1)
    started = time.perf_counter()
    try:
        with deadline_scope(4):
            model.fit(X, X[:, 0] ** 2)
        assert received.is_set()
        assert model.stop_reason_ == "time_limit"
        assert np.isfinite(model.predict(X)).all()
        assert time.perf_counter() - started < 6
        assert {p.pid for p in multiprocessing.active_children()} == before
    finally:
        release.set()
        server.shutdown()
        server.server_close()


def _slow_evaluation_worker(*args):
    """Force an unfinished evaluator independently of process startup speed."""
    time.sleep(60)


def test_llmsr_evaluator_deadline_cleans_up_and_no_candidate_fails(monkeypatch):
    from kd.model.kd_llmsr import KD_LLMSR
    monkeypatch.setattr("kd.model.kd_llmsr._evaluation_worker", _slow_evaluation_worker)
    model = KD_LLMSR(endpoint="http://127.0.0.1:1", num_evaluators=2)
    model.variable_names_ = ["x1"]
    model._fit_deadline = FitDeadline(0.1)
    before = {p.pid for p in multiprocessing.active_children()}
    X = np.arange(10.0).reshape(-1, 1)
    result = model._evaluate_parallel([("p0*x1", 1)] * 3, X, X[:, 0])
    assert all(status == "timeout" for status, _, _ in result)
    assert model._fit_deadline.stopped
    assert {p.pid for p in multiprocessing.active_children()} == before
    with deadline_scope(0), pytest.raises(RuntimeError, match="initial skeleton failed"):
        model.fit(X, X[:, 0])


def test_dso_preserves_candidate_when_deadline_occurs_during_reward(monkeypatch):
    from test_kd_dso import _make_policy, _make_task

    from kd.model.dso.program import from_str_tokens
    from kd.model.dso.train import Trainer

    task, _, _ = _make_task()
    policy = _make_policy(task, max_length=8)
    trainer = Trainer(policy, batch_size=1, n_samples=10, verbose=False)
    actions, obs, priors = policy.sample(1)
    program = from_str_tokens("x1")
    native = task.reward_function
    def reward(program):
        value = native(program)
        current_deadline().expires = time.perf_counter() - 1
        return value
    monkeypatch.setattr(task, "reward_function", reward)
    with deadline_scope(5), pytest.raises(SoftDeadline):
        trainer.run_one_step((actions, obs, priors, [program]))
    assert trainer.p_r_best is program and np.isfinite(trainer.r_best)


def test_dscv_preserves_partial_batch_candidate_and_adapter_reason(monkeypatch):
    import torch

    from kd.model.discover.searcher import Searcher
    from kd.model.kd_dscv import KD_DSCV

    class Program:
        str_expression = "u_t = u"
        STRidge = SimpleNamespace(terms=["u"])
        @property
        def r_ridge(self):
            current_deadline().expires = time.perf_counter() - 1
            return 0.8
    program = Program()
    searcher = Searcher.__new__(Searcher)
    searcher.total_epoch = searcher.nevals = 0
    searcher.batch_size = 1
    searcher.n_iterations = 5
    searcher.best_p = None
    searcher.r_best = -np.inf
    searcher.prev_r_best = None
    searcher.pq = SimpleNamespace(iter_in_order=lambda: [])
    # search() only uses len(pq) when keep_history is false.
    searcher.stability_selection = 0
    searcher.controller = SimpleNamespace(
        sample_episodes=lambda batch: (torch.tensor([[0]]), None, None, None, None))
    monkeypatch.setattr("kd.model.discover.searcher.from_tokens", lambda actions: program)
    with deadline_scope(5):
        result = searcher.search(n_epochs=5, verbose=False, keep_history=True)
    assert result["program"] is program and result["stop_reason"] == "time_limit"
    model = KD_DSCV.__new__(KD_DSCV)
    model.searcher = SimpleNamespace(search=lambda **kwargs: result)
    assert model.train(5, False)["expression"] == "u_t = u"
    assert model.stop_reason_ == "time_limit"


def test_real_target_worker_returns_time_limit_and_preserves_first_result(tmp_path):
    from benchmark_helpers import make_case

    import run_benchmark as benchmark

    case = make_case()
    case["model_params"].update(population_size=32, generations=100000,
                                init_depth=[1, 2], n_jobs=1, stopping_criteria=-1)
    rows = benchmark.execute_case(case, tmp_path, timeout=0.12, setup_timeout=60)
    assert len(rows) == 1 and rows[0]["status"] == "ok", rows
    assert rows[0]["stop_reason"] == "time_limit"
    assert rows[0]["stop_reason_source"] == "generation_deadline"
    assert np.isfinite(rows[0]["test_mse"])
    before = (tmp_path / "cases" / case["name"] / "result.json").read_bytes()
    assert benchmark.execute_case(case, tmp_path, timeout=0.12,
                                  setup_timeout=60, resume=True)[0]["stop_reason"] == "time_limit"
    assert (tmp_path / "cases" / case["name"] / "result.json").read_bytes() == before


@pytest.mark.parametrize("method", ["pysr", "pyoperon", "physo"])
def test_native_soft_stop_on_toy_data_under_process_watchdog(tmp_path, method):
    import json
    import os
    import sys

    from kd.benchmark_execution import supervise

    budget = {"pysr": 60, "pyoperon": 0.01, "physo": 0.3}[method]
    from benchmark_helpers import host_slots

    resources = host_slots()["cpu"] if method == "pysr" else None
    policy = execution_policy(budget, setup_timeout=180, score_timeout=30,
                              resource_policy=resources)
    report, phase = tmp_path / "native.json", tmp_path / "phase.json"
    common = (
        "import json, os, numpy as np\n"
        "from kd.benchmark_execution import PhaseClock, fit_call\n"
        "from kd.benchmark_deadline import current_deadline\n"
        f"policy={policy!r}\n"
        f"clock=PhaseClock({str(phase)!r}, started=float(os.environ['KD_BENCHMARK_STARTED']), policy=policy)\n"
        "X=np.linspace(-1,1,24).reshape(-1,1); y=np.cos(3*X[:,0])+0.2*X[:,0]\n"
    )
    if method == "pysr":
        if not os.environ.get("PYTHON_JULIAPKG_EXE"):
            pytest.skip("Requires configured PySR/Julia environment")
        body = (
            "from pysr import PySRRegressor, jl\n"
            "from kd.benchmark_deadline import FitDeadline, _pysr_deadline_condition\n"
            "model=PySRRegressor(niterations=100000,populations=4,population_size=20,maxsize=8,"
            "parallelism='multiprocessing',procs=4,deterministic=False,random_state=0,"
            "binary_operators=['+','*'],unary_operators=[],verbosity=0,progress=False,"
            f"output_directory={str(tmp_path / 'equations')!r})\n"
            # Check the public Julia predicate for None, a numeric threshold,
            # and a user function before testing cold four-worker initialization.
            "for prior, low in [(None,False),(0.5,True),('(l,c)->l<0.5 && c<10',True)]:\n"
            "    model.early_stop_condition=prior\n"
            "    predicate=jl.seval(_pysr_deadline_condition(model,FitDeadline(30),jl))\n"
            "    assert bool(predicate(0.1,3))==low\n"
            "    assert not bool(predicate(1.0,3))\n"
            "    expired=jl.seval(_pysr_deadline_condition(model,FitDeadline(0),jl))\n"
            "    assert bool(expired(1.0,3))\n"
            "model.early_stop_condition='(l,c)->l<0.0'\n"
            "with clock: fit_call(model.fit,X,y)\n"
            "assert model.timeout_in_seconds is None\n"
            "assert model.early_stop_condition=='(l,c)->l<0.0'\n"
            "assert np.all(np.isfinite(model.predict(X)))\n"
            "reason=model.stop_reason_\n"
        )
    elif method == "pyoperon":
        body = (
            "from pyoperon.sklearn import SymbolicRegressor\n"
            "model=SymbolicRegressor(population_size=32,generations=100000,"
            "max_evaluations=100000000,optimizer_iterations=2,n_threads=1,random_state=1,"
            "allowed_symbols='add,mul,constant,variable',max_length=12,max_depth=5)\n"
            "with clock: fit_call(model.fit,X,y)\n"
            "assert model.max_time is None\n"
            "assert np.all(np.isfinite(model.predict(X)))\n"
            "reason=model.stop_reason_\n"
        )
    else:
        body = (
            "import copy, torch, physo\n"
            "torch.set_num_threads(1)\n"
            "from physo.learn.monitoring import RunLogger, RunVisualiser\n"
            "config=copy.deepcopy(physo.config.config0.config0)\n"
            "config['learning_config'].update(batch_size=32)\n"
            "with clock:\n"
            "    expr, logger=fit_call(physo.SR,X.T,y,epochs=100000,stop_reward=2,"
            "X_names=['x1'],X_units=[[0,0,0]],y_name='y',y_units=[0,0,0],"
            "fixed_consts=[1.0],fixed_consts_units=[[0,0,0]],"
            "op_names=['add','mul'],run_config=config,parallel_mode=False,device='cpu',"
            "get_run_logger=lambda:RunLogger(do_save=False),"
            "get_run_visualiser=lambda:RunVisualiser(do_show=False,do_prints=False,do_save=False))\n"
            "assert expr is not None\n"
            "prediction=expr(torch.tensor(X.T,dtype=torch.float32)).detach().numpy()\n"
            "assert np.all(np.isfinite(prediction))\n"
            "reason=logger.stop_reason_\n"
        )
    script = common + body + f"json.dump(dict(reason=reason,times=clock.times()),open({str(report)!r},'w'))\n"
    outcome = supervise([sys.executable, "-c", script], cwd=tmp_path,
                        log_path=tmp_path / "run.log", phase_path=phase, policy=policy, model=method)
    assert outcome["returncode"] == 0 and outcome["timeout_phase"] is None, (
        tmp_path / "run.log").read_text(encoding="utf-8", errors="replace")
    result = json.loads(report.read_text())
    assert result["reason"] == "time_limit"
    assert budget <= result["times"]["fit_seconds"] < budget + 5


def test_hard_only_control_preserves_no_soft_deadline():
    with PhaseClock(policy=execution_policy(0.01, soft_deadline=False)):
        assert fit_call(lambda: current_deadline()) is None
    for invalid in ("false", 0, None):
        with pytest.raises(ValueError, match="boolean"):
            execution_policy(1, soft_deadline=invalid)
    import run_benchmark as benchmark
    config, _ = benchmark.load_benchmark_config("configs/benchmark/llmsr_llama31_8b_1k.json")
    assert config["soft_deadline"] is False
    assert config["timeout"] == 43200


def test_llmsr_final_evaluation_crossing_deadline_has_time_limit(monkeypatch):
    from kd.model.kd_llmsr import KD_LLMSR

    model = KD_LLMSR(endpoint="http://127.0.0.1:1", max_llm_calls=1)
    evaluated = []
    def evaluate(jobs, X, y):
        evaluated.append(jobs)
        if len(evaluated) == 2:
            current_deadline().expires = time.perf_counter() - 1
        return [("ok", np.ones(2), 1.0 if len(evaluated) == 1 else 0.1) for _ in jobs]
    monkeypatch.setattr(model, "_evaluate_parallel", evaluate)
    monkeypatch.setattr(model, "_request_batch", lambda prompt, indices: [
        (index, ("def equation(x1, params):\n    return params[0]*x1**2+params[1]", {}, 0))
        for index in indices])
    X = np.linspace(-1, 1, 12).reshape(-1, 1)
    with deadline_scope(5):
        model.fit(X, X[:, 0] ** 2)
    assert model.search_stats_["llm_calls"] == 1
    assert model.stop_reason_ == "time_limit"
    assert np.isfinite(model.predict(X)).all()


def test_dso_final_gradient_crossing_deadline_retains_result(monkeypatch):
    from kd.model.dso.core import DeepSymbolicOptimizer

    X = np.linspace(-1, 1, 20).reshape(-1, 1)
    model = DeepSymbolicOptimizer({
        "task": {"task_type": "regression", "dataset": (X, X[:, 0] ** 2),
                 "function_set": ["add", "mul"], "metric": "inv_nrmse",
                 "metric_params": [1.0]},
        "training": {"n_samples": 8, "batch_size": 8, "verbose": False,
                     "early_stopping": False},
        "policy": {"max_length": 8},
    })
    model.setup()
    native = model.policy.train_step
    def train_step(*args, **kwargs):
        value = native(*args, **kwargs)
        current_deadline().expires = time.perf_counter() - 1
        return value
    monkeypatch.setattr(model.policy, "train_step", train_step)
    with deadline_scope(5):
        result = model.train()
    assert result["program"] is not None and result["iterations"] == 1
    assert result["stop_reason"] == "time_limit"


def test_dscv_final_step_crossing_deadline_retains_result(monkeypatch):
    from kd.model.discover.searcher import Searcher

    program = SimpleNamespace(r_ridge=0.8, str_expression="u_t = u",
                              STRidge=SimpleNamespace(terms=["u"]))
    searcher = Searcher.__new__(Searcher)
    searcher.n_iterations = 1
    searcher.pq = SimpleNamespace(iter_in_order=lambda: [])
    searcher.best_p = None
    searcher.stability_selection = 0
    def step(**kwargs):
        searcher.best_p = program
        current_deadline().expires = time.perf_counter() - 1
    monkeypatch.setattr(searcher, "search_one_step", step)
    with deadline_scope(5):
        result = searcher.search(n_epochs=1, verbose=False, keep_history=True)
    assert result["program"] is program
    assert result["stop_reason"] == "time_limit"
