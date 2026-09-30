"""Exercise real OS processes: deadlines, descendants, locks and native target fits."""

import json
import os
import sys
import time
from pathlib import Path

import pytest

import run_benchmark as benchmark
from kd.benchmark_execution import atomic_json, case_lock, execution_policy, supervise


@pytest.mark.skipif(os.name != "nt", reason="Windows read/replace sharing race")
def test_atomic_phase_write_retries_transient_windows_reader_lock(monkeypatch, tmp_path):
    path = tmp_path / "phase.json"
    atomic_json(path, {"phase": "setup"})
    original = Path.replace
    attempts = []

    def replace(source, destination):
        attempts.append(source)
        if len(attempts) <= 2:
            assert json.loads(path.read_text())["phase"] == "setup"
            raise PermissionError("reader has not closed yet")
        return original(source, destination)

    monkeypatch.setattr(Path, "replace", replace)
    atomic_json(path, {"phase": "fit"})
    assert len(attempts) == 3
    assert json.loads(path.read_text())["phase"] == "fit"


def run_script(tmp_path, body, **limits):
    phase_path = tmp_path / "phase.json"
    script = (
        "import os, time\n"
        "from kd.benchmark_execution import PhaseClock, fit_call\n"
        f"clock = PhaseClock({str(phase_path)!r}, started=float(os.environ['KD_BENCHMARK_STARTED']))\n"
        "with clock:\n" + "\n".join("    " + line for line in body.splitlines())
    )
    policy = execution_policy(timeout=limits.pop("timeout", 0.15), timeout_grace=0,
                              setup_timeout=10, score_timeout=limits.pop("score_timeout", 2), **limits)
    return supervise([sys.executable, "-c", script], cwd=benchmark.ROOT,
                     log_path=tmp_path / "run.log", phase_path=phase_path, policy=policy)


def test_scoring_is_excluded_from_fit_deadline(tmp_path):
    outcome = run_script(tmp_path, "fit_call(time.sleep, 0.02)\ntime.sleep(0.3)")
    assert outcome["returncode"] == 0 and outcome["timeout_phase"] is None
    assert 0 < outcome["fit_seconds"] < 0.15
    assert outcome["score_seconds"] >= 0.3


@pytest.mark.parametrize("phase", ["fit", "score"])
def test_fit_and_scoring_have_separate_hard_limits(tmp_path, phase):
    body = "fit_call(time.sleep, 60)" if phase == "fit" else "fit_call(time.sleep, 0.01)\ntime.sleep(60)"
    outcome = run_script(tmp_path, body, score_timeout=0.1)
    assert outcome["timeout_phase"] == phase
    assert outcome["worker_elapsed_seconds"] < 10


def test_setup_timeout_before_worker_phase_file(tmp_path):
    outcome = supervise([sys.executable, "-c", "import time; time.sleep(60)"],
                        cwd=benchmark.ROOT, log_path=tmp_path / "run.log",
                        phase_path=tmp_path / "missing.json",
                        policy=execution_policy(1, setup_timeout=0.1))
    assert outcome["timeout_phase"] == "setup"


def test_hard_timeout_kills_descendant_process(tmp_path):
    psutil = pytest.importorskip("psutil")
    pid_path = tmp_path / "child.pid"
    body = (
        "import subprocess, sys\n"
        "clock.change('fit')\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], "
        + ("creationflags=subprocess.CREATE_NO_WINDOW" if os.name == "nt" else "start_new_session=False")
        + ")\n"
        f"open({str(pid_path)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(60)"
    )
    outcome = run_script(tmp_path, body, timeout=0.5)
    assert outcome["timeout_phase"] == "fit"
    pid = int(pid_path.read_text())
    deadline = time.monotonic() + 3
    while psutil.pid_exists(pid) and time.monotonic() < deadline:
        if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
            break
        time.sleep(0.05)
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


def test_case_lock_rejects_concurrent_orchestrators(tmp_path):
    with case_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already running"):
            with case_lock(tmp_path):
                pytest.fail("acquired the same case twice")
    with case_lock(tmp_path):
        pass


def test_real_ode_targets_run_and_resume_without_rewriting(tmp_path):
    catalog, _ = benchmark.dataset_catalog()
    case = benchmark.build_experiments(catalog, models=["sindy"],
                                       datasets=["ode_core_oscillator"])[0]
    rows = benchmark.execute_case(case, tmp_path, timeout=10)
    assert [row["status"] for row in rows] == ["ok", "ok"]
    assert all(row["stop_reason"] == "converged" and row["fit_seconds"] > 0 for row in rows)
    directory = tmp_path / "cases" / case["name"]
    workers = [json.loads(path.read_text()) for path in directory.glob("targets/**/worker.json")]
    assert len(workers) == 2 and workers[0]["pid"] != workers[1]["pid"]
    original = {path: path.read_bytes() for path in directory.glob("targets/**/result.json")}
    assert benchmark.execute_case(case, tmp_path, timeout=10, resume=True) == rows
    assert all(path.read_bytes() == value for path, value in original.items())


def test_real_first_target_timeout_still_runs_second_target(monkeypatch, tmp_path):
    catalog, _ = benchmark.dataset_catalog()
    case = benchmark.build_experiments(catalog, models=["sindy"],
                                       datasets=["ode_core_oscillator"])[0]
    wrapper = tmp_path / "slow_first_target.py"
    wrapper.write_text(
        "import json, sys, time\n"
        f"sys.path.insert(0, {str(benchmark.ROOT)!r})\n"
        "import run_benchmark as benchmark\n"
        "spec = json.load(open(sys.argv[sys.argv.index('--_target-file') + 1]))\n"
        "original = benchmark._new_model\n"
        "def create(*args, **kwargs):\n"
        "    model = original(*args, **kwargs)\n"
        "    fit = model.fit\n"
        "    def delayed(*a, **kw):\n"
        "        if spec['target_index'] == 0:\n"
        "            time.sleep(60)\n"
        "        return fit(*a, **kw)\n"
        "    model.fit = delayed\n"
        "    return model\n"
        "benchmark._new_model = create\n"
        "raise SystemExit(benchmark.main(sys.argv[1:]))\n", encoding="utf-8",
    )
    original = benchmark._launch_worker

    def launch(case_path, attempt, policy, spec=None):
        if spec is None:
            return original(case_path, attempt, policy)
        benchmark.write_json(attempt / "target.json", spec)
        return supervise(
            [sys.executable, str(wrapper), "--_case-file", str(case_path),
             "--_attempt-dir", str(attempt), "--_target-file", str(attempt / "target.json")],
            cwd=benchmark.ROOT, log_path=attempt / "run.log",
            phase_path=attempt / "phase.json", policy=policy,
        )

    monkeypatch.setattr(benchmark, "_launch_worker", launch)
    rows = benchmark.execute_case(case, tmp_path / "results", timeout=0.2, timeout_grace=0)
    assert [row["status"] for row in rows] == ["timeout", "ok"]
    assert rows[0]["timeout_phase"] == "fit"
    assert rows[1]["stop_reason"] == "converged"


def test_parallel_reuses_terminal_failures_without_overwriting_destination(monkeypatch, tmp_path):
    from test_run_benchmark import make_case, mock_target_workers

    from scripts import run_parallel_benchmark as parallel

    case = make_case()
    mock_target_workers(monkeypatch, case, ["timeout"])
    rows = benchmark.execute_case(case, tmp_path / "source", timeout=1)
    policy = execution_policy(1)
    assert parallel._reuse_results([case], tmp_path / "copy", [tmp_path / "source"], policy) == 1
    monkeypatch.setattr(benchmark, "_launch_worker", lambda *a, **kw: pytest.fail("reran timeout"))
    assert parallel._run_one(case, tmp_path / "copy", 1) == rows
    assert parallel._reuse_results([case], tmp_path / "copy", [tmp_path / "source"], policy) == 0
    assert parallel._reuse_results([case], tmp_path / "different", [tmp_path / "source"],
                                   execution_policy(2)) == 0
    path = tmp_path / "copy" / "cases" / case["name"] / "result.json"
    original = path.read_bytes()
    with pytest.raises(ValueError, match="timing policy"):
        parallel._run_one(case, tmp_path / "copy", 2)
    assert path.read_bytes() == original


def test_parallel_rejects_automatic_retries(monkeypatch):
    from scripts import run_parallel_benchmark as parallel

    argv = ["runner", "--manifest", "manifest.json", "--compatibility", "compat.json",
            "--output-dir", "output"]
    monkeypatch.setattr(sys, "argv", argv)
    assert parallel._parse_args().retries == 0
    monkeypatch.setattr(sys, "argv", argv + ["--retries", "1"])
    with pytest.raises(SystemExit):
        parallel._parse_args()
