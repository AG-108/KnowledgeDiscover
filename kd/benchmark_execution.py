"""Target worker timing and process supervision (no model imports)."""

from __future__ import annotations

import contextlib
import json
import math
import os
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path

from kd.benchmark_resources import (
    ProcessTreeMonitor,
    policy_hash,
    validate_host,
    validate_policy,
    worker_command,
    worker_environment,
)


class SkipCase(Exception):
    """Unsupported capability or source prerequisite, shared by CLI and imports."""


EXECUTION_PROTOCOL_VERSION = "target-v2"
TERMINAL_STATUSES = frozenset({"ok", "error", "timeout", "skipped"})
STOP_REASONS = (
    "budget_exhausted", "time_limit", "converged", "early_stop", "hard_timeout", "error",
    "resource_limit",
)
_LOCAL = threading.local()


def execution_policy(timeout=None, *, timeout_grace=5.0, setup_timeout=300.0,
                     score_timeout=300.0, resource_policy=None, soft_deadline=True):
    values = dict(timeout=timeout, timeout_grace=timeout_grace,
                  setup_timeout=setup_timeout, score_timeout=score_timeout)
    for key, value in values.items():
        if key == "timeout" and value is None:
            continue
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0
                or (key != "timeout_grace" and value == 0)):
            raise ValueError(f"Invalid {key}: {value!r}")
    if not isinstance(soft_deadline, bool):
        raise ValueError("soft_deadline must be a boolean")
    result = {"version": EXECUTION_PROTOCOL_VERSION, "soft_deadline": soft_deadline, **values}
    if resource_policy is not None:
        result["resource_policy"] = validate_policy(resource_policy)
    return result


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                         encoding="utf-8")
    # Windows readers briefly deny replacement unless opened with share-delete.
    # Keep the previous complete snapshot visible while retrying that transient race.
    deadline = time.perf_counter() + 1.0
    while True:
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if os.name != "nt" or time.perf_counter() >= deadline:
                raise
            time.sleep(0.01)


@contextlib.contextmanager
def case_lock(directory):
    """An OS lock is released even when the orchestrator/container is killed."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f"Case is already running: {directory}") from exc
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError(f"Case is already running: {directory}") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class PhaseClock:
    """Publish cumulative phase times so even short phase transitions are observed."""

    def __init__(self, path=None, started=None, policy=None):
        self.policy = policy or {}
        self.path = path
        self.since = time.perf_counter()
        self.started = self.since if started is None else started
        self.phase = "setup"
        self.elapsed = dict(setup=self.since - self.started, fit=0.0, score=0.0)
        self.publish()

    def publish(self):
        if self.path:
            atomic_json(self.path, dict(started=self.started, since=self.since,
                                       phase=self.phase, elapsed=self.elapsed))

    def change(self, phase):
        now = time.perf_counter()
        self.elapsed[self.phase] += now - self.since
        self.phase, self.since = phase, now
        self.publish()

    def times(self):
        result = dict(self.elapsed)
        result[self.phase] += time.perf_counter() - self.since
        return {f"{key}_seconds": value for key, value in result.items()}

    def __enter__(self):
        self.previous = getattr(_LOCAL, "clock", None)
        _LOCAL.clock = self
        return self

    def __exit__(self, *args):
        _LOCAL.clock = self.previous


def fit_call(function, *args, **kwargs):
    """Measure the adapter's training/search call, excluding held-out scoring."""
    clock = getattr(_LOCAL, "clock", None)
    if clock:
        clock.change("fit")
    from kd.benchmark_deadline import anytime_call, deadline_scope

    limit = clock.policy.get("timeout") if clock and clock.policy.get("soft_deadline") else None
    remaining = max(0.0, limit - clock.times()["fit_seconds"]) if limit is not None else None
    try:
        with deadline_scope(remaining):
            return anytime_call(function, *args, **kwargs)
    finally:
        if clock:
            clock.change("score")


def score_call(function, *args, **kwargs):
    """Enter scoring without a training phase (reference and coupled rollout)."""
    clock = getattr(_LOCAL, "clock", None)
    if clock:
        clock.change("score")
    return function(*args, **kwargs)


def _phase_times(path, started):
    now = time.perf_counter()
    try:
        state = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "setup", dict(setup=now - started, fit=0.0, score=0.0)
    elapsed = dict(state["elapsed"])
    elapsed["setup"] += max(0.0, state["started"] - started)  # Interpreter/import startup.
    elapsed[state["phase"]] += max(0.0, now - state["since"])
    return state["phase"], elapsed


def _terminate_tree(process, descendants):
    if os.name != "nt":
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    else:
        # taskkill handles the live tree; sampled handles also cover children
        # whose parent exited before cleanup. psutil validates PID creation times.
        if process.poll() is None:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=subprocess.CREATE_NO_WINDOW, check=False)
    for child in descendants.values():
        with contextlib.suppress(Exception):
            child.kill()
    if process.poll() is None:
        process.kill()
    process.wait()


def reap_orphan(attempt):
    """Under the case lock, reap a prior worker left by a killed orchestrator."""
    path = Path(attempt) / "worker.json"
    if not path.exists():
        return
    identity = json.loads(path.read_text(encoding="utf-8"))
    if identity["hostname"] != socket.gethostname() or identity["create_time"] is None:
        return
    import psutil

    try:
        process = psutil.Process(identity["pid"])
        if process.create_time() != identity["create_time"]:
            return
        children = process.children(recursive=True)
        if os.name != "nt":
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        else:
            for child in reversed(children):
                with contextlib.suppress(psutil.Error):
                    child.kill()
            process.kill()
        psutil.wait_procs([process, *children], timeout=5)
    except psutil.NoSuchProcess:
        pass


def supervise(command, *, cwd, log_path, phase_path, policy, device="cpu", model=None):
    """Bound setup, fit and scoring independently and reap the whole process tree.

    KeyboardInterrupt is propagated without manufacturing a terminal target row.
    """
    limits = dict(setup=policy["setup_timeout"], score=policy["score_timeout"],
                  fit=(None if policy["timeout"] is None
                       else policy["timeout"] + policy["timeout_grace"]))
    resources = policy.get("resource_policy")
    if resources:
        validate_host(resources)
    started = time.perf_counter()
    descendants = {}
    outcome = None
    monitor = None
    environment = worker_environment(resources, model=model)
    environment.update(PYTHONIOENCODING="utf-8", MPLBACKEND="Agg",
                       KD_BENCHMARK_STARTED=repr(started))
    with Path(log_path).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            worker_command(command, resources), cwd=cwd, env=environment,
            stdout=log, stderr=subprocess.STDOUT,
            **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt"
               else {"start_new_session": True}),
        )
        try:
            # Windows affinity is set before Python reaches model imports. Linux
            # taskset applies it before exec, and descendants inherit it.
            if resources and os.name == "nt":
                import psutil
                psutil.Process(process.pid).cpu_affinity(resources["cpu_affinity"])
            monitor = ProcessTreeMonitor(process.pid,
                                         service_pids=resources["service_pids"] if resources else (),
                                         required=bool(resources))
            try:
                created = next((p.create_time() for p in monitor.roots if p.pid == process.pid), None) if monitor.psutil else None
            except Exception:
                created = None
            atomic_json(Path(log_path).parent / "worker.json",
                        dict(pid=process.pid, create_time=created, hostname=socket.gethostname()))
            while True:
                rss = monitor.sample()
                descendants = {key: child for key, child in monitor.processes.items()
                               if child.pid != process.pid}
                limited = bool(resources and rss is not None and rss > resources["memory_limit_mb"])
                if resources and monitor.error:
                    raise RuntimeError(f"Cannot enforce RSS limit: {monitor.error}")
                phase, elapsed = _phase_times(phase_path, started)
                expired = next((key for key, limit in limits.items()
                                if limit is not None and elapsed[key] > limit), None)
                code = process.poll()
                if limited or expired or code is not None:
                    outcome = dict(returncode=code, timeout_phase=expired, phase=phase,
                                   resource_limit="rss" if limited else None,
                                   **{f"{key}_seconds": value for key, value in elapsed.items()})
                    break
                time.sleep(resources["sample_interval"] if resources else 0.02)
        finally:
            # Service roots are long lived and remain alive after ordinary target
            # completion; on an RSS breach they are part of the offending tree.
            if monitor:
                monitor.sample()
                if not (outcome and outcome.get("resource_limit")):
                    descendants = {k: v for k, v in descendants.items() if k not in monitor.service_keys}
            _terminate_tree(process, descendants)
        outcome["worker_elapsed_seconds"] = time.perf_counter() - started
        outcome.update(monitor.usage())
        outcome.update(resource_slot=resources["slot"] if resources else "unconstrained",
                       resource_policy=resources, resource_policy_hash=policy_hash(resources),
                       thread_environment=({key: environment.get(key) for key in (
                           "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                           "NUMEXPR_NUM_THREADS", "JULIA_NUM_THREADS", "PYTHON_JULIACALL_THREADS",
                           "KD_SLOT_THREADS")} if resources else None),
                       gpu_occupied_seconds=(outcome["worker_elapsed_seconds"]
                                             if (resources and resources["slot"] == "gpu")
                                             or (not resources and str(device).startswith("cuda")) else 0.0),
                       gpu_occupied_source="exclusive_slot_wall" if resources and resources["slot"] == "gpu"
                                           else "device_assignment_wall" if str(device).startswith("cuda")
                                           else "cpu_no_gpu")
        # Durable supervision makes a finished worker recoverable even if the
        # orchestrator stops before promoting its candidate to a terminal row.
        atomic_json(Path(log_path).parent / "supervision.json", outcome)
        return outcome
