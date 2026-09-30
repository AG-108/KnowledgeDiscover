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


class SkipCase(Exception):
    """Unsupported capability or source prerequisite, shared by CLI and imports."""


EXECUTION_PROTOCOL_VERSION = "target-v1"
TERMINAL_STATUSES = frozenset({"ok", "error", "timeout", "skipped"})
STOP_REASONS = (
    "budget_exhausted", "time_limit", "converged", "early_stop", "hard_timeout", "error",
)
_LOCAL = threading.local()


def execution_policy(timeout=None, *, timeout_grace=5.0, setup_timeout=300.0,
                     score_timeout=300.0):
    values = dict(timeout=timeout, timeout_grace=timeout_grace,
                  setup_timeout=setup_timeout, score_timeout=score_timeout)
    for key, value in values.items():
        if key == "timeout" and value is None:
            continue
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0
                or (key != "timeout_grace" and value == 0)):
            raise ValueError(f"Invalid {key}: {value!r}")
    return {"version": EXECUTION_PROTOCOL_VERSION, **values}


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

    def __init__(self, path=None, started=None):
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
    try:
        return function(*args, **kwargs)
    finally:
        if clock:
            clock.change("score")


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


def supervise(command, *, cwd, log_path, phase_path, policy):
    """Bound setup, fit and scoring independently and reap the whole process tree.

    KeyboardInterrupt is propagated without manufacturing a terminal target row.
    """
    limits = dict(setup=policy["setup_timeout"], score=policy["score_timeout"],
                  fit=(None if policy["timeout"] is None
                       else policy["timeout"] + policy["timeout_grace"]))
    started = time.perf_counter()
    descendants = {}
    try:
        import psutil
    except ImportError:
        psutil = None
    with Path(log_path).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=cwd,
            env=dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg",
                     KD_BENCHMARK_STARTED=repr(started)),
            stdout=log, stderr=subprocess.STDOUT,
            **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt"
               else {"start_new_session": True}),
        )
        try:
            try:
                created = psutil.Process(process.pid).create_time() if psutil else None
            except psutil.NoSuchProcess:
                created = None
            atomic_json(Path(log_path).parent / "worker.json",
                        dict(pid=process.pid, create_time=created, hostname=socket.gethostname()))
            while True:
                if os.name == "nt" and psutil is not None:
                    try:
                        for child in psutil.Process(process.pid).children(recursive=True):
                            descendants[(child.pid, child.create_time())] = child
                    except psutil.Error:
                        pass
                phase, elapsed = _phase_times(phase_path, started)
                expired = next((key for key, limit in limits.items()
                                if limit is not None and elapsed[key] > limit), None)
                code = process.poll()
                if expired or code is not None:
                    return dict(returncode=code, timeout_phase=expired, phase=phase,
                                worker_elapsed_seconds=time.perf_counter() - started,
                                **{f"{key}_seconds": value for key, value in elapsed.items()})
                time.sleep(0.02)
        finally:
            _terminate_tree(process, descendants)
