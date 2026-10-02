"""Resource slots and sampled process-tree costs; no model imports."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil

RESOURCE_PROTOCOL = "slots-v1"
PARALLELISM_PROTOCOL = "d2-parallel-v1"
CPU_METHODS = frozenset({"dscv", "gplearn", "pysr", "pyoperon", "sindy", "pysindy",
                         "pdefind", "weakform", "sga", "pdefind_nd", "implicit_pde", "green_basis"})
GPU_METHODS = frozenset({"dso", "symbolicgpt", "physo", "e2e", "eqgpt", "deepmod", "dlga",
                         "pdenet", "spr", "llmsr", "green_dso"})
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
                    "JULIA_NUM_THREADS")


def slot_for_model(model):
    if model in CPU_METHODS or model == "reference":
        return "cpu"
    if model in GPU_METHODS:
        return "gpu"
    raise ValueError(f"No D2 resource slot declared for {model!r}")


def slot_policy(slot, *, cpus=None, gpu="0", memory_limit_mb=28_000_000_000 / (1024 ** 2),
                sample_interval=0.05, service_pids=()):
    if slot not in {"cpu", "gpu"}:
        raise ValueError(f"Invalid resource slot: {slot!r}")
    cpus = list(range(4) if slot == "cpu" else range(4, 8)) if cpus is None else list(cpus)
    if (len(cpus) != 4 or len(set(cpus)) != 4
            or any(isinstance(c, bool) or not isinstance(c, int) or c < 0 for c in cpus)):
        raise ValueError("Each D2 slot needs four distinct non-negative CPU indices")
    for key, value in (("memory_limit_mb", memory_limit_mb), ("sample_interval", sample_interval)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid {key}: {value!r}")
    if not isinstance(gpu, str) or not gpu or "," in gpu:
        raise ValueError("GPU slot requires one CUDA device index or UUID")
    pids = list(service_pids)
    if slot == "cpu" and pids:
        raise ValueError("External GPU services belong to the GPU slot")
    if any(isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 for pid in pids):
        raise ValueError("Service PIDs must be positive integers")
    return dict(version=RESOURCE_PROTOCOL, parallelism_protocol=PARALLELISM_PROTOCOL,
                slot=slot, cpu_affinity=cpus, threads=4,
                cuda_visible_devices="" if slot == "cpu" else gpu,
                memory_limit_mb=float(memory_limit_mb), sample_interval=float(sample_interval),
                service_pids=sorted(set(pids)))


def validate_policy(policy):
    expected = slot_policy(policy["slot"], cpus=policy["cpu_affinity"],
                           gpu=policy["cuda_visible_devices"] or "0",
                           memory_limit_mb=policy["memory_limit_mb"],
                           sample_interval=policy["sample_interval"], service_pids=policy["service_pids"])
    # Preserve the original C5 policy for existing immutable histories; new slot
    # bindings carry a distinct parallelism version and therefore a new hash.
    if "parallelism_protocol" not in policy:
        expected.pop("parallelism_protocol")
    if policy != expected:
        raise ValueError("Invalid or unsupported resource policy")
    return expected


def policy_hash(policy):
    return (hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()
            if policy else "unconstrained")


def slot_model_params(model, params, *, profile, cores):
    """Bind parallelism to the allocated cores without changing search budgets."""
    params = copy.deepcopy(params)
    key = {"gplearn": "n_jobs", "pyoperon": "n_threads", "llmsr": "num_evaluators"}.get(model)
    if key:
        requested = params.get(key, 4 if model == "llmsr" else 1)
        if isinstance(requested, bool) or not isinstance(requested, int) or requested == 0:
            raise ValueError(f"Invalid {model} {key}: {requested!r}")
        if requested < 0 and not (model == "gplearn" and requested == -1):
            raise ValueError(f"Invalid {model} {key}: {requested!r}")
        params[key] = cores if profile == "full" or requested < 0 else min(requested, cores)
    if model == "pysr":
        if profile == "full" or params.get("parallelism") != "serial":
            params.update(parallelism="multiprocessing", procs=cores, deterministic=False)
        else:
            params.pop("procs", None)
    return params


def worker_environment(policy, environ=None, *, model=None):
    env = dict(os.environ if environ is None else environ)
    if policy:
        validate_policy(policy)
        # The search/evaluator pool supplies the parallelism for these methods.
        # Each numerical library gets one thread per task instead of 4 x 4.
        inner_threads = (1 if policy.get("parallelism_protocol") == PARALLELISM_PROTOCOL
                         and model in {"gplearn", "pyoperon", "pysr", "llmsr"}
                         else policy["threads"])
        env.update({key: str(inner_threads) for key in THREAD_VARIABLES})
        # JuliaCall otherwise defaults to "auto" and can override JULIA_NUM_THREADS.
        if policy.get("parallelism_protocol") == PARALLELISM_PROTOCOL:
            env["PYTHON_JULIACALL_THREADS"] = str(inner_threads)
        env.update(CUDA_VISIBLE_DEVICES=policy["cuda_visible_devices"],
                   KD_SLOT_THREADS=str(policy["threads"]))
    return env


def worker_command(command, policy):
    if not policy or os.name == "nt":
        return list(command)
    taskset = shutil.which("taskset")
    if not taskset:
        raise RuntimeError("D2 slot execution requires taskset on Linux")
    return [taskset, "--cpu-list", ",".join(map(str, policy["cpu_affinity"])), *command]


def validate_host(policy):
    """Fail before spawning if affinity cannot be enforced; never silently fall back."""
    import psutil

    validate_policy(policy)
    available = psutil.Process().cpu_affinity()
    if not set(policy["cpu_affinity"]) <= set(available):
        raise ValueError(f"Slot CPUs {policy['cpu_affinity']} outside available affinity {available}")
    if os.name != "nt" and not shutil.which("taskset"):
        raise RuntimeError("D2 slot execution requires taskset on Linux")
    for pid in policy["service_pids"]:
        process = psutil.Process(pid)
        if set(process.cpu_affinity()) != set(policy["cpu_affinity"]):
            raise ValueError(f"Service PID {pid} must already be bound to the GPU slot CPUs")


class ProcessTreeMonitor:
    """Sum user+system CPU per observed PID/create-time, including exited children.

    CPU and RSS are sampled estimates. Children born and reaped between polls can
    be missed; children CPU reported by the parent is deliberately not added,
    since that would double count sampled children.
    """

    def __init__(self, pid, *, service_pids=(), required=False):
        self.worker_pid = pid
        self.processes = {}
        self.service_keys = set()
        self.cpu = {}
        self.baseline = {}
        self.peak_memory_mb = None
        self.error = None
        try:
            import psutil
        except ImportError:
            if required:
                raise RuntimeError("Slot enforcement requires psutil") from None
            self.psutil = None
            return
        self.psutil = psutil
        self.roots = []
        for root_pid in (pid, *service_pids):
            try:
                root = psutil.Process(root_pid)
                self.roots.append(root)
                self._remember(root)
                if root_pid in service_pids:
                    for proc in [root, *root.children(recursive=True)]:
                        key = self._remember(proc)
                        self.service_keys.add(key)
                        times = proc.cpu_times()
                        self.baseline[key] = times.user + times.system
            except psutil.Error as exc:
                if root_pid in service_pids:
                    raise
                if required and not isinstance(exc, psutil.NoSuchProcess):
                    raise

    def _remember(self, process):
        key = (process.pid, process.create_time())
        self.processes[key] = process
        return key

    def sample(self):
        if self.psutil is None:
            return None
        for root in self.roots:
            try:
                for child in root.children(recursive=True):
                    key = self._remember(child)
                    if root.pid != self.worker_pid:
                        self.service_keys.add(key)
            except self.psutil.NoSuchProcess:
                pass
            except self.psutil.AccessDenied:
                self.error = "access_denied"
        rss = 0
        for key, proc in list(self.processes.items()):
            try:
                if not proc.is_running():
                    self.processes.pop(key, None)
                    continue
                times = proc.cpu_times()
                self.cpu[key] = max(self.cpu.get(key, 0),
                                    times.user + times.system - self.baseline.get(key, 0))
                rss += proc.memory_info().rss
            except self.psutil.NoSuchProcess:
                self.processes.pop(key, None)
            except self.psutil.AccessDenied:
                self.error = "access_denied"
        mb = rss / (1024 ** 2)
        self.peak_memory_mb = max(self.peak_memory_mb or 0, mb)
        return mb

    def usage(self):
        return dict(cpu_seconds=sum(self.cpu.values()) if self.cpu else None,
                    worker_cpu_seconds=sum(value for key, value in self.cpu.items()
                                           if key[0] == self.worker_pid),
                    peak_memory_mb=self.peak_memory_mb,
                    resource_monitor_source="psutil_process_tree_sampled" if self.psutil else "unavailable",
                    resource_monitor_error=self.error)
