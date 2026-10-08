"""Execute manifests in the two D2 slots (one case at a time per slot).

Targets within each case run sequentially in isolated workers. Slot locks also
cover preparation and scoring, so neither phase overlaps another fit in that slot.
Dry-run writes only the resource-bound manifest and never loads data or models.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kd.benchmark_batch import organize_by_baseline, round_robin, save_checkpoint  # noqa: E402
from kd.benchmark_identity import attach as attach_identity  # noqa: E402
from kd.benchmark_identity import canonical_case, restore_paths  # noqa: E402
from kd.benchmark_identity import verify as verify_identity  # noqa: E402
from kd.benchmark_protocol import (  # noqa: E402
    inherited_policy,
    read_manifests,
    validate_main_matrix,
    validate_shared_manifests,
)
from kd.benchmark_resources import (  # noqa: E402
    bind_services,
    gpu_uuid,
    host_hardware_profile,
    slot_for_model,
    slot_model_params,
    slot_policy,
    validate_host,
)
from kd.benchmark_store import freeze_run, verify_freeze  # noqa: E402
from kd.dataset._metadata import DATASET_PROTOCOL_VERSION  # noqa: E402
from run_benchmark import (  # noqa: E402
    EXECUTION_PROTOCOL_VERSION,
    case_lock,
    execute_case,
    execution_policy,
    save_manifest,
    write_json,
)


def bind_case(case, slots):
    """Resource/device changes form a new identity; source cases stay untouched."""
    bound = copy.deepcopy(case)
    slot = slot_for_model(bound["model"])
    bound["resource_policy"] = copy.deepcopy(slots[slot])
    bound["device"] = "cuda:0" if slot == "gpu" else "cpu"
    bound["model_params"] = slot_model_params(bound["model"], bound["model_params"],
                                             profile=bound["profile"], cores=slots[slot]["threads"])
    if "device" in bound["model_params"]:
        bound["model_params"]["device"] = bound["device"]
    bound["source_case_name"] = bound.pop("name")
    digest = hashlib.sha256(json.dumps(canonical_case(bound), sort_keys=True).encode()).hexdigest()[:12]
    bound["name"] = f"{bound['model']}__{bound['dataset']}__s{bound['seed']}__{digest}"
    return bound


def load_cases(paths, slots, excluded=()):
    cases, sources = {}, []
    for path in paths:
        path = Path(path).resolve()
        raw = path.read_bytes()
        manifest = json.loads(raw)
        bindings_path = path.parent / "runtime_asset_paths.json"
        bindings = json.loads(bindings_path.read_text(encoding="utf-8")) if bindings_path.is_file() else {}
        source_cases = restore_paths(manifest["cases"], bindings)
        if manifest.get("identity"):
            verify_identity(manifest["identity"], ROOT, source_cases)
        if manifest.get("execution_protocol_version") != EXECUTION_PROTOCOL_VERSION:
            raise ValueError("Legacy manifest; generate a new target-v2 manifest")
        sources.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest()))
        for case in source_cases:
            if case.get("dataset_protocol_version") != DATASET_PROTOCOL_VERSION:
                raise ValueError(f"Legacy dataset protocol; generate a new {DATASET_PROTOCOL_VERSION} manifest")
            if case["model"] in excluded:
                continue
            bound = bind_case(case, slots)
            if bound["name"] in cases and cases[bound["name"]] != bound:
                raise ValueError("Conflicting case identity in source manifests")
            cases[bound["name"]] = bound
    return list(cases.values()), sources


def run_slot(slot, cases, output, timeout, timing):
    # This lock prevents a second executor on the same output directory from
    # scheduling overlapping fits on the slot, even across different cases.
    with contextlib.ExitStack() as locks:
        locks.enter_context(case_lock(output / ".slots" / slot))
        if cases:
            resources = cases[0]["resource_policy"]
            lock_root = Path(tempfile.gettempdir()) / "kd-benchmark-slot-locks"
            keys = [f"cpu-{cpu}" for cpu in resources["cpu_affinity"]]
            if slot == "gpu":
                gpu = hashlib.sha256(resources["cuda_visible_devices"].encode()).hexdigest()[:16]
                keys.append(f"gpu-{gpu}")
            for key in sorted(keys):
                locks.enter_context(case_lock(lock_root / key))
        for case in round_robin(cases):
            validate_host(case["resource_policy"])
            try:
                rows = execute_case(case, output, timeout=timeout, resume=True, **timing)
            except Exception as exc:
                yield case, None, f"{type(exc).__name__}: {exc}"
            else:
                yield case, rows, None


def _cpus(value):
    try:
        result = [int(part) for part in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("Use four comma-separated CPU indices") from None
    if len(result) != 4 or len(set(result)) != 4 or min(result) < 0:
        raise argparse.ArgumentTypeError("Use four distinct non-negative CPU indices")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude-model", action="append", default=[])
    parser.add_argument("--cpu-cpus", type=_cpus, default=[0, 1, 2, 3])
    parser.add_argument("--gpu-cpus", type=_cpus, default=[4, 5, 6, 7])
    parser.add_argument("--gpu-device", default="0", help="Physical CUDA index or GPU UUID; workers see cuda:0")
    parser.add_argument("--gpu-service-pid", action="append", type=int, default=[],
                        help="Already bound vLLM service PID; GPU queue must contain only llmsr")
    parser.add_argument("--memory-limit-gb", type=float, default=28.0, help="RSS limit per slot, in decimal GB")
    parser.add_argument("--sample-interval", type=float, default=0.05)
    parser.add_argument("--timeout", type=float, default=None, help="Fit seconds per target (inherits source manifest)")
    parser.add_argument("--soft-deadline", action=argparse.BooleanOptionalAction, default=None,
                        help="Return the best completed candidate at T (default enabled)")
    parser.add_argument("--timeout-grace", type=float, default=None)
    parser.add_argument("--setup-timeout", type=float, default=None)
    parser.add_argument("--score-timeout", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--portable-resources", action="store_true",
                        help="Allow new CPU indices/GPU UUID only on the same verified hardware profile")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if set(args.cpu_cpus) & set(args.gpu_cpus):
        raise SystemExit("CPU and GPU slot affinities must be disjoint")
    try:
        source_manifests = read_manifests(args.manifest)
        validate_shared_manifests(source_manifests)
        inherited = inherited_policy(source_manifests, {key: getattr(args, key) for key in
                                     ("timeout", "soft_deadline", "timeout_grace", "setup_timeout", "score_timeout")})
        args.timeout = inherited.pop("timeout")
        timing = inherited
        policy = execution_policy(args.timeout, **timing)
        if not args.dry_run:
            args.gpu_device = gpu_uuid(args.gpu_device) if any(
                slot_for_model(case["model"]) == "gpu" for manifest in source_manifests
                for case in manifest["cases"] if case["model"] not in args.exclude_model) else args.gpu_device
        bindings = bind_services(args.gpu_service_pid) if args.gpu_service_pid else []
        slots = {slot: slot_policy(slot, cpus=cpus, gpu=args.gpu_device,
                                  memory_limit_mb=args.memory_limit_gb * 1000 ** 3 / (1024 ** 2),
                                  sample_interval=args.sample_interval,
                                  service_pids=args.gpu_service_pid if slot == "gpu" else (),
                                  hardware_profile=host_hardware_profile(slot, args.gpu_device)
                                  if args.portable_resources else None)
                 for slot, cpus in (("cpu", args.cpu_cpus), ("gpu", args.gpu_cpus))}
        if bindings:
            slots["gpu"]["service_bindings"] = bindings
            slots["gpu"]["service_identity"] = source_manifests[0].get("service_identity")
            if source_manifests[0].get("benchmark_mode") == "main" and not slots["gpu"]["service_identity"]:
                raise ValueError("Formal LLM-SR requires a verified model/weight service_identity")
        cases, sources = load_cases(args.manifest, slots, args.exclude_model)
        modes = {manifest.get("benchmark_mode", "diagnostic") for manifest in source_manifests}
        if len(modes) != 1:
            raise ValueError("Main, diagnostic and control manifests must run separately")
        mode = next(iter(modes))
        configuration_sources = source_manifests[0].get("configuration_sources", {})
        if mode == "main":
            validate_main_matrix(cases, policy, configuration_sources=configuration_sources)
        if not cases:
            raise ValueError("No cases selected")
        if args.gpu_service_pid and any(c["model"] != "llmsr" for c in cases
                                       if c["resource_policy"]["slot"] == "gpu"):
            raise ValueError("A live vLLM service requires an llmsr-only GPU queue")
        if not args.dry_run and any(c["model"] == "llmsr" for c in cases) and not args.gpu_service_pid:
            raise ValueError("D2 llmsr execution requires --gpu-service-pid for its local vLLM process tree")
        if not args.dry_run:
            for slot in {c["resource_policy"]["slot"] for c in cases}:
                validate_host(slots[slot])
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    output = args.output_dir.resolve()
    manifest = dict(execution_protocol_version=EXECUTION_PROTOCOL_VERSION,
                               benchmark_mode=mode, configuration_sources=configuration_sources,
                               identity=source_manifests[0].get("identity"),
                               parent_matrix_hash=source_manifests[0].get("parent_matrix_hash"),
                               experiment_identity_hash=source_manifests[0].get("experiment_identity_hash"),
                               dataset_protocol_version=DATASET_PROTOCOL_VERSION,
                               execution_policy=policy, resource_slots=slots,
                               source_manifests=sources, cases=cases,
                               created_at=datetime.now(timezone.utc).isoformat())
    bound_sources = [case for source in source_manifests for case in source["cases"]]
    if manifest.get("identity") and any(not case.get("resource_policy") for case in bound_sources):
        # Hardware binding creates a new parent matrix. Subsequent chunks retain
        # this identity and must use the same stable resources.
        asset_paths = attach_identity(manifest, ROOT)
        cases = manifest["cases"]
    else:
        asset_paths = {}
        for path in args.manifest:
            mapping = path.parent / "runtime_asset_paths.json"
            if mapping.is_file():
                asset_paths.update(json.loads(mapping.read_text(encoding="utf-8")))
        source_by_name = {case["name"]: case for case in bound_sources}
        for case in cases:
            source = source_by_name.get(case.get("source_case_name"))
            if source and source.get("resource_policy") and canonical_case(source) != canonical_case(case):
                raise ValueError("Frozen D2 case resources/configuration changed; bind a new parent matrix")
    save_manifest(output, manifest)
    write_json(output / "runtime_asset_paths.json", asset_paths)
    print(f"Selected {len(cases)} cases; D2 CPU/GPU slots; output={output}", flush=True)
    if args.dry_run:
        return 0
    if (output / "benchmark_freeze.json").exists():
        verify_freeze(output)

    # Each slot has one dedicated thread. Events are consumed by the main thread
    # to keep summary/checkpoint writes serialized.
    import queue

    events = queue.Queue()

    def consume(slot, selected):
        try:
            for event in run_slot(slot, selected, output, args.timeout, timing):
                events.put(event)
        finally:
            events.put(None)

    results, errors = {}, {}
    started_at = datetime.now(timezone.utc).isoformat()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(consume, slot, [c for c in cases if c["resource_policy"]["slot"] == slot])
                   for slot in ("cpu", "gpu")]
        finished = 0
        while finished < 2:
            event = events.get()
            if event is None:
                finished += 1
                continue
            case, rows, error = event
            if error:
                errors[case["name"]] = error
                write_json(output / "orchestration_errors.json", errors)
                print(f"Orchestration error: {case['name']}: {error}", flush=True)
            else:
                results[case["name"]] = rows
                print(f"[{len(results)}/{len(cases)}] {case['name']}: "
                      + ",".join(r["status"] for r in rows), flush=True)
            save_checkpoint(output, cases, results, started_at=started_at, attempt=1, retries=0)
        for future in as_completed(futures):
            future.result()
    save_checkpoint(output, cases, results, started_at=started_at, attempt=1, retries=0)
    rows = [row for group in results.values() for row in group]
    organize_by_baseline(output, rows)
    if manifest.get("identity"):
        freeze_run(output)
    return int(bool(errors) or any(r["status"] in {"error", "timeout"} for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
