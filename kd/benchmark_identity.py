"""Portable code/environment/asset fingerprints and immutable experiment identities."""

from __future__ import annotations

import copy
import importlib.metadata
import importlib.util
import os
import platform
import subprocess
import sys
from pathlib import Path

from kd.benchmark_protocol import digest

SCHEMA = "benchmark-identity-v2"
DEPENDENCIES = (
    "numpy", "scipy", "pandas", "sympy", "scikit-learn", "matplotlib", "psutil",
    "torch", "gplearn", "pysindy", "pysr", "pyoperon", "physo", "DeePyMoD",
    "epde", "juliacall", "juliapkg", "sympytorch", "requests", "numba", "xarray",
)
ASSET_PARAMETERS = ("checkpoint_path", "source_dir", "model_path")
DATASET_ASSET_PARAMETERS = ("root", "benchmark_source")
METHOD_ASSETS = {
    "eqgpt": ("kd/model/eqgpt/data/dict_datas_0725.json",
              "kd/model/eqgpt/data/PDE_dataset_0725.xlsx"),
}


def _dataset_sources(root, case):
    """Resolve the files used by loaders without importing or loading data."""
    params = case.get("dataset_params", {})
    dataset_root = Path(root) / "kd/dataset"
    symbolic = case.get("source") == "symbolic"
    root_capable = symbolic or case.get("dataset", "").startswith(
        ("fde_", "dso_file_", "green_")
    ) or case.get("dataset") == "pyqg_subgrid"
    overridden_root = root_capable and params.get("root") is not None
    default_base = dataset_root / "data" if symbolic else dataset_root
    base = Path(params["root"]) if overridden_root else default_base
    base = base.resolve()
    if symbolic:
        benchmark_source = params.get("benchmark_source", "benchmarks.csv")
        benchmark = Path(benchmark_source)
        if not benchmark.is_absolute():
            benchmark = base / benchmark
        sources = [(logical, [path] if path.is_file() else []) for logical, path in (
            ("data/benchmarks.csv", benchmark.resolve()),
            ("data/function_sets.csv", base / "function_sets.csv"),
        )]
        parameters = {"root": base} if overridden_root else {}
        if "benchmark_source" in params:
            parameters["benchmark_source"] = benchmark.resolve()
        return sources, parameters
    sources = []
    for pattern in case.get("dataset_metadata", {}).get("source_paths", []):
        matches = sorted(path for path in base.glob(pattern) if path.is_file())
        if matches:
            sources.extend((path.relative_to(base).as_posix(), [path]) for path in matches)
        else:
            sources.append((pattern, []))
    return sources, {"root": base} if overridden_root else {}


def file_digest(path):
    import hashlib

    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_fingerprint(root):
    root = Path(root)
    paths = [root / "run_benchmark.py", root / "examples/_common.py"]
    for directory in ("kd", "scripts", "server/launch", "configs/benchmark"):
        paths.extend(path for path in (root / directory).rglob("*")
                     if path.is_file() and path.suffix in {".py", ".json", ".toml", ".sh"}
                     and "__pycache__" not in path.parts)
    paths.extend(root / name for name in ("pyproject.toml", "requirements/core.txt",
                                        "requirements/server-unified-py310.txt",
                                        "requirements/server-operon-py310.txt"))
    files = {path.relative_to(root).as_posix(): file_digest(path)
             for path in sorted(set(paths)) if path.is_file()}
    return {"sha256": digest(files), "files": files}


def environment_fingerprint():
    packages = {}
    for name in DEPENDENCIES:
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
            continue
        record = distribution.read_text("RECORD")
        implementations = {}
        for item in distribution.files or ():
            if Path(str(item)).suffix in {".py", ".pyd", ".so"}:
                path = distribution.locate_file(item)
                if path.is_file():
                    implementations[str(item).replace("\\", "/")] = file_digest(path)
        packages[name] = {"version": distribution.version,
                          "record_sha256": digest(record) if record else None,
                          "implementation_sha256": digest(implementations)}
    projects = {}
    for variable in ("JULIA_PROJECT", "PYTHON_JULIAPKG_PROJECT"):
        value = os.environ.get(variable)
        if value and not value.startswith("@"):
            for name in ("Project.toml", "Manifest.toml"):
                path = Path(value) / name
                if path.is_file():
                    projects[f"{variable}/{name}"] = file_digest(path)
    if packages.get("juliapkg") and not os.environ.get("PYTHON_JULIAPKG_PROJECT"):
        # Mirror juliapkg.state without resolving or installing Julia.
        prefix = sys.prefix if sys.prefix != sys.base_prefix else os.environ.get("CONDA_PREFIX")
        depot = os.environ.get("JULIA_DEPOT_PATH", str(Path.home() / ".julia")).split(os.pathsep)[0]
        project = Path(prefix) / "julia_env" if prefix else Path(depot) / "environments/pyjuliapkg"
        for name in ("Project.toml", "Manifest.toml"):
            path = project / name
            if path.is_file():
                projects[f"juliapkg_default/{name}"] = file_digest(path)
    julia = os.environ.get("PYTHON_JULIAPKG_EXE")
    native = {"julia_executable_sha256": file_digest(julia) if julia and Path(julia).is_file() else None,
              "julia_projects": projects}
    # CMake installs SISSO++ without Python distribution metadata. Bind both the
    # extension and its Python wrappers without importing the native runtime.
    extension = importlib.util.find_spec("_sissopp")
    package = importlib.util.find_spec("sissopp")
    wrappers = {}
    if package and package.submodule_search_locations:
        for location in package.submodule_search_locations:
            directory = Path(location)
            wrappers.update({p.relative_to(directory).as_posix(): file_digest(p)
                             for p in sorted(directory.rglob("*.py"))})
    native["sissopp"] = {
        "extension_sha256": file_digest(extension.origin)
        if extension and extension.origin and Path(extension.origin).is_file() else None,
        "python_sha256": digest(wrappers) if wrappers else None,
    }
    return {"python": list(sys.version_info[:3]), "implementation": platform.python_implementation(),
            "system": platform.system(), "machine": platform.machine(),
            "packages": packages, "native_runtime": native}


def asset_fingerprints(root, cases):
    root = Path(root)
    assets, bindings, per_case = {}, {}, {}
    file_hashes, tree_hashes = {}, {}

    def checked_hash(path):
        key = str(path.resolve())
        if key not in file_hashes:
            file_hashes[key] = file_digest(path)
        return file_hashes[key]
    for case in cases:
        references = {}
        sources, dataset_parameters = _dataset_sources(root, case)
        dataset_references = {}
        benchmark_hash = None
        for logical_path, matches in sources:
            if not matches:
                dataset_references[f"dataset:{logical_path}"] = None
            for path in matches:
                fingerprint = checked_hash(path)
                # Different roots may contain different bytes for the same
                # logical source. Neither their records nor bindings may alias.
                asset_id = f"dataset:{logical_path}:sha256:{fingerprint}"
                assets[asset_id] = {"sha256": fingerprint, "bytes": path.stat().st_size}
                bindings[asset_id] = str(path.resolve())
                dataset_references[asset_id] = fingerprint
                if logical_path == "data/benchmarks.csv":
                    benchmark_hash = fingerprint
                expected = case.get("dataset_metadata", {}).get("source_evidence", {}).get("asset_sha256", {}).get(
                    logical_path)
                params = case.get("dataset_params", {})
                symbolic_override = (case.get("source") == "symbolic"
                                     and (params.get("root") is not None or "benchmark_source" in params))
                if expected and not symbolic_override and fingerprint != expected:
                    raise ValueError(f"Verified source asset content changed: {asset_id}")
        references.update(dataset_references)
        for parameter, path in dataset_parameters.items():
            fingerprint = (benchmark_hash or digest({"missing": "benchmark_source"})
                           if parameter == "benchmark_source" else digest(dataset_references))
            references[f"dataset.{parameter}"] = fingerprint
            bindings[f"dataset:{parameter}:{fingerprint}"] = str(path)
        for relative in METHOD_ASSETS.get(case["model"], ()):
            asset_id = f"method:{case['model']}:{relative}"
            path = root / relative
            if not path.is_file():
                references[asset_id] = None
                continue
            assets[asset_id] = {"sha256": checked_hash(path), "bytes": path.stat().st_size}
            bindings[asset_id] = str(path.resolve())
            references[asset_id] = assets[asset_id]["sha256"]
        for parameter in ASSET_PARAMETERS:
            value = case.get("model_params", {}).get(parameter)
            if not value:
                continue
            path = Path(value)
            if not path.is_absolute():
                path = root / path
            asset_id = f"method:{case['model']}:{parameter}"
            if path.is_dir():
                key = str(path.resolve())
                if key not in tree_hashes:
                    tree_hashes[key] = {p.relative_to(path).as_posix(): checked_hash(p)
                                       for p in sorted(path.rglob("*.py")) if "__pycache__" not in p.parts}
                contents = tree_hashes[key]
                if not contents:
                    raise ValueError(f"External source tree has no Python files: {asset_id}")
                assets[asset_id] = {"sha256": digest(contents), "files": contents}
            elif path.is_file():
                assets[asset_id] = {"sha256": checked_hash(path), "bytes": path.stat().st_size}
            else:
                raise FileNotFoundError(f"Missing declared asset: {asset_id}")
            bindings[asset_id] = str(path.resolve())
            references[parameter] = assets[asset_id]["sha256"]
        per_case[case["name"]] = references
    return assets, bindings, per_case


def stable_resources(policy):
    if not policy:
        return policy
    policy = copy.deepcopy(policy)
    # slots-v1 keeps its historic PID-sensitive identity; new histories separate
    # immutable resource declarations from live process bindings.
    if policy.get("version") in {"slots-v2", "slots-v3"}:
        policy.pop("service_pids", None)
        policy.pop("service_bindings", None)
    if policy.get("version") == "slots-v3":
        # Physical bindings change across containers; the four-core allocation
        # and verified hardware profile remain scientific identity fields.
        policy["cpu_affinity"] = list(range(4) if policy["slot"] == "cpu" else range(4, 8))
        policy["cuda_visible_devices"] = "exclusive_gpu" if policy["slot"] == "gpu" else ""
    return policy


def canonical_case(case):
    value = copy.deepcopy(case)
    for key in ("name", "source_case_name", "runtime_bindings", "experiment_identity_hash"):
        value.pop(key, None)
    if value.get("resource_policy"):
        value["resource_policy"] = stable_resources(value["resource_policy"])
    for parameter in ASSET_PARAMETERS:
        if parameter in value.get("model_params", {}):
            fingerprint = value.get("asset_fingerprints", {}).get(parameter)
            if fingerprint:
                value["model_params"][parameter] = "sha256:" + fingerprint
    for parameter in DATASET_ASSET_PARAMETERS:
        fingerprint = value.get("asset_fingerprints", {}).get(f"dataset.{parameter}")
        if fingerprint and parameter in value.get("dataset_params", {}):
            value["dataset_params"][parameter] = "sha256:" + fingerprint
    return value


def stable_execution(policy):
    if policy is None:
        return None
    value = copy.deepcopy(policy)
    if value.get("resource_policy"):
        value["resource_policy"] = stable_resources(value["resource_policy"])
    return value


def capture(root, cases):
    assets, bindings, per_case = asset_fingerprints(root, cases)
    identity = {"schema": SCHEMA, "code": source_fingerprint(root),
                "environment": environment_fingerprint(), "assets": assets}
    identity["sha256"] = digest(identity)
    provenance = {}
    for key, arguments in (("git_commit", ["rev-parse", "HEAD"]),
                           ("git_dirty", ["status", "--porcelain"])):
        result = subprocess.run(["git", "-C", str(root), *arguments], capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=10)
        provenance[key] = result.stdout.strip() if result.returncode == 0 else None
    identity["provenance"] = provenance
    return identity, bindings, per_case


def attach(manifest, root):
    """Capture once at launch, before any data preparation or model construction."""
    identity, bindings, per_case = capture(root, manifest["cases"])
    cases = copy.deepcopy(manifest["cases"])
    for case in cases:
        case["asset_fingerprints"] = per_case[case["name"]]
        if case.get("source") == "symbolic":
            metadata = case.get("dataset_metadata", {})
            evidence = metadata.get("source_evidence", {})
            mismatches = []
            for logical, expected in evidence.get("asset_sha256", {}).items():
                prefix = f"dataset:{logical}:sha256:"
                actual = next((value for key, value in case["asset_fingerprints"].items()
                               if key.startswith(prefix)), None)
                if actual is not None and actual != expected:
                    mismatches.append(logical)
            if mismatches:
                # A different definition is a diagnostic input, not the
                # certified named benchmark represented by the original card.
                metadata["source_status"] = "custom_definition_unverified"
                metadata["source_evidence"] = {
                    "unverified_assets": mismatches,
                    "declared_source_evidence": copy.deepcopy(evidence),
                }
                if "data/benchmarks.csv" in mismatches:
                    metadata["reference_status"] = "unknown"
                    case["ground_truth"] = None
        # Resolve relative overrides before workers change their working
        # directory; the public manifest records only the logical binding.
        for parameter in DATASET_ASSET_PARAMETERS:
            fingerprint = case["asset_fingerprints"].get(f"dataset.{parameter}")
            if fingerprint and parameter in case.get("dataset_params", {}):
                case["dataset_params"][parameter] = bindings[f"dataset:{parameter}:{fingerprint}"]
    matrix = [canonical_case(case) for case in cases]
    matrix.sort(key=digest)
    matrix_hash = digest({"cases": matrix, "execution_policy": stable_execution(manifest["execution_policy"]),
                          "benchmark_mode": manifest.get("benchmark_mode", "diagnostic")})
    run_hash = digest({"implementation": identity["sha256"], "matrix": matrix_hash})
    for case in cases:
        case["experiment_identity_hash"] = run_hash
    manifest.update(identity=identity, parent_matrix_hash=matrix_hash,
                    experiment_identity_hash=run_hash, cases=cases)
    return bindings


def verify(identity, root, cases):
    """Reject changed implementations/assets even if the protocol number is unchanged."""
    current, _, per_case = capture(root, cases)
    for field in ("code", "environment"):
        if current[field] != identity.get(field):
            raise ValueError(f"Frozen experiment {field} changed; create a new run identity")
    for asset_id, asset in current["assets"].items():
        if asset != identity.get("assets", {}).get(asset_id):
            raise ValueError(f"Frozen input asset changed: {asset_id}")
    for case in cases:
        if case.get("asset_fingerprints") != per_case[case["name"]]:
            raise ValueError(f"Frozen asset availability/content changed for {case['name']}")


def public_manifest(manifest):
    """Machine paths live in a separate local mapping, outside scientific identity."""
    value = copy.deepcopy(manifest)
    value.pop("python", None)
    for key in ("config", "compatibility_report"):
        if value.get(key):
            value[key] = Path(value[key]).name
    for item in value.get("source_manifests", []):
        if isinstance(item, dict) and item.get("path"):
            item["path"] = Path(item["path"]).name
    if value.get("source_manifests") and all(isinstance(item, str) for item in value["source_manifests"]):
        value["source_manifests"] = [Path(item).name for item in value["source_manifests"]]
    for case in value.get("cases", []):
        for parameter in ASSET_PARAMETERS:
            if case.get("asset_fingerprints", {}).get(parameter):
                case["model_params"][parameter] = f"asset:method:{case['model']}:{parameter}"
        for parameter in DATASET_ASSET_PARAMETERS:
            fingerprint = case.get("asset_fingerprints", {}).get(f"dataset.{parameter}")
            if fingerprint and parameter in case.get("dataset_params", {}):
                case["dataset_params"][parameter] = f"asset:dataset:{parameter}:{fingerprint}"
    return value


def restore_paths(cases, bindings):
    cases = copy.deepcopy(cases)
    for case in cases:
        for parameter in ASSET_PARAMETERS:
            value = case.get("model_params", {}).get(parameter)
            if isinstance(value, str) and value.startswith("asset:"):
                asset_id = value.removeprefix("asset:")
                if asset_id not in bindings:
                    raise ValueError(f"Missing local asset binding: {asset_id}")
                case["model_params"][parameter] = bindings[asset_id]
        for parameter in DATASET_ASSET_PARAMETERS:
            value = case.get("dataset_params", {}).get(parameter)
            if isinstance(value, str) and value.startswith("asset:"):
                asset_id = value.removeprefix("asset:")
                if asset_id not in bindings:
                    raise ValueError(f"Missing local asset binding: {asset_id}")
                case["dataset_params"][parameter] = bindings[asset_id]
    return cases
