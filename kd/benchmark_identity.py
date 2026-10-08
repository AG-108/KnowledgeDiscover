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

SCHEMA = "benchmark-identity-v1"
DEPENDENCIES = (
    "numpy", "scipy", "pandas", "sympy", "scikit-learn", "matplotlib", "psutil",
    "torch", "gplearn", "pysindy", "pysr", "pyoperon", "physo", "DeePyMoD",
    "epde", "juliacall", "juliapkg", "sympytorch", "requests", "numba", "xarray",
)
ASSET_PARAMETERS = ("checkpoint_path", "source_dir", "model_path")


def file_digest(path):
    import hashlib

    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_fingerprint(root):
    root = Path(root)
    paths = [root / "run_benchmark.py"]
    for directory in ("kd", "scripts", "configs/benchmark"):
        paths.extend(path for path in (root / directory).rglob("*")
                     if path.is_file() and path.suffix in {".py", ".json", ".toml", ".sh"}
                     and "__pycache__" not in path.parts)
    paths.extend(root / name for name in ("pyproject.toml", "requirements.txt",
                                        "requirements-server-unified-py310.txt",
                                        "requirements-operon-py310.txt"))
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
        for pattern in case.get("dataset_metadata", {}).get("source_paths", []):
            matches = sorted(path for path in (root / "kd/dataset").glob(pattern) if path.is_file())
            if not matches:
                references[f"dataset:{pattern}"] = None
            for path in matches:
                asset_id = "dataset:" + path.relative_to(root / "kd/dataset").as_posix()
                assets[asset_id] = {"sha256": checked_hash(path), "bytes": path.stat().st_size}
                bindings[asset_id] = str(path.resolve())
                references[asset_id] = assets[asset_id]["sha256"]
                expected = case.get("dataset_metadata", {}).get("source_evidence", {}).get("asset_sha256", {}).get(
                    path.relative_to(root / "kd/dataset").as_posix())
                if expected and references[asset_id] != expected:
                    raise ValueError(f"Verified source asset content changed: {asset_id}")
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
    return cases
