"""Export asset hashes, provenance, availability and dataset mappings; no downloads."""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kd.dataset._metadata import pending_datasets  # noqa: E402

SUFFIXES = {
    ".mat",
    ".npy",
    ".npz",
    ".nc",
    ".csv",
    ".xlsx",
    ".xls",
    ".pkl",
    ".dat",
    ".zip",
    ".json",
    ".txt",
    ".png",
    ".pt",
    ".pth",
    ".h5",
    ".hdf5",
    ".index",
    ".meta",
    ".data-00000-of-00001",
}
SOURCE_ROOT = ROOT / "kd/dataset"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _origin(relative):
    from kd.dataset._core_sources import ASSETS

    if relative in ASSETS:
        repository, revision, upstream_path, sha256 = ASSETS[relative]
        return {"repository": repository, "revision": revision, "path": upstream_path,
                "status": "upstream_byte_verified", "expected_sha256": sha256}
    folder = relative.split("/")[0]
    if folder == "Discovery_of_soild_consititutive":
        return {
            "repository": "https://github.com/woshixuhao/Discovery_of_soild_consititutive",
            "revision": "b4af0b5b49a1c1d0392a6504ec12a2c763136957",
            "revision_evidence": "docs/validation.md",
        }
    if folder == "DeepSymbolicOptimization":
        return {
            "repository": "https://github.com/dso-org/deep-symbolic-optimization",
            "revision": None,
            "repository_evidence": "kd/model/dso/task/regression/regression.py",
        }
    return {
        "repository": None,
        "revision": None,
        "retained_source_directory": folder,
        "status": "original_revision_or_repository_unverified",
    }


def _role(relative):
    """Identify retained models, definitions and results before using file suffixes."""
    path = Path(relative)
    suffix = path.suffix.lower()
    if suffix == ".zip":
        return "archive_copy"
    if "results_save/" in relative or "/results/" in relative:
        return "upstream_result"
    if suffix == ".png":
        return "illustration"
    if path.name == "function_sets.csv":
        return "grammar_definition"
    if path.name == "benchmarks.csv":
        return "benchmark_definition"
    if (suffix in {".pt", ".pth", ".index", ".meta", ".data-00000-of-00001"}
            or "SymbolicGPT" in relative or "PDE_dataset_0725" in relative
            or "dict_datas_0725" in relative or "/control/data/" in relative
            or path.name == "word_dict.pkl"):
        return "training_asset"
    if suffix in {".json", ".txt"}:
        return "generator_configuration"
    return "observation_or_definition"


def asset_manifest(*, root=SOURCE_ROOT, expected=None):
    root = Path(root)
    assets = {}
    paths = sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUFFIXES and "__pycache__" not in p.parts
    )
    for path in paths:
        relative = path.relative_to(root).as_posix()
        role = _role(relative)
        assets[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": digest(path),
            "availability": "local_available",
            "source": _origin(relative),
            "role": role,
            "paper": None,
            "generator_script": None,
            "generator_version": None,
            "license_status": "unknown",
            "license_evidence": None,
            "redistribution_status": "not_established",
            "code_license_is_data_license": False,
            "datasets": [],
            "families": [],
            "instances": [],
            "targets": [],
        }
    import run_benchmark as benchmark

    catalog, _ = benchmark.dataset_catalog()
    cards = [{"dataset": n, **card} for n, card in catalog.items()] + pending_datasets()
    for card in cards:
        for pattern in card["source_paths"]:
            for path in root.glob(pattern):
                if not path.is_file():
                    continue
                row = assets.get(path.relative_to(root).as_posix())
                if row:
                    declared_role = card.get("benchmark_role")
                    if declared_role in {"training_asset", "duplicate", "fixed_sample_variant"}:
                        row["role"] = declared_role
                    elif row["role"] == "generator_configuration":
                        row["role"] = "observation_or_definition"
                    row.setdefault("integration", []).append({
                        "dataset": card["dataset"],
                        "status": card.get("integration_status", "registered"),
                        "requirements": card.get("requirements", []),
                    })
                    row["datasets"].append(card["dataset"])
                    row["families"].append(card["physical_family"])
                    row["instances"].append(card["dataset"] + "; see expanded dataset inventory")
                    row["targets"].append(card["target_definition"])
    groups = defaultdict(list)
    for row in assets.values():
        groups[row["sha256"]].append(row["path"])
    for row in assets.values():
        row["byte_aliases"] = [p for p in groups[row["sha256"]] if p != row["path"]]
        for key in ("datasets", "families", "instances", "targets"):
            row[key] = sorted(set(row[key]))
    if expected:
        for prior in expected["assets"]:
            relative = prior["path"]
            if relative not in assets:
                assets[relative] = {
                    **prior,
                    "availability": "local_missing",
                    "missing_reason": "optional external asset not present in checkout",
                }
            else:
                assets[relative]["expected_sha256_matches"] = (
                    prior["sha256"] == assets[relative]["sha256"]
                )
    related = []
    for path in sorted((ROOT/"kd/model/eqgpt/data").glob("*")):
        if path.is_file():
            related.append({"path":path.relative_to(ROOT).as_posix(),"bytes":path.stat().st_size,
                "sha256":digest(path),"role":"training_asset","availability":"local_available",
                "license_status":"unknown","revision":None,
                "evidence":"kd/model/eqgpt/vocab.py uses these resources"})
    return {
        "related_training_assets":related,
        "schema": "assets-v1",
        "root": "kd/dataset",
        "raw_data_included": False,
        "absolute_paths_included": False,
        "license_policy": "unknown is distinct from missing and unknown source; metadata does not authorize redistribution",
        "counts": dict(Counter(row["availability"] for row in assets.values())),
        "assets": list(assets.values()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-manifest", type=Path)
    parser.add_argument("--root", type=Path, default=SOURCE_ROOT)
    args = parser.parse_args(argv)
    expected = (
        json.loads(args.expected_manifest.read_text(encoding="utf8"))
        if args.expected_manifest else None
    )
    manifest = asset_manifest(root=args.root, expected=expected)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "asset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf8",
    )
    print(json.dumps({"assets": len(manifest["assets"]), "availability": manifest["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
