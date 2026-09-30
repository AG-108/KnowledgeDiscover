"""Check the local LLM-SR backbone files and, optionally, the serving endpoint.

Run after downloading meta-llama/Llama-3.1-8B-Instruct; this script never
downloads anything:

    python scripts/check_llmsr_backbone.py --model-path /path/to/Llama-3.1-8B-Instruct
    python scripts/check_llmsr_backbone.py --model-path ... --server --calls 4

``--server`` queries ``/v1/models`` and runs a short LLM-SR search with the
protocol settings from ``configs/benchmark/baselines/llmsr.json`` (generation,
program parsing and BFGS constant fitting) on a synthetic problem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kd.model.kd_llmsr import DEFAULT_BACKBONE, KD_LLMSR  # noqa: E402

# Architecture of meta-llama/Llama-3.1-8B-Instruct (config.json on the Hub).
EXPECTED_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "vocab_size": 128256,
}


def check_files(model_path, compute_sha256=False):
    path = Path(model_path)
    problems = []
    config_path = path / "config.json"
    if not config_path.is_file():
        return {"ok": False, "problems": [f"missing {config_path}"]}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key, expected in EXPECTED_CONFIG.items():
        if config.get(key) != expected:
            problems.append(f"config.json {key}={config.get(key)!r}, expected {expected!r}")
    if (config.get("rope_scaling") or {}).get("rope_type") != "llama3":
        problems.append("config.json rope_scaling.rope_type is not 'llama3' (Llama 3.1)")
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        if not (path / name).is_file():
            problems.append(f"missing {name}")
    tokenizer_config = path / "tokenizer_config.json"
    if tokenizer_config.is_file():
        if not json.loads(tokenizer_config.read_text(encoding="utf-8")).get("chat_template"):
            problems.append("tokenizer_config.json has no chat_template")
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        shards = sorted(set(json.loads(index_path.read_text(encoding="utf-8"))["weight_map"].values()))
    else:
        shards = sorted(item.name for item in path.glob("*.safetensors"))
        if not shards:
            problems.append("no safetensors weights found")
    missing = [name for name in shards if not (path / name).is_file()]
    problems.extend(f"missing weight shard {name}" for name in missing)
    present = [path / name for name in shards if name not in missing]
    report = {
        "ok": not problems,
        "problems": problems,
        "model_path": str(path.resolve()),
        "weight_shards": len(shards),
        "weight_bytes": sum(item.stat().st_size for item in present),
    }
    if compute_sha256:
        hashes = {}
        for item in present:
            digest = hashlib.sha256()
            with item.open("rb") as stream:
                for block in iter(lambda: stream.read(1 << 24), b""):
                    digest.update(block)
            hashes[item.name] = digest.hexdigest()
        report["sha256"] = hashes
    return report


def check_server(endpoint, model_name, calls, api_key_env):
    base = endpoint.split("/chat/completions")[0].rstrip("/")
    headers = {}
    key = os.environ.get(api_key_env) or os.environ.get("VLLM_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    response = requests.get(f"{base}/models", headers=headers, timeout=30)
    response.raise_for_status()
    served = response.json().get("data", [])
    report = {
        "endpoint": endpoint,
        "served_models": [{"id": item.get("id"), "root": item.get("root")} for item in served],
    }
    try:  # vLLM exposes its version next to /v1; other servers may not.
        version = requests.get(f"{base.rsplit('/v1', 1)[0]}/version", headers=headers, timeout=10)
        if version.ok:
            report["server_version"] = version.json().get("version")
    except (requests.RequestException, ValueError):
        pass
    if model_name not in {item.get("id") for item in served}:
        report.update(ok=False, problem=f"{model_name!r} is not served at {base}/models")
        return report
    settings = json.loads(
        (ROOT / "configs/benchmark/baselines/llmsr.json").read_text(encoding="utf-8")
    )["model"]
    settings.update(endpoint=endpoint, api_model=model_name, max_llm_calls=calls, random_state=0)
    X = np.linspace(-2.0, 2.0, 64).reshape(-1, 2)
    y = 1.5 * np.sin(X[:, 0]) + 0.3 * X[:, 1] ** 2
    model = KD_LLMSR(**settings).fit(X, y, variable_names=["x", "v"], target_name="a")
    report.update(
        ok=model.search_stats_["llm_calls"] == calls,
        best_expression=model.best_expression_,
        search_stats=model.search_stats_,
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-path", default=os.environ.get("LLMSR_BACKBONE_PATH"))
    parser.add_argument("--sha256", action="store_true", help="Hash every weight shard")
    parser.add_argument("--server", action="store_true", help="Also check the endpoint")
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("LLMSR_ENDPOINT", "http://127.0.0.1:10005/v1/chat/completions"),
    )
    parser.add_argument("--model-name", default=DEFAULT_BACKBONE)
    parser.add_argument("--calls", type=int, default=4, help="LLM calls for the server check")
    parser.add_argument("--api-key-env", default="LLMSR_API_KEY")
    args = parser.parse_args(argv)

    report = {}
    if args.model_path:
        report["files"] = check_files(args.model_path, args.sha256)
    elif not args.server:
        parser.error("pass --model-path (or set LLMSR_BACKBONE_PATH) and/or --server")
    if args.server:
        try:
            report["server"] = check_server(
                args.endpoint, args.model_name, args.calls, args.api_key_env
            )
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            report["server"] = {"ok": False, "problem": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(report, indent=2, default=str))
    return 0 if all(part.get("ok") for part in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
