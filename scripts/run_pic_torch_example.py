"""Small reproducible, opt-in CPU smoke run for PDE PIC v2."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kd.metrics import TorchPICConfig, evaluate_torch_pic, prepare_torch_pic_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="optional JSON diagnostic artifact")
    args = parser.parse_args()
    config = TorchPICConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    x = np.linspace(0.0, np.pi, 12)
    t = np.linspace(0.0, 0.8, 12)
    coordinates = np.stack(np.meshgrid(x, t, indexing="ij"), axis=-1).reshape(-1, 2)
    observations = np.exp(-coordinates[:, 1]) * np.sin(coordinates[:, 0])
    prepared = prepare_torch_pic_reference(coordinates, observations, config=config)
    result = evaluate_torch_pic(prepared, ["u_xx"], candidate_id="synthetic_heat",
                                original_coefficients=[1.0])
    payload = {
        "status": result.status, "pic": result.pic, "r_loss": result.r_loss,
        "p_loss": result.p_loss, "message": result.message,
        "cache_key": result.cache_key, "reference_train_rmse": prepared.reference_train_rmse,
        "reference_train_normalized_rmse": prepared.reference_train_normalized_rmse,
        "cost": dict(result.cost),
        "implementation_version": result.version,
        "validation_scope": "bounded real-training execution smoke; not paper reproduction",
        "original_coefficients": None if result.original_coefficients is None else result.original_coefficients.tolist(),
        "reference_fit_coefficients": None if result.reference_fit_coefficients is None else result.reference_fit_coefficients.tolist(),
        "refitted_coefficients": None if result.refitted_coefficients is None else result.refitted_coefficients.tolist(),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if result.status != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
