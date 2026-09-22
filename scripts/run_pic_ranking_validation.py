"""Bounded multi-candidate/multi-seed Burgers and KdV PIC validation.

This is an opt-in calibration experiment, not part of ordinary benchmark runs.
It checks the operational claim that the declared true structure receives the
smallest finite PIC among two nearby structural perturbations.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kd.metrics import (
    PIC_IMPLEMENTATION_VERSION, TorchPICConfig, evaluate_torch_pic,
    prepare_torch_pic_reference,
)


def _simulate_periodic(kind, nx, nt, *, internal_nx=96, final_time=0.6, time_step=0.001):
    """Generate an independently integrated, non-travelling-wave calibration field."""
    if internal_nx % nx:
        raise ValueError("internal_nx must be divisible by data_nx")
    x = np.linspace(-np.pi, np.pi, internal_nx, endpoint=False)
    times = np.linspace(0.0, final_time, nt)
    spacing = 2 * np.pi / internal_nx
    wave_numbers = 2 * np.pi * np.fft.fftfreq(internal_nx, d=spacing)
    modes = np.fft.fftfreq(internal_nx) * internal_nx
    dealias = np.abs(modes) <= internal_nx / 3
    if kind == "burgers":
        state = np.sin(x) + 0.35 * np.sin(2 * x) - 0.15 * np.cos(3 * x)
    elif kind == "kdv":
        # Retain resolved higher modes so the small 0.0025*u_xxx contribution
        # is structurally identifiable instead of being negligible beside u*u_x.
        state = (
            0.35 * np.cos(x) + 0.2 * np.sin(3 * x)
            + 0.12 * np.cos(6 * x) + 0.08 * np.sin(8 * x)
        )
    else:
        raise ValueError(f"unknown calibration PDE: {kind}")

    def rhs(values):
        transformed = np.fft.fft(values)
        ux = np.fft.ifft(1j * wave_numbers * transformed).real
        nonlinear_hat = np.fft.fft(values * ux)
        nonlinear_hat[~dealias] = 0
        result = -np.fft.ifft(nonlinear_hat).real
        if kind == "burgers":
            result += 0.1 * np.fft.ifft(-(wave_numbers ** 2) * transformed).real
        else:
            result -= 0.0025 * np.fft.ifft((1j * wave_numbers) ** 3 * transformed).real
        return result

    snapshots = [state.copy()]
    current_time = 0.0
    for target_time in times[1:]:
        while current_time < target_time - 1e-15:
            step = min(time_step, target_time - current_time)
            k1 = rhs(state)
            k2 = rhs(state + 0.5 * step * k1)
            k3 = rhs(state + 0.5 * step * k2)
            k4 = rhs(state + step * k3)
            state = state + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            current_time += step
        snapshots.append(state.copy())
    stride = internal_nx // nx
    output_x = x[::stride]
    field = np.asarray(snapshots)[:, ::stride].T
    coordinates = np.stack(
        np.meshgrid(output_x, times, indexing="ij"), axis=-1
    ).reshape(-1, 2)
    return coordinates, field.reshape(-1)


def _problems(nx, nt, simulator=None):
    simulator = simulator or {}
    burgers_coordinates, burgers_values = _simulate_periodic(
        "burgers", nx, nt, **simulator
    )
    kdv_coordinates, kdv_values = _simulate_periodic("kdv", nx, nt, **simulator)
    viscosity, dispersion = 0.1, 0.0025

    return {
        "burgers": {
            "coordinates": burgers_coordinates,
            "values": burgers_values,
            "candidates": [
                {"name": "true", "terms": ["u_xx", "u*u_x"],
                 "coefficients": [viscosity, -1.0]},
                {"name": "missing_diffusion", "terms": ["u*u_x"],
                 "coefficients": [-1.0]},
                {"name": "extra_linear", "terms": ["u", "u_xx", "u*u_x"],
                 "coefficients": [0.0, viscosity, -1.0]},
            ],
        },
        "kdv": {
            "coordinates": kdv_coordinates,
            "values": kdv_values,
            "candidates": [
                {"name": "true", "terms": ["u_xxx", "u*u_x"],
                 "coefficients": [-dispersion, -1.0]},
                {"name": "missing_dispersion", "terms": ["u*u_x"],
                 "coefficients": [-1.0]},
                {"name": "extra_diffusion", "terms": ["u_xx", "u_xxx", "u*u_x"],
                 "coefficients": [0.0, -dispersion, -1.0]},
            ],
        },
    }


def _serial(value):
    return None if value is None else np.asarray(value).tolist()


def run(configuration, *, output=None):
    seeds = configuration.get("seeds", [17, 29, 43])
    backend = dict(configuration.get("backend", {}))
    nx_data = int(configuration.get("data_nx", 24))
    nt_data = int(configuration.get("data_nt", 24))
    cache_dir = configuration.get("cache_dir")
    if cache_dir is not None:
        cache_dir = Path(cache_dir).expanduser().resolve()
    rows = []
    started = time.perf_counter()
    for problem_name, problem in _problems(
        nx_data, nt_data, configuration.get("simulator")
    ).items():
        for seed in seeds:
            candidate_config = dict(backend)
            candidate_config["seed"] = int(seed)
            config = TorchPICConfig(**candidate_config)
            prepared = prepare_torch_pic_reference(
                problem["coordinates"], problem["values"], config=config,
                cache_dir=cache_dir,
            )
            for candidate in problem["candidates"]:
                evaluated = evaluate_torch_pic(
                    prepared, candidate["terms"],
                    candidate_id=f"{problem_name}:{candidate['name']}:seed-{seed}",
                    original_coefficients=candidate["coefficients"],
                )
                rows.append({
                    "problem": problem_name,
                    "seed": int(seed),
                    "candidate": candidate["name"],
                    "terms": candidate["terms"],
                    "status": evaluated.status,
                    "pic": evaluated.pic,
                    "r_loss": evaluated.r_loss,
                    "p_loss": evaluated.p_loss,
                    "message": evaluated.message,
                    "reference_train_normalized_rmse":
                        prepared.reference_train_normalized_rmse,
                    "reference_fit_coefficients": _serial(
                        evaluated.reference_fit_coefficients
                    ),
                    "refitted_coefficients": _serial(evaluated.refitted_coefficients),
                    "cost": dict(evaluated.cost),
                })
    trials = []
    for problem_name in sorted({row["problem"] for row in rows}):
        for seed in seeds:
            group = [
                row for row in rows
                if row["problem"] == problem_name and row["seed"] == seed
            ]
            finite = [row for row in group if row["status"] == "ok" and row["pic"] is not None]
            winner = min(finite, key=lambda row: row["pic"])["candidate"] if finite else None
            trials.append({
                "problem": problem_name,
                "seed": int(seed),
                "winner": winner,
                "true_ranked_first": winner == "true",
                "finite_candidates": len(finite),
                "candidate_count": len(group),
            })
    payload = {
        "protocol": "pic-ranking-validation-v1",
        "implementation_version": PIC_IMPLEMENTATION_VERSION,
        "configuration": {**configuration, "backend": asdict(TorchPICConfig(**{
            **backend, "seed": int(seeds[0])
        }))},
        "rows": rows,
        "trials": trials,
        "summary": {
            "n_trials": len(trials),
            "n_true_ranked_first": sum(item["true_ranked_first"] for item in trials),
            "true_top1_rate": (
                sum(item["true_ranked_first"] for item in trials) / len(trials)
                if trials else None
            ),
            "all_candidates_finite_rate": (
                sum(item["finite_candidates"] == item["candidate_count"] for item in trials)
                / len(trials) if trials else None
            ),
            "wall_seconds": time.perf_counter() - started,
            "claim_scope": (
                "bounded independently integrated ranking calibration; not a reproduction of paper-scale noise results"
            ),
        },
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    print(rendered)
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    configuration = json.loads(args.config.read_text(encoding="utf-8"))
    payload = run(configuration, output=args.output)
    required = float(configuration.get("required_true_top1_rate", 1.0))
    if payload["summary"]["true_top1_rate"] < required:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
