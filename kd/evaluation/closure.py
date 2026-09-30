"""Frozen balanced point sampling; independent run/layer/frame scoring."""

import hashlib

import numpy as np

from kd.dataset._closure import TARGET, resolved_features


def problems(dataset, case):
    train_run, test_run = dataset.runs
    train_fields, names = resolved_features(train_run)
    test_fields, _ = resolved_features(test_run)
    nt = len(train_run["time"])
    end = max(2, int(nt * 0.75))
    train_frames = np.arange(end)
    val_frames = np.arange(end + 2, nt)
    limit = case["run_params"].get("max_train_samples")
    n_points = min(64, max(2, int(limit) // len(train_frames))) if limit else 64
    ny, nx = train_fields.shape[1:3]
    rng = np.random.default_rng(case["seed"])
    # Same space samples per frame; balanced frame weights, fixed before labels.
    train_points = rng.choice(ny * nx, n_points, replace=False)
    test_points = np.random.default_rng(case["seed"] + 7919).choice(
        ny * nx, min(256, ny * nx), replace=False
    )
    for layer in range(2):
        X_train = (
            train_fields[train_frames]
            .reshape(len(train_frames), ny * nx, -1)[:, train_points]
            .reshape(-1, len(names))
        )
        X_test = test_fields.reshape(len(test_run["time"]), ny * nx, -1)[:, test_points].reshape(
            -1, len(names)
        )
        y_train = (
            train_run["fields"][TARGET][train_frames, layer]
            .reshape(len(train_frames), ny * nx)[:, train_points]
            .ravel()
        )
        y_test = (
            test_run["fields"][TARGET][:, layer]
            .reshape(len(test_run["time"]), ny * nx)[:, test_points]
            .ravel()
        )
        xm, xs = X_train.mean(0), X_train.std(0)
        xs = np.where(xs > 0, xs, 1.0)
        ym, ys = float(y_train.mean()), float(y_train.std())
        if not ys > 0:
            raise ValueError("Constant pyqg training closure target")
        sample = {
            "seed": case["seed"],
            "layer": layer + 1,
            "train_frames": train_frames.tolist(),
            "validation_frames": val_frames.tolist(),
            "support_gap_frames": 2,
            "train_grid_indices_yx": [[int(i // nx), int(i % nx)] for i in train_points],
            "test_frames": list(range(len(test_run["time"]))),
            "test_grid_indices_yx": [[int(i // nx), int(i % nx)] for i in test_points],
            "train_domain_sample_fraction": float(len(train_points)/(ny*nx)),
            "test_domain_sample_fraction": float(len(test_points)/(ny*nx)),
            "valid_domain_fraction": 1.,
            "train_sample_weight": float(1 / len(y_train)),
            "test_sample_weight": float(1 / len(y_test)),
            "independent_units": "two simulation runs; correlated frame/grid observations",
        }
        sample["sha256"] = hashlib.sha256(str(sample).encode()).hexdigest()
        yield {
            "target": f"{TARGET}_L{layer + 1}",
            "X_train": (X_train - xm) / xs,
            "y_train": (y_train - ym) / ys,
            "X_test": (X_test - xm) / xs,
            "y_test": (y_test - ym) / ys,
            "variable_names": names,
            "split": "run0_train_run1_test",
            "ground_truth": None,
            "metadata": {
                **dataset.protocol_metadata,
                "sampling": sample,
                "layer": layer + 1,
                "benchmark_function_set": ["add", "sub", "mul", "div", "sin", "cos", "const"],
                "normalization": {
                    "x_mean": xm.tolist(),
                    "x_scale": xs.tolist(),
                    "y_mean": ym,
                    "y_scale": ys,
                    "fit_role": "run0_training_frames_only",
                },
                "validation_use": "reserved; initial adapters select/stop on training only",
                "derivative_library": "resolved-only per-frame periodic FFT first x/y derivatives, both layers",
                "normalization_expression": "reported expression uses standardized resolved inputs/output",
            },
        }


def finish(result, problem, model):
    from run_benchmark import regression_metrics

    meta = problem["metadata"]
    scale = meta["normalization"]["y_scale"]
    result.update(
        metric_target="offline_PV_closure",
        recovery_eligible=False,
        recovery_evaluator_status="not_applicable",
        closure_protocol_version="closure-v1",
    )
    for key in ("test_mse",):
        if result.get(key) is not None:
            result[key] *= scale**2
    if result.get("test_rmse") is not None:
        result["test_rmse"] *= scale
    if model is None or not hasattr(model, "predict"):
        result["closure_frame_status"] = "abstained_no_prediction_api"
        return result
    predicted = np.asarray(model.predict(problem["X_test"])).reshape(-1) * scale
    truth = problem["y_test"] * scale
    nframes = len(meta["sampling"]["test_frames"])
    truth, predicted = truth.reshape(nframes, -1), predicted.reshape(nframes, -1)
    result["closure_frame_metrics"] = [
        {"run": 1, "layer": meta["layer"], "frame": i, **regression_metrics(a, b)}
        for i, (a, b) in enumerate(zip(truth, predicted))
    ]
    result["closure_layer"] = meta["layer"]
    result["closure_online_status"] = "not_evaluated"
    return result
