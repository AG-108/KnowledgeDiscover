"""Server-side PDE calibration; fit/validation/test blocks are kept separate.

Run in a copied source tree with --root pointing at a new experiment directory.
No truth or final holdout metric participates in configuration selection.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import functools
import hashlib
import itertools
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "2")
SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE))
DATASETS = ["burgers", "burgers_sine", "kdv", "chafee-infante", "allen_cahn",
            "convection_diffusion", "fisher", "fisher_linear", "PDE_divide", "PDE_compound"]
MODELS = ["pdefind", "pysindy", "weakform", "dlga", "deepmod"]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def candidates():
    result = {m: [] for m in MODELS}
    for library, threshold, ridge, stencil in itertools.product(
        ["compact", "extended"], [.001, .005, .02, .05], [1e-8, 1e-4], [7, 9]
    ):
        result["pdefind"].append(dict(solver="stridge", library_kind=library, coordinate_terms=True,
            threshold=threshold, ridge=ridge, stencil=stencil, derivative_order=3))
    for library, weight, nu in itertools.product(["compact", "extended"], [1e-6, 5e-5, 5e-4, .005], [.1, 1.]):
        result["pysindy"].append(dict(library_kind=library, coordinate_terms=True, stencil=7,
            reg_weight_lam=weight, relax_coeff_nu=nu, max_iter=500, tolerance=1e-8,
            regularizer="L0", debias=True))
    compact = [[0, 0], [1, 0], [2, 0], [3, 0], [1, 1], [1, 2], [1, 3], [2, 1], [2, 2]]
    for terms, support, tradeoff in itertools.product([compact, None], [.08, .12, .18, .25], [.25, .5, .75]):
        result["weakform"].append(dict(terms=terms, polynomial_degree=3, max_derivative=3,
            coordinate_terms=True, support_fraction=support, sparsity_tradeoff=tradeoff,
            max_weak_samples=5000, n_thresholds=50))
    for depth, width, library, penalty, lm in [(2, 32, "compact", 1e-3, 0), (3, 32, "extended", 1e-3, 0),
                                             (2, 64, "compact", 1e-4, 0), (1, 32, "extended", 1e-3, 5000)]:
        result["dlga"].append(dict(operators=["u", "u_x", "u_xx", "u_xxx"], epi=penalty,
            input_dim=2, max_iter=500, lbfgs_max_iter=5000, hidden_layers=depth, hidden_width=width,
            least_squares_nfev=lm, bias_init_scale=1.,
            library_kind=library, coordinate_terms=True, normalize_coordinates=True,
            fitness_kind="normalized_lstsq", precision="float64", learning_rate=.003,
            target_derivative="u_t", verbose=False, device="cuda:0"))
    for hidden, activation, threshold, weight, library in [
        ([32, 32, 32], "tanh", .01, .1, "compact"),
        ([64, 64, 64], "tanh", .03, .1, "extended"),
        ([32, 32, 32], "sin", .005, .01, "extended"),
        ([64, 64], "sin", .01, .1, "compact"),
    ]:
        result["deepmod"].append(dict(training_variant="scheduled", hidden_layers=hidden,
            library_kind=library, coordinate_terms=True, activation=activation, threshold=threshold,
            pde_weight=weight, precision="float64", max_steps=5000, warmup_steps=400, lbfgs_steps=1000,
            sparsity_interval=200, batch_size=1024, learning_rate=.002, device="cuda:0"))
    return result


@functools.lru_cache(maxsize=12)
def load_views(name):
    import run_benchmark as b
    from kd.benchmark_preparation import slice_pde_time, temporal_block_holdout
    _, dataset = next(b._load_instances(dict(dataset=name, dataset_params={}, source="catalog", seed=0)))
    dataset = b.prepare_pde(dataset, "weakform")
    development, test, split = temporal_block_holdout(dataset, .2)
    inner_split = round(len(development.t) * .75)
    train = slice_pde_time(development, 0, inner_split)
    validation = slice_pde_time(development, inner_split, len(development.t))
    return dataset, development, train, validation, test, split, inner_split


@functools.lru_cache(maxsize=24)
def score_namespace(dataset_name, final):
    import numpy as np
    from kd.dataset._derivatives import axis_derivative
    _, _, _, validation, test, _, _ = load_views(dataset_name)
    dataset = test if final else validation
    u, x, t = dataset.usol, dataset.x, dataset.t
    values = {"u": u, "u1": u, "x": np.broadcast_to(x[:, None], u.shape)}
    values.update({"u_" + "x" * n: axis_derivative(u, x, n, 0, 7) for n in (1, 2, 3)})
    target = axis_derivative(u, t, 1, 1, 7)
    good = np.isfinite(target)
    for value in values.values():
        good &= np.isfinite(value)
    return {k: v[good] for k, v in values.items()}, target[good]


def validation_score(expression, dataset_name, final=False):
    import numpy as np
    import sympy as sp
    import run_benchmark as b
    values, target = score_namespace(dataset_name, final)
    parsed = b._pde_rhs_sympy(expression, tuple(values))
    if parsed is None:
        raise ValueError("Unparseable exported PDE")
    symbols = sorted(parsed.free_symbols, key=str)
    prediction = np.asarray(sp.lambdify(symbols, parsed, "numpy")(*(values[str(k)] for k in symbols)))
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite validation prediction")
    error = float(np.sqrt(np.mean((target - prediction)**2)) / max(float(np.std(target)), 1e-12))
    terms = len(sp.Add.make_args(sp.expand(parsed))) if parsed != 0 else 0
    return dict(nrmse=error, terms=terms, selection_loss=math.log10(max(error, 1e-10)) + .1 * terms,
                points=len(target), metric="interior_fd7_std_nrmse")


def trial(job):
    import numpy as np
    import torch
    from kd.benchmark_deadline import deadline_scope
    import run_benchmark as b
    root = Path(job["root"])
    identifier = digest({k: v for k, v in job.items() if k != "root"})
    directory = root / "trials" / identifier
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    write_json(directory / "job.json", job)
    result = dict(job, trial_id=identifier, config_id=digest(job["params"]))
    torch.set_num_threads(2)
    random.seed(job["seed"])
    np.random.seed(job["seed"])
    torch.manual_seed(job["seed"])
    os.chdir(directory)
    start = time.monotonic()
    with (directory / "run.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            dataset, development, train, validation, test, split, inner_split = load_views(job["dataset"])
            fit_data = development if job["stage"] == "final" else train
            params = dict(job["params"])
            model_name = job["model"]
            if model_name == "pdefind":
                from kd.model.kd_pdefind import PDEFindModel
                model = PDEFindModel(**params)
                fitting = lambda: model.fit(fit_data)
                exporting = lambda: "u_t = " + b._linear_expression(model._feature_names, model.coefficients())
            elif model_name == "pysindy":
                from kd.model.kd_sindy import PySINDyModel
                model = PySINDyModel(**params)
                fitting = lambda: model.fit_dataset(fit_data)
                exporting = lambda: "u_t = " + model.best_expression_
            elif model_name == "weakform":
                from kd.model.kd_wsindy import WSINDyPDEModel
                model = WSINDyPDEModel(**params)
                fitting = lambda: model.fit(fit_data.usol, fit_data.x, fit_data.t)
                exporting = lambda: "u_t = " + model.best_expression_
            else:
                X, y = b.pde_coordinates(fit_data)
                if model_name == "dlga":
                    from kd.model.kd_dlga import KD_DLGA
                    chosen = np.random.default_rng(job["seed"]).choice(len(X), min(len(X), 6000), replace=False)
                    X, y = X[chosen], y[chosen]
                    model = KD_DLGA(**params)
                    model.pop_size, model.n_generations = 200, 60
                else:
                    from kd.model.kd_deepmod import KD_DeepMoD
                    X = X[:, [1, 0]]
                    model = KD_DeepMoD(**params, seed=job["seed"])
                fitting = lambda: model.fit(X.copy(), y.copy())
                exporting = lambda: model.best_equation_
            fit_start = time.monotonic()
            with deadline_scope(job["budget"]):
                fitting()
            result["fit_seconds"] = time.monotonic() - fit_start
            equation = exporting()
            result.update(status="ok", expression=equation,
                          score=validation_score(equation, job["dataset"], job["stage"] == "final"),
                          stop_reason=getattr(model, "stop_reason_", "completed"),
                          split=dict(total_frames=len(dataset.t), outer_test_start=split, inner_validation_start=inner_split))
            for key in ("completed_steps_", "sparsity_updates_", "observation_validation_nmse_"):
                if hasattr(model, key): result[key.rstrip("_")] = getattr(model, key)
            if job["stage"] == "final":
                result["legacy_test_score"] = b.pde_equation_residual(equation, test)
                truth = b.PDE_REFERENCE_RHS.get(job["dataset"], getattr(dataset, "sym_true", None))
                result["recovery"] = b.pde_structure_metrics(equation, truth)
                if job["dataset"] in b.PDE_REFERENCE_RHS:
                    result["reference_legacy_test_score"] = b.pde_equation_residual(truth, test)
                    result["reference_test_score"] = validation_score(truth, job["dataset"], True)
        except Exception as exc:
            traceback.print_exc()
            result.update(status="error", error=f"{type(exc).__name__}: {exc}")
    result["elapsed"] = time.monotonic() - start
    write_json(result_path, result)
    return result


def rank(rows, model, stage):
    grouped = {}
    for row in rows:
        if row["model"] == model and row["stage"] == stage:
            grouped.setdefault(row["config_id"], []).append(row)
    ranking = []
    for identity, group in grouped.items():
        scores = [r["score"]["selection_loss"] if r["status"] == "ok" else 6. for r in group]
        ranking.append(dict(config_id=identity, params=group[0]["params"],
                            mean_selection_loss=statistics.mean(scores), n=len(group),
                            ok=sum(r["status"] == "ok" for r in group)))
    return sorted(ranking, key=lambda r: r["mean_selection_loss"])


def run(root):
    root.mkdir(parents=True, exist_ok=True)
    search = candidates()
    protocol = dict(schema="pde-calibration-v2", datasets=DATASETS,
        selection="mean over datasets of log10(inner_validation_NRMSE) + 0.1 * additive_terms",
        split="first 60% fit, next 20% validation, final 20% evaluation after selection",
        evaluation_note="Previously inspected dataset holdouts; this is configuration calibration, not new independent benchmark evidence.",
        validation_derivatives="fixed seven-point interior finite differences within validation block",
        seed_policy="seed 0 selection; final CPU seed 0, GPU seeds 0 and 1",
        reference_usage="reference equations are evaluated only after configurations have been selected",
        candidates=search, started=time.time())
    write_json(root / "protocol.json", protocol)
    rows = []
    context = multiprocessing.get_context("spawn")
    cpu = cf.ProcessPoolExecutor(max_workers=3, mp_context=context)
    gpu = cf.ProcessPoolExecutor(max_workers=2, mp_context=context)
    try:
        def jobs(model, configs, datasets, stage, budget, seeds=(0,)):
            return [dict(root=str(root), model=model, params=params, dataset=dataset,
                         stage=stage, budget=budget, seed=seed)
                    for params, dataset, seed in itertools.product(configs, datasets, seeds)]

        def execute(work, label):
            pending = {(gpu if job["model"] in {"dlga", "deepmod"} else cpu).submit(trial, job): job for job in work}
            complete = 0
            while pending:
                done, _ = cf.wait(pending, timeout=20, return_when=cf.FIRST_COMPLETED)
                for future in done:
                    job = pending.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = dict(job, config_id=digest(job["params"]), status="error", error=str(exc))
                    rows.append(result)
                    complete += 1
                    with (root / "progress.jsonl").open("a") as stream:
                        stream.write(json.dumps(result) + "\n")
                write_json(root / "status.json", dict(stage=label, completed=complete, planned=len(work),
                    total_recorded=len(rows), errors=sum(r["status"] != "ok" for r in rows), updated=time.time()))
            write_json(root / "all_results.json", rows)

        first = []
        for model in MODELS:
            first.extend(jobs(model, search[model], DATASETS if model not in {"dlga", "deepmod"}
                              else ["burgers", "kdv", "allen_cahn"], "screen", 60))
        execute(first, "screen")
        rankings = {model: rank(rows, model, "screen") for model in MODELS}
        write_json(root / "screen_rankings.json", rankings)
        gpu_work = []
        for model in ("dlga", "deepmod"):
            gpu_work.extend(jobs(model, [x["params"] for x in rankings[model][:2]], DATASETS, "validate", 120))
        execute(gpu_work, "validate")
        selected = {model: (rank(rows, model, "validate") if model in {"dlga", "deepmod"}
                            else rankings[model])[0] for model in MODELS}
        write_json(root / "selected_configs.json", selected)
        final_work = []
        for model, config in selected.items():
            final_work.extend(jobs(model, [config["params"]], DATASETS, "final", 300,
                                   (0, 1) if model in {"dlga", "deepmod"} else (0,)))
        execute(final_work, "final")
        write_json(root / "status.json", dict(stage="complete", total_recorded=len(rows),
                    errors=sum(r["status"] != "ok" for r in rows), finished=time.time()))
    finally:
        cpu.shutdown()
        gpu.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--one", type=Path)
    args = parser.parse_args()
    if args.one:
        print(json.dumps(trial(json.loads(args.one.read_text()))))
    else:
        run(args.root.resolve())
