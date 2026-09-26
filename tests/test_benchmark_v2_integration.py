"""Small synthetic integration checks, not full baseline performance measurements."""

import numpy as np
import run_benchmark as benchmark
from kd.dataset import GridPDEDataset, load_dataset


def test_wsindy_uses_temporal_holdout_and_validated_common_solver():
    x = np.linspace(0, 2 * np.pi, 81, endpoint=False)
    t = np.linspace(0, 1, 61)
    dataset = GridPDEDataset("synthetic_advection", x=x, t=t,
                            usol=np.sin(x[:, None] - 0.7 * t[None, :]),
                            pde_data=None, legacy=True, domain={"x": (0, 2 * np.pi), "t": (0, 1)}, epi=0.0)
    dataset.sym_true = "-0.7*u_x"
    dataset.rollout_metadata = {
        "boundary_conditions": {"type": "periodic"},
        "endpoint_convention": "periodic_endpoint_excluded",
        "spatial_discretization": "spectral_fft",
        "solver": {"method": "RK45", "rtol": 1e-8, "atol": 1e-10, "max_step": 0.01},
        "reference_nrmse_tolerance": 1e-5,
    }
    catalog, _ = benchmark.dataset_catalog()
    case = benchmark.build_experiments(catalog, models=["weakform"], datasets=["burgers"])[0]
    case["dataset"] = "synthetic_advection"
    case["model_params"].update(
        terms=[(1, 1)], m_x=12, m_t=8, test_function_power=6, threshold=0
    )
    result = benchmark.run_pde(dataset, case)
    assert result["coefficient_recovery"] == 1
    assert result["equation_rollout_status"] == "ok"
    assert result["equation_rollout_nrmse"] < 1e-3
    assert result["train_time_range"][1] < result["test_time_range"][0]
    assert result["method_provenance"]["method"] == "WSINDy-PDE"
    assert result["wsindy_n_weak_samples"] > 0


def test_generated_ode_provenance_and_disjoint_trajectories_survive_runner_bridge():
    catalog, _ = benchmark.dataset_catalog()
    case = benchmark.build_experiments(catalog, models=["sindy"], datasets=["ode_core_oscillator"])[0]
    dataset = load_dataset(case["dataset"], n_points=31)
    problems = list(benchmark.regression_problems(dataset, case))
    assert len(problems) == 2
    details = problems[0]["metadata"]
    assert set(details["train_trajectories"]).isdisjoint(details["test_trajectories"])
    assert details["generation_metadata"]["parameter_protocol"] == "unknown_fixed_constants"


def test_expanded_ode_systems_survive_runner_bridge():
    from kd.dataset._ode_core import SYSTEMS

    catalog, _ = benchmark.dataset_catalog()
    names = [f"ode_core_{name}" for name in SYSTEMS]
    cases = benchmark.build_experiments(catalog, models=["sindy"], datasets=names)
    assert len(cases) == len(SYSTEMS)
    for case in cases:
        dataset = load_dataset(case["dataset"], n_trajectories=3, n_points=31)
        problems = list(benchmark.regression_problems(dataset, case))
        assert len(problems) == len(dataset.state_vars)
        for problem in problems:
            assert problem["split"] == "trajectory_holdout"
            assert problem["ground_truth"]
            assert len(problem["X_train"]) > 0 and len(problem["X_test"]) > 0
            assert set(problem["metadata"]["train_trajectories"]).isdisjoint(
                problem["metadata"]["test_trajectories"]
            )


def test_core_configs_have_executable_noise_and_sparsity_parameters():
    from kd.dataset._ode_core import CORE_ODE_DATASETS

    clean_config, _ = benchmark.load_benchmark_config("configs/benchmark/core_v2.json")
    assert tuple(clean_config["datasets"][:8]) == CORE_ODE_DATASETS
    config, _ = benchmark.load_benchmark_config("configs/benchmark/core_v2_ode_noisy_sparse.json")
    catalog, _ = benchmark.dataset_catalog()
    benchmark.validate_benchmark_config(config, catalog)
    cases = benchmark.build_experiments(
        catalog, tasks=["ode"], models=["sindy"], datasets=config["datasets"],
        config=config["overrides"],
    )
    degrees = {case["dataset"]: case["model_params"]["polynomial_degree"] for case in cases}
    assert tuple(config["datasets"]) == CORE_ODE_DATASETS
    assert len(cases) == 8
    assert degrees["ode_core_damped_oscillator"] == 1
    assert all(degrees[f"ode_core_{name}"] == 3 for name in (
        "duffing", "van_der_pol", "fitzhugh_nagumo"))
    overrides = config["overrides"]["datasets"]["ode_core_rational"]
    dataset = load_dataset("ode_core_rational", **overrides)
    assert dataset.generation_metadata["observation_noise"] == 0.01
    assert len(dataset.trajectories[0]["t"]) == 121
    robertson = load_dataset("ode_core_robertson", n_trajectories=2, n_points=31,
                             **config["overrides"]["datasets"]["ode_core_robertson"])
    assert all(np.isfinite(tr["state"]).all() for tr in robertson.trajectories)


def test_formal_core_ode_config_selects_all_ode_methods_and_only_eight_systems():
    from kd.dataset._ode_core import CORE_ODE_DATASETS

    config, _ = benchmark.load_benchmark_config("configs/benchmark/core_ode_track.json")
    catalog, _ = benchmark.dataset_catalog()
    benchmark.validate_benchmark_config(config, catalog)
    assert tuple(config["datasets"]) == CORE_ODE_DATASETS
    ode_models = {name for name, info in benchmark.BASELINES.items() if "ode" in info["tasks"]}
    assert set(config["models"]) == ode_models
    cases = benchmark.build_experiments(
        catalog, tasks=config["tasks"], models=config["models"],
        datasets=config["datasets"], seeds=config["seeds"],
        profile=config["profile"], config=config["overrides"],
    )
    assert len(cases) == 8 * len(ode_models) * len(config["seeds"])
    assert {case["dataset"] for case in cases} == set(CORE_ODE_DATASETS)
    assert all(case["task"] == "ode" for case in cases)
