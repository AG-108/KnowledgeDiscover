"""Small synthetic integration checks, not full baseline performance measurements."""

import numpy as np
import run_benchmark as benchmark
from kd.dataset import GridPDEDataset, load_dataset


def test_integral_weak_method_uses_temporal_holdout_and_validated_common_solver():
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
    case = benchmark.build_experiments(catalog, models=["integral_weak_pde"], datasets=["burgers"])[0]
    case["dataset"] = "synthetic_advection"
    case["model_params"].update(terms=[(1, 1)], threshold=0)
    result = benchmark.run_pde(dataset, case)
    assert result["coefficient_recovery"] == 1
    assert result["equation_rollout_status"] == "ok"
    assert result["equation_rollout_nrmse"] < 1e-3
    assert result["train_time_range"][1] < result["test_time_range"][0]


def test_generated_ode_provenance_and_disjoint_trajectories_survive_runner_bridge():
    catalog, _ = benchmark.dataset_catalog()
    case = benchmark.build_experiments(catalog, models=["sindy"], datasets=["ode_core_oscillator"])[0]
    dataset = load_dataset(case["dataset"], n_points=31)
    problems = list(benchmark.regression_problems(dataset, case))
    assert len(problems) == 2
    details = problems[0]["metadata"]
    assert set(details["train_trajectories"]).isdisjoint(details["test_trajectories"])
    assert details["generation_metadata"]["parameter_protocol"] == "unknown_fixed_constants"


def test_core_configs_have_executable_noise_and_sparsity_parameters():
    config, _ = benchmark.load_benchmark_config("configs/benchmark/core_v2_ode_noisy_sparse.json")
    catalog, _ = benchmark.dataset_catalog()
    benchmark.validate_benchmark_config(config, catalog)
    overrides = config["overrides"]["datasets"]["ode_core_rational"]
    dataset = load_dataset("ode_core_rational", **overrides)
    assert dataset.generation_metadata["observation_noise"] == 0.01
    assert len(dataset.trajectories[0]["t"]) == 121
