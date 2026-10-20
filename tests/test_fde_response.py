"""FDE responses retain source axes, tied times and prediction-only semantics."""

import copy
import json

import numpy as np
import pytest
from benchmark_helpers import FDE_RESPONSE_DATASETS, SR_METHODS
from scipy.io import loadmat, savemat

import run_benchmark as benchmark
from kd.dataset import FractionalResponseDataset, load_dataset
from kd.dataset._fractional import FDE_READY_DATASETS, FDE_SPECS
from kd.dataset._metadata import pending_datasets


@pytest.fixture
def source_root(tmp_path):
    directory = tmp_path / "FDE/data"
    directory.mkdir(parents=True)
    x = np.array([0.0, 0.5, 1.0])
    t = np.array([0.0, 0.1, 0.4, 1.0, 2.0])
    field = 100 * t[:, None] + x[None, :]
    for name, spec in FDE_SPECS.items():
        path = tmp_path / spec["path"]
        if spec["layout"] == "curve":
            times = np.array([0.0, 0.1, 0.1, 0.5, 1.0, 2.0])
            savemat(path, {"t": times[:, None], "ex": np.arange(6.0)[:, None]})
        elif spec["layout"] == "tx":
            x_key, t_key, u_key = spec["keys"]
            savemat(path, {x_key: x[None, :], t_key: t[None, :], u_key: field})
        else:
            np.savetxt(path, field)
    return tmp_path


def case(name="fde_pt_fade", **overrides):
    catalog, _ = benchmark.dataset_catalog()
    return benchmark.build_experiments(
        catalog,
        models=["gplearn"],
        datasets=[name],
        config=overrides,
    )[0]


def test_fde_catalog_config_and_prediction_only_identity():
    catalog, _ = benchmark.dataset_catalog()
    selected = benchmark.build_experiments(catalog, tasks=["sr"], datasets=["fde_*"])
    sr_models = {name for name, info in benchmark.BASELINES.items() if "sr" in info["tasks"]}
    assert len(selected) == len(FDE_SPECS) * len(sr_models)
    assert {c["dataset"] for c in selected} == set(FDE_SPECS)
    assert {c["model"] for c in selected} == sr_models
    default = benchmark.build_experiments(catalog, models=["gplearn"])
    assert not any(c["dataset"].startswith("fde_") for c in default)
    for c in selected:
        assert c["task"] == "sr"
        assert c["scientific_task"] == "fractional_equation_response"
        assert c["metric_protocol_version"] == "2.3"
        assert c["ground_truth"] is None
        assert c["dataset_metadata"]["source_status"] == "user_supplied_archive"
        assert c["dataset_metadata"]["source_evidence"]["fractional_orders"] is None
        row = benchmark._base_row(
            c, target="ex" if c["dataset"].endswith(("creep", "silt")) else "u"
        )
        assert row["metric_target"] == "observed_response"
        assert row["fde_response_protocol_version"] == "fde-response-v1"
        assert row["recovery_eligible"] is False
        assert row["recovery_evaluator_status"] == "not_applicable"
        assert row["benchmark_role"] == "diagnostic"


@pytest.mark.parametrize("name", ["fde_pt_fade", "fde_pt_fde", "fde_tsfade_fft"])
def test_mat_field_keeps_time_space_orientation_and_coordinates(source_root, name):
    dataset = load_dataset(name, root=source_root)
    np.testing.assert_array_equal(dataset.X[:3], [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    np.testing.assert_array_equal(dataset.X[3:6], [[0.0, 0.1], [0.5, 0.1], [1.0, 0.1]])
    np.testing.assert_array_equal(dataset.y[:6], [0.0, 0.5, 1.0, 10.0, 10.5, 11.0])
    assert dataset.variable_names == ["x", "t"]
    assert dataset.values.shape == (5, 3)
    assert dataset.protocol_metadata["derivatives_computed"] is False
    with pytest.raises(ValueError, match="cannot be overridden"):
        load_dataset(name, root=source_root, coordinates={})


def test_tied_curve_times_and_raw_rows_stay_in_the_same_temporal_partition(source_root):
    dataset = load_dataset("fde_silt", root=source_root)
    c = case("fde_silt", run={"max_train_samples": None})
    problem = next(benchmark.regression_problems(dataset, c))
    assert problem["target"] == "ex"
    assert problem["split"] == "temporal_unique_time_holdout"
    assert problem["variable_names"] == ["t"]
    np.testing.assert_array_equal(problem["X_train"].ravel(), [0.0, 0.1, 0.1, 0.5, 1.0])
    np.testing.assert_array_equal(problem["y_train"], np.arange(5.0))
    np.testing.assert_array_equal(problem["X_test"].ravel(), [2.0])
    assert problem["metadata"]["duplicate_time_rows"] == 1
    dataset.train_test_split(seed=99)
    np.testing.assert_array_equal(dataset.X_train, problem["X_train"])
    np.testing.assert_array_equal(dataset.y_test, problem["y_test"])


def test_time_split_and_training_sampling_do_not_depend_on_heldout_responses(source_root):
    dataset = load_dataset("fde_pt_fde", root=source_root)
    c = case("fde_pt_fde", run={"max_train_samples": 4})
    first = next(benchmark.regression_problems(dataset, c))
    changed = copy.deepcopy(dataset)
    changed.y[changed.groups == changed.t[-1]] = 1e20
    second = next(benchmark.regression_problems(changed, c))
    np.testing.assert_array_equal(first["X_train"], second["X_train"])
    np.testing.assert_array_equal(first["y_train"], second["y_train"])
    np.testing.assert_array_equal(first["X_test"], second["X_test"])
    assert len(first["X_train"]) == 4
    assert len(first["X_test"]) == 3
    assert first["X_train"][:, 1].max() < first["X_test"][:, 1].min()
    assert first["metadata"]["n_train_rows_before_sampling"] == 12
    assert first["metadata"]["n_train_rows_used"] == 4
    assert second["ground_truth"] is None


@pytest.mark.parametrize("name", ["fde_convection_diffusion", "fde_ground_water_flow"])
def test_dat_requires_explicit_axes_and_evidence(source_root, name):
    with pytest.raises(ValueError, match="source_blocked"):
        load_dataset(name, root=source_root)
    coordinates = {
        "x": [0.0, 0.5, 1.0],
        "t": [0.0, 0.1, 0.4, 1.0, 2.0],
        "layout": "tx",
        "evidence": "synthetic test generator",
    }
    dataset = load_dataset(name, root=source_root, coordinates=coordinates)
    assert dataset.protocol_metadata["coordinate_status"] == "caller_declared"
    np.testing.assert_array_equal(dataset.y[-3:], [200.0, 200.5, 201.0])
    with pytest.raises(ValueError, match="evidence"):
        load_dataset(name, root=source_root, coordinates={**coordinates, "evidence": ""})
    with pytest.raises(ValueError, match="coordinate shape"):
        load_dataset(name, root=source_root, coordinates={**coordinates, "layout": "xt"})
    np.savetxt(source_root / FDE_SPECS[name]["path"], dataset.values.T)
    transposed = load_dataset(name, root=source_root, coordinates={**coordinates, "layout": "xt"})
    np.testing.assert_array_equal(transposed.X, dataset.X)
    np.testing.assert_array_equal(transposed.y, dataset.y)


@pytest.mark.parametrize(
    "failure", ["nonfinite", "complex", "unordered_time", "unordered_x", "shape", "missing_key"]
)
def test_invalid_source_data_fail_without_repair(source_root, failure):
    path = source_root / FDE_SPECS["fde_pt_fade"]["path"]
    raw = {key: value for key, value in loadmat(path).items() if not key.startswith("__")}
    if failure == "nonfinite":
        raw["un"][0, 0] = np.nan
    elif failure == "complex":
        raw["un"] = raw["un"].astype(complex) + 1j
    elif failure == "unordered_time":
        raw["t_det"] = raw["t_det"][:, ::-1]
    elif failure == "unordered_x":
        raw["x_lin"] = raw["x_lin"][:, ::-1]
    elif failure == "shape":
        raw["un"] = raw["un"].T
    else:
        raw.pop("t_det")
    savemat(path, raw)
    with pytest.raises(ValueError):
        load_dataset("fde_pt_fade", root=source_root)


def test_missing_and_source_blocked_targets_keep_their_names_and_denominators(
    source_root, tmp_path
):
    blocked = case(
        "fde_ground_water_flow", datasets={"fde_ground_water_flow": {"root": str(source_root)}}
    )
    target = benchmark._prepare_targets(blocked)[0]
    assert target["target"] == "u"
    assert target["preparation_row"]["status"] == "skipped"
    assert target["preparation_row"]["capability_status"] == "source_blocked"
    missing = case("fde_creep", datasets={"fde_creep": {"root": str(tmp_path / "missing")}})
    target = benchmark._prepare_targets(missing)[0]
    assert target["target"] == "ex"
    assert target["preparation_row"]["status"] == "error"
    assert target["preparation_row"]["capability_status"] == "missing_asset"
    assert target["preparation_row"]["recovery_eligible"] is False


def test_fde_aliases_are_inventory_entries_without_independent_fits(source_root):
    from scripts.export_asset_manifest import asset_manifest

    catalog, _ = benchmark.dataset_catalog()
    pending = {row["dataset"]: row for row in pending_datasets()}
    aliases = {"fde_burgers_alias", "fde_kdv_alias", "fde_tsfade_fft_dat_alias"}
    assert aliases.isdisjoint(catalog)
    assert all(pending[name]["benchmark_role"] == "duplicate" for name in aliases)
    report = asset_manifest(root=source_root)
    field = next(row for row in report["assets"] if row["path"] == "FDE/data/pt_fade.mat")
    assert field["datasets"] == ["fde_pt_fade", "fde_pt_fade_pde"]
    assert "u=f(x,t)" in field["targets"]


def test_response_config_dry_run_uses_all_sr_adapters_and_five_ready_records(tmp_path, lightweight_cli_environment):
    output = tmp_path / "manifest"
    assert (
        benchmark.main(
            [
                "--config",
                "configs/benchmark/fde_response.json",
                "--output-dir",
                str(output),
                "--dry-run",
            ]
        )
        == 0
    )
    report = json.loads((output / "benchmark_manifest.json").read_text(encoding="utf-8"))
    selected = report["cases"]
    assert {c["model"] for c in selected} == SR_METHODS
    assert {c["dataset"] for c in selected} == FDE_RESPONSE_DATASETS
    assert {c["metric_protocol_version"] for c in selected} == {"2.3"}


@pytest.mark.external_data(
    *(
        spec["path"].replace("FDE/", "kd/dataset/FDE/", 1)
        for name, spec in FDE_SPECS.items()
        if name in FDE_READY_DATASETS
    )
)
def test_real_archive_coordinates_and_observations_survive_the_runner():
    expected = {
        "fde_creep": (56, 1),
        "fde_silt": (43, 1),
        "fde_pt_fade": (1500, 2),
        "fde_pt_fde": (1500, 2),
        "fde_tsfade_fft": (18000, 2),
    }
    for name, shape in expected.items():
        dataset = load_dataset(name)
        assert isinstance(dataset, FractionalResponseDataset)
        assert dataset.X.shape == shape
        problem = next(benchmark.regression_problems(dataset, case(name)))
        time_axis = dataset.variable_names.index("t")
        assert problem["X_train"][:, time_axis].max() < problem["X_test"][:, time_axis].min()
        assert len(problem["X_test"]) > 0
        assert problem["ground_truth"] is None
    tsfade = load_dataset("fde_tsfade_fft")
    assert tsfade.x[-1] == 30.0 and tsfade.t[-1] == 15.0
    assert tsfade.x[1] == pytest.approx(30 / 119)
    assert tsfade.t[1] == pytest.approx(15 / 149)
    silt = load_dataset("fde_silt")
    np.testing.assert_array_equal(silt.X[:2, 0], [0.02447, 0.02447])
    np.testing.assert_array_equal(silt.y[:2], [0.91021, 1.15183])
