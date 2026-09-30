"""Protect scientific identities, leakage boundaries and field axis semantics."""

import copy

import numpy as np
import pandas as pd
import pytest

import run_benchmark as benchmark
from examples._common import point_cloud_to_regular_grid
from kd.dataset import (
    GridPDEDataset,
    ScatterPDEDataset,
    TabularRegressionDataset,
    _base,
    _domain,
    load_csv_tlc,
    load_dataset,
)
from kd.dataset._metadata import pending_datasets
from kd.dataset._provenance import source_evidence
from kd.evaluation.reporting import paired_comparison


def case(name, model="gplearn", **kwargs):
    catalog, _ = benchmark.dataset_catalog()
    return benchmark.build_experiments(catalog, models=[model], datasets=[name], **kwargs)[0]


def test_cartesian_tlc_preserves_permuted_values_and_channel_order(tmp_path):
    points = np.array([[2, 3], [1, 2], [2, 2], [1, 3]])
    table = {"X": points[:, 0], "Y": points[:, 1]}
    for t in (0, 1, 2):
        table[f"v @ t={t}"] = points[:, 0] + 10 * points[:, 1] + 100 * t
        table[f"u @ t={t}"] = -table[f"v @ t={t}"]
    path = tmp_path / "field.csv"
    pd.DataFrame(table).to_csv(path, index=False)
    grid = load_csv_tlc(path, return_dataset=True, grid_if_cartesian=True)
    assert isinstance(grid, GridPDEDataset)
    assert grid.state_vars == ["v", "u"]
    assert grid.usol.shape == (2, 2, 2, 3)
    for ix, x in enumerate(grid.coords["x"]):
        for iy, y in enumerate(grid.coords["y"]):
            np.testing.assert_array_equal(grid.usol[0, ix, iy], x + 10*y + 100*np.arange(3))
    assert grid.protocol_metadata["interpolated"] is False
    # Missing coordinates must not be filled or interpolated into a grid.
    pd.DataFrame(table).iloc[:-1].to_csv(path, index=False)
    assert isinstance(load_csv_tlc(path, return_dataset=True, grid_if_cartesian=True), ScatterPDEDataset)
    duplicate = pd.DataFrame(table)
    duplicate.iloc[0] = duplicate.iloc[1]
    duplicate.to_csv(path, index=False)
    assert isinstance(load_csv_tlc(path, return_dataset=True, grid_if_cartesian=True), ScatterPDEDataset)


def test_cyt_blocks_use_only_coordinates_and_hold_out_whole_regions(monkeypatch):
    x, y = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
    raw = {key: np.ones(x.size) for key in _base._CYT_COLUMNS}
    raw.update(X=x.ravel(), Y=y.ravel(), Mut=(x+y).ravel())
    monkeypatch.setattr(_base, "load_cyt_flowfeature_raw", lambda _: raw)
    dataset = _base.load_cyt_dataset()
    original = dataset.groups.copy()
    problem = next(benchmark.regression_problems(dataset, case("cyt_flatplate")))
    assert problem["target"] == "Mut"
    assert problem["split"] == "spatial_block_holdout"
    assert not set(problem["metadata"]["train_groups"]) & set(problem["metadata"]["test_groups"])
    raw["Mut"] = np.random.default_rng(4).normal(size=x.size)
    np.testing.assert_array_equal(_base.load_cyt_dataset().groups, original)
    points = np.repeat(np.column_stack([x.ravel(), y.ravel()]), 2, axis=0)
    groups, _ = _domain.spatial_blocks(points)
    np.testing.assert_array_equal(groups[::2], groups[1::2])
    with pytest.raises(ValueError, match="input feature"):
        _base.load_cyt_dataset(feature_columns=["Mut"])


def test_constitutive_materials_are_separate_and_dif_cannot_leak(monkeypatch, tmp_path):
    directory = tmp_path / "Discovery_of_soild_consititutive/data/saved_data_hardening_strain_rate"
    directory.mkdir(parents=True)
    for name in ("material_a", "material_b"):
        (directory / f"{name}.pkl").touch()
    monkeypatch.setattr(_domain, "ROOT", tmp_path)
    X = np.array([[i, rate, 900+i] for _ in range(2) for rate in (1, 2, 3) for i in range(4)])
    y = np.arange(len(X), dtype=float)
    groups = np.array([f"material_{m}_curve{r}" for m in ("a", "b") for r in range(3) for _ in range(4)])
    raw = TabularRegressionDataset("pooled", X, y, variable_names=["strain", "strain_rate", "DIF"], groups=groups)
    monkeypatch.setattr(_base, "load_solid_hardening_dataset", lambda: raw)
    instances = list(_domain.load_constitutive_instances("solid_hardening"))
    assert [name for name, _ in instances] == ["solid_hardening/material_a", "solid_hardening/material_b"]
    for i, (_, dataset) in enumerate(instances):
        assert dataset.variable_names == ["strain", "strain_rate"]
        np.testing.assert_array_equal(dataset.y, y[i*12:(i+1)*12])
        problem = next(benchmark.regression_problems(dataset, case("solid_hardening")))
        assert problem["target"] == "stress"
        assert set(problem["X_train"][:, 1]).isdisjoint(problem["X_test"][:, 1])
        assert problem["metadata"]["DIF_predictor_used"] is False
    failed = benchmark._case_failure_targets(case("solid_hardening"), RuntimeError("bad input"))
    assert len(failed) == 2
    assert all(s["preparation_row"]["target_manifest_complete"] for s in failed)


def test_task_metadata_changes_identity_and_separates_results():
    catalog, excluded = benchmark.dataset_catalog()
    assert catalog["cyt_flatplate"]["scientific_task"] == "turbulence_closure"
    assert catalog["solid_dif"]["scientific_task"] == "constitutive_law"
    assert catalog["heat"]["reference_status"] == "unknown"
    assert "green_laplace" in catalog and "tlc_steady_poisson_3d" in catalog
    assert "green_negative_helmholtz_data2" in excluded
    assert catalog["green_laplace"]["task"] == "green"
    assert catalog["tlc_steady_poisson_3d"]["task"] == "steady_pde"
    assert len([r for r in pending_datasets() if r["dataset"].startswith("dso_file_")]) == 0
    assert len([n for n in catalog if n.startswith("dso_file_")]) == 25
    first = case("Koza-2")
    catalog["Koza-2"]["dataset_protocol_version"] = "future-protocol"
    second = benchmark.build_experiments(catalog, datasets=["Koza-2"], models=["gplearn"])[0]
    assert first["name"] != second["name"]
    row = benchmark._base_row(first, target="y")
    row.update(status="ok", test_nrmse=0.2)
    other = {**row, "dataset_protocol_version": "future-protocol", "model": "dso"}
    assert paired_comparison([row, other], "test_nrmse", "gplearn", "dso")["n_pairs"] == 0
    stats = benchmark.aggregate_results([row, {**row, "scientific_task": "constitutive_law"}])
    assert len([s for s in stats if s["model"] == "gplearn"]) == 2


def test_old_dataset_protocol_is_rejected_before_creating_results(tmp_path):
    old_case = case("solid_dif")
    old_case.pop("dataset_protocol_version")
    with pytest.raises(ValueError, match="Legacy dataset protocol"):
        benchmark.execute_case(old_case, tmp_path, resume=True)
    assert not list(tmp_path.iterdir())


def test_wake_source_validation_is_required_even_on_equal_spacing():
    coords = {"x": np.arange(5.), "y": np.arange(5.), "t": np.arange(6.)}
    # The raw loader calls the equation Wake, whereas the registry key is wdwake.
    dataset = GridPDEDataset("Wake", None, None, 0, coords=coords, usol=np.zeros((2, 5, 5, 6)))
    with pytest.raises(benchmark.SkipCase, match="physical coordinates"):
        benchmark.prepare_pde(dataset, "pdenet")


@pytest.mark.external_data("kd/dataset/TLC/burgers/burgers1d.csv",
                           "kd/dataset/TLC/heat/heat_longtime.csv",
                           "kd/dataset/TLC/heat/heat_multiscale_lesspoints.csv")
def test_three_real_tlc_grids_are_restored_without_interpolation():
    for name, shape in [("tlc_burgers1d", (1, 101, 11)), ("tlc_heat_longtime", (1, 16, 12, 501)),
                        ("tlc_heat_multiscale_lesspoints", (1, 300, 10, 26))]:
        dataset = load_dataset(name)
        assert dataset.usol.shape == shape
        assert dataset.protocol_metadata["interpolated"] is False
        if name == "tlc_burgers1d":
            prepared = benchmark.prepare_pde(dataset, "pdefind")
            train, test, _ = benchmark.temporal_block_holdout(prepared, 0.2)
            assert train.t[-1] == test.t[0]  # only shared initial condition
        else:
            with pytest.raises(benchmark.SkipCase, match="anisotropic"):
                benchmark.prepare_pde(dataset, "pdenet")


@pytest.mark.external_data("kd/dataset/data/Heat_equation.mat", "kd/dataset/data/burgers.mat",
                           "kd/dataset/EqGPT/data/Allen_Cahn/Allen_Cahn.mat",
                           "kd/dataset/EqGPT/data/Convection_diffusion_equation/data.mat")
def test_new_fields_have_embedded_axes_and_declared_reference_agreement():
    for name in _domain.FIELD_SPECS:
        dataset = load_dataset(name)
        assert dataset.usol.shape == (len(dataset.x), len(dataset.t))
        assert dataset.protocol_metadata["coordinates"] == "embedded"
        if name == "heat":
            assert not dataset.sym_true  # fitted diffusivity is not published truth
            continue
        u, x, t = dataset.usol, dataset.x, dataset.t
        ux = np.gradient(u, x, axis=0, edge_order=2)
        uxx = np.gradient(ux, x, axis=0, edge_order=2)
        ut = np.gradient(u, t, axis=1, edge_order=2)
        rhs = {"burgers_sine": -u*ux + .1*uxx,
               "allen_cahn": .003*uxx + u - u**3,
               "convection_diffusion": -ux + .25*uxx}[name]
        interior = (slice(4, -4), slice(2, -2))
        residual = np.linalg.norm((ut-rhs)[interior]) / np.linalg.norm(ut[interior])
        assert residual < .15, (name, residual)


@pytest.mark.external_data("kd/dataset/Discovery_of_soild_consititutive/data/data_DIF/*.xlsx",
                           "kd/dataset/Discovery_of_soild_consititutive/data/data_strain_stress/*.xlsx",
                           "kd/dataset/Discovery_of_soild_consititutive/data/saved_data_hardening_strain_rate/*.pkl")
def test_real_constitutive_instances_keep_all_rows_and_rate_holdouts():
    for name, count, n_rows in [("solid_dif", 63, 408), ("solid_strain_stress", 12, 1314),
                                ("solid_hardening", 10, 1096)]:
        instances = list(benchmark._load_instances(case(name)))
        assert len(instances) == count
        assert sum(len(data.y) for _, data in instances) == n_rows
        for _, dataset in instances:
            train, test = benchmark.split_indices(len(dataset.X), .2, 0, dataset.groups)
            if len(np.unique(dataset.groups[train])) < 2 or np.ptp(dataset.y[train]) == 0:
                with pytest.raises(benchmark.SkipCase, match="eligibility"):
                    next(benchmark.regression_problems(dataset, case(name)))
                continue
            problem = next(benchmark.regression_problems(dataset, case(name)))
            assert len(problem["X_train"]) >= 2 and len(problem["X_test"]) >= 1
            assert not set(problem["metadata"]["train_groups"]) & set(problem["metadata"]["test_groups"])


def test_unknown_units_and_coefficients_remain_unknown():
    catalog, _ = benchmark.dataset_catalog()
    assert catalog["heat"]["reference_status"] == "unknown"
    assert catalog["rubber_train"]["source_evidence"]["units"]["T"].startswith("unknown")
    for name in ("solid_dif", "solid_strain_stress", "solid_hardening"):
        assert catalog[name]["source_evidence"]["material_id"] is None
    assert source_evidence("convection_diffusion")["boundary"] == "unknown"


@pytest.mark.parametrize("features", [["prod"], ["Mut"], ["Txx"], ["Sup_var1"], [], ["U", "U"]])
def test_cyt_rejects_unverified_target_derived_features_before_read(monkeypatch, features):
    def unexpected_read(_):
        pytest.fail("Rejected schema must not read the external asset")
    monkeypatch.setattr(_base, "load_cyt_flowfeature_raw", unexpected_read)
    with pytest.raises(ValueError):
        _base.load_cyt_dataset(feature_columns=features)


def test_cyt_test_target_perturbation_preserves_training_arrays(monkeypatch):
    x, y = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
    raw = {key: np.ones(x.size) for key in _base._CYT_COLUMNS}
    raw.update(X=x.ravel(), Y=y.ravel(), Mut=(x+y).ravel().astype(float))
    monkeypatch.setattr(_base, "load_cyt_flowfeature_raw", lambda _: raw)
    first = _base.load_cyt_dataset()
    problem = next(benchmark.regression_problems(first, case("cyt_flatplate")))
    held_out = np.isin(first.groups, problem["metadata"]["test_groups"])
    raw["Mut"][held_out] += 100000
    second = _base.load_cyt_dataset()
    changed = next(benchmark.regression_problems(second, case("cyt_flatplate")))
    for key in ("X_train", "y_train", "X_test"):
        np.testing.assert_array_equal(problem[key], changed[key])
    np.testing.assert_array_equal(first.groups, second.groups)
    assert problem["metadata"] == changed["metadata"]


def test_constitutive_qualification_and_setup_denominator(monkeypatch):
    data = TabularRegressionDataset("small", [[0, 1], [1, 1], [0, 2], [1, 2]],
                                    [2, 3, 4, 5], groups=[1, 1, 2, 2],
                                    variable_names=["strain", "strain_rate"])
    data.target_name = "stress"
    data.protocol_metadata = {"source_asset_ids": ["synthetic/small"], "material_id": None}
    selected = case("solid_strain_stress")
    monkeypatch.setattr(benchmark, "_load_instances", lambda _: iter([("small", data)]))
    specs = benchmark._prepare_targets(selected)
    assert len(specs) == 1 and specs[0]["target"] == "stress"
    row = specs[0]["preparation_row"]
    assert row["status"] == "skipped" and row["target_manifest_complete"]
    assert "eligibility" in row["error"]
    assert row["details"]["source_asset_ids"] == ["synthetic/small"]
    data.groups = np.arange(4)
    data.y[:] = 3
    with pytest.raises(benchmark.SkipCase, match="constant training target"):
        next(benchmark.regression_problems(data, selected))


def test_exact_repeated_rates_and_raw_linear_coefficients():
    rates = np.repeat([.001, .001, .1, 1, 10], 4)
    strain = np.tile([.1, .2, .3, .4], 5)
    X = np.column_stack([strain, rates])
    physical = 7 + 3*strain + 2*rates
    data = TabularRegressionDataset("scaled", X, physical/1000, groups=rates,
                                    variable_names=["strain", "strain_rate"])
    selected = case("solid_strain_stress")
    problem = next(benchmark.regression_problems(data, selected))
    assert set(problem["X_train"][:, 1]).isdisjoint(problem["X_test"][:, 1])
    design = np.column_stack([np.ones(len(problem["X_train"])), problem["X_train"]])
    coeffs = np.linalg.lstsq(design, problem["y_train"], rcond=None)[0]
    np.testing.assert_allclose(coeffs*1000, [7, 3, 2], rtol=1e-10, atol=1e-10)
    prediction = np.column_stack([np.ones(len(problem["X_test"])), problem["X_test"]]) @ coeffs
    np.testing.assert_allclose(prediction*1000, 7+3*problem["X_test"][:, 0]+2*problem["X_test"][:, 1])
    # Test targets are never consulted for scaling or qualification.
    held_out = np.isin(rates, problem["metadata"]["test_groups"])
    data.y[held_out] += 999
    changed = next(benchmark.regression_problems(data, selected))
    np.testing.assert_array_equal(problem["X_train"], changed["X_train"])
    np.testing.assert_array_equal(problem["y_train"], changed["y_train"])


def test_similar_source_names_remain_separate_and_rate_tokens_normalize(tmp_path, monkeypatch):
    directory = tmp_path / "Discovery_of_soild_consititutive/data/data_strain_stress"
    directory.mkdir(parents=True)
    for stem in ("Steel_2010_1e-4", "Steel_2010_1-e4", "steel_2010_0.1", "Steel_2011_0.1"):
        (directory / (stem+".xlsx")).touch()
    monkeypatch.setattr(_domain, "ROOT", tmp_path)
    specs = _domain.constitutive_instance_specs("solid_strain_stress")
    assert len(specs) == 3
    assert all(s["material_id"] is None for s in specs)
    assert _base._parse_strain_rate_token("1e-4") == _base._parse_strain_rate_token("1-e4")


def test_mat_axis_orientation_is_declared_even_for_square_arrays(tmp_path, monkeypatch):
    from scipy.io import savemat
    spec = _domain.FIELD_SPECS["convection_diffusion"]
    path = tmp_path / spec["path"]
    path.parent.mkdir(parents=True)
    x, t = np.arange(5.), np.arange(5.)*.2
    physical = 3*x[:, None] + 7*t[None, :]
    savemat(path, {"x": x, "t": t, "u": physical.T})
    monkeypatch.setattr(_domain, "ROOT", tmp_path)
    loaded = _domain.load_domain_field("convection_diffusion")
    np.testing.assert_array_equal(loaded.usol, physical)
    np.testing.assert_allclose(np.gradient(loaded.usol, t, axis=1, edge_order=2), 7)
    np.testing.assert_allclose(np.gradient(loaded.usol, x, axis=0, edge_order=2), 3)


def test_pde_derivatives_and_metadata_are_isolated_after_split():
    x, t = np.linspace(-1, 1, 21), np.arange(10.)*.1
    u = x[:, None]**2 + 3*t[None, :]
    dataset = GridPDEDataset("synthetic", None, None, 0, x=x, t=t, usol=u, legacy=True)
    dataset.protocol_metadata = {"coordinate_units": {"x": "m", "t": "s"},
                                 "boundary": {"type": "unknown"}, "state_vars": ["u"]}
    dataset.rollout_metadata = {"boundary_condition": "periodic"}
    prepared = benchmark.prepare_pde(dataset, "pdefind")
    train, evaluation, split = benchmark.temporal_block_holdout(prepared, .3)
    before = np.gradient(train.usol, train.t, axis=1, edge_order=2)
    dataset.usol[..., split:] += 12345
    changed = benchmark.prepare_pde(dataset, "pdefind")
    next_train, _, _ = benchmark.temporal_block_holdout(changed, .3)
    np.testing.assert_array_equal(before, np.gradient(next_train.usol, next_train.t, axis=1, edge_order=2))
    np.testing.assert_allclose(before, 3, atol=1e-12)
    # Mutating a scoring view must not mutate either the fitting view or source.
    evaluation.protocol_metadata["coordinate_units"]["x"] = "bad"
    evaluation.usol[:] = 0
    assert train.protocol_metadata["coordinate_units"]["x"] == "m"
    assert dataset.protocol_metadata["coordinate_units"]["x"] == "m"
    assert train.rollout_metadata == dataset.rollout_metadata
    assert train.usol.any()


def test_cartesian_multichannel_validation_preserves_evidence():
    payload = {"points": np.array([[1.], [0.]]), "usol": np.zeros((2, 3, 2)),
               "t": np.arange(3.), "spatial_vars": ["X"], "time_var": "t",
               "state_vars": ["v", "u"], "equation_name": "test", "domain": None, "epi": 0,
               "coordinate_units": {"x": "m", "t": "s"}, "boundary": "unknown",
               "protocol_metadata": {"source_asset_id": "test.csv"}}
    grid = _domain.cartesian_grid(payload)
    assert grid.n_response == 2 and grid.state_vars == ["v", "u"]
    assert grid.protocol_metadata["coordinate_units"]["x"] == "m"
    assert grid.protocol_metadata["source_asset_id"] == "test.csv"
    with pytest.raises(benchmark.SkipCase):
        benchmark.prepare_pde(grid, "pdefind")
    malformed = copy.deepcopy(payload)
    malformed["state_vars"] = ["u"]
    with pytest.raises(ValueError, match="channel layout"):
        _domain.cartesian_grid(malformed)


def test_wave_projection_never_interpolates_across_time():
    rows = np.array([[t, x, 2*x+t] for t in range(6) for x in np.linspace(0, 1, 8)])
    _, _, original = point_cloud_to_regular_grid(rows, nx=12)
    rows[rows[:, 0] >= 4, 2] += 10000
    _, _, changed = point_cloud_to_regular_grid(rows, nx=12)
    np.testing.assert_array_equal(original[:4], changed[:4])


@pytest.mark.external_data("kd/dataset/TLC/burgers/burgers1d.csv")
def test_real_tlc_csv_row_permutation_has_identical_field(tmp_path):
    import pandas as pd
    original = load_dataset("tlc_burgers1d")
    path = _domain.ROOT / "TLC/burgers/burgers1d.csv"
    table = pd.read_csv(path)
    shuffled = tmp_path / "permuted.csv"
    table.sample(frac=1, random_state=13).to_csv(shuffled, index=False, float_format="%.17g")
    # Compare both serializations so CSV parser roundoff is the same.
    ordered = tmp_path / "ordered.csv"
    table.to_csv(ordered, index=False, float_format="%.17g")
    baseline = _base.load_csv_tlc(ordered, return_dataset=True, grid_if_cartesian=True)
    changed = _base.load_csv_tlc(shuffled, return_dataset=True, grid_if_cartesian=True)
    np.testing.assert_array_equal(baseline.usol, changed.usol)
    assert original.usol.shape == changed.usol.shape


def test_training_sample_cap_cannot_remove_rate_variation():
    data = TabularRegressionDataset("cap", [[i, rate] for rate in (1, 2, 3) for i in range(20)],
                                    np.arange(60), groups=np.repeat([1, 2, 3], 20),
                                    variable_names=["strain", "strain_rate"])
    selected = case("solid_strain_stress")
    selected["run_params"]["max_train_samples"] = 2
    # Choose a seed whose permitted cap happens to sample a single rate.
    for seed in range(50):
        selected["seed"] = seed
        train, _ = benchmark.split_indices(len(data.X), .2, seed, data.groups)
        ids = np.random.default_rng(seed).choice(len(train), 2, replace=False)
        if len(np.unique(data.X[train[ids], 1])) == 1:
            with pytest.raises(benchmark.SkipCase, match="sample cap"):
                next(benchmark.regression_problems(data, selected))
            return
    pytest.fail("Synthetic cap fixture must exercise a single-rate subsample")


def test_hardening_loader_rejects_truncated_parallel_lists(monkeypatch, tmp_path):
    import pickle
    root = tmp_path / "saved_data_hardening_strain_rate"
    root.mkdir()
    path = root / "broken.pkl"
    path.write_bytes(pickle.dumps({"strain": [np.array([.1, .2])],
                                  "stress": [np.array([1., 2.])],
                                  "strain_rate": [1], "DIF": []}))
    monkeypatch.setattr(_base, "_SOLID_DATA_DIR", tmp_path)
    with pytest.raises(ValueError, match="curve lists"):
        _base.load_solid_hardening_dataset()


def test_nonfinite_solid_observations_are_not_silently_dropped(monkeypatch, tmp_path):
    import pandas as pd
    root = tmp_path / "data_DIF"
    root.mkdir()
    (root / "broken.xlsx").touch()
    monkeypatch.setattr(_base, "_SOLID_DATA_DIR", tmp_path)
    monkeypatch.setattr(pd, "read_excel", lambda _: pd.DataFrame(
        {"strain rate": [1, 2, 3], "DIF": [1, np.nan, 2]}))
    with pytest.raises(ValueError, match="no silent row filtering"):
        _base.load_solid_dif_dataset()


@pytest.mark.external_data("kd/dataset/Discovery_of_soild_consititutive/data/data_rubber/train/*.xlsx",
                           "kd/dataset/Discovery_of_soild_consititutive/data/data_rubber/test/*.xlsx")
def test_rubber_official_split_holds_out_parameter_combinations():
    train, test = load_dataset("rubber_train"), load_dataset("rubber_test")
    train_conditions = set(map(tuple, train.X[:, 1:]))
    test_conditions = set(map(tuple, test.X[:, 1:]))
    assert len(train_conditions) == 20 and len(test_conditions) == 4
    assert train_conditions.isdisjoint(test_conditions)
    assert set(train.X[:, 1]) == set(test.X[:, 1])
    assert set(test.X[:, 2]).issubset(train.X[:, 2])
