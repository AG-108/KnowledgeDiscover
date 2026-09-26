import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from kd.dataset import (
    DedalusBackend,
    GridPDEDataset,
    ODEDataset,
    ScatterPDEDataset,
    SymbolicRegressionDataset,
    TabularRegressionDataset,
    get_dataset_category,
    get_dataset_info,
    get_dataset_sym_true,
    list_available_datasets,
    list_datasets,
    load_ball_drop_dataset,
    load_cyt_dataset,
    load_cyt_flowfeature_raw,
    load_dataset,
    load_pde_grid,
    load_pdeformer_sinus_benchmark,
    load_rubber_dataset,
    load_solid_dif_dataset,
    load_solid_hardening_dataset,
    load_solid_strain_stress_dataset,
    load_vgs_dataset,
)


def test_list_available_datasets_contains_core_names():
    datasets = list_available_datasets()
    assert "kdv" in datasets
    assert "burgers" in datasets
    assert "chafee-infante" in datasets


def test_get_dataset_info_unknown_raises():
    with pytest.raises(ValueError):
        get_dataset_info("non-existent-dataset")


def test_load_pde_chafee_infante_uses_npy_bundle():
    dataset = load_pde_grid("chafee-infante")
    assert isinstance(dataset, GridPDEDataset)

    data = dataset.get_data()
    assert data["usol"].ndim == 2
    assert data["usol"].shape == (len(dataset.x), len(dataset.t))
    assert np.allclose(data["usol"], dataset.usol)


def test_load_pde_divide_single_npy_builds_domain():
    dataset = load_pde_grid("PDE_divide")
    info = get_dataset_info("PDE_divide")

    assert dataset.usol.shape == info["shape"]

    boundaries = dataset.get_boundaries()
    assert boundaries["x"] == pytest.approx(info["domain"]["x"])
    assert boundaries["t"] == pytest.approx(info["domain"]["t"])


def test_load_pde_fisher_mat_file():
    pytest.importorskip("scipy")

    dataset = load_pde_grid("fisher")
    assert dataset.usol.shape == (len(dataset.x), len(dataset.t))
    assert get_dataset_sym_true("fisher") is not None


def test_symbolic_regression_equidistant_specs_support_counts_and_cartesian_grids():
    counted = SymbolicRegressionDataset("Const-Test-1")
    cartesian = SymbolicRegressionDataset("Keijzer-11")

    assert counted.X_train.shape == (20, 1)
    assert cartesian.X_test.shape == (601 * 601, 2)
    assert np.isfinite(cartesian.y_test).all()


def test_symbolic_regression_keeps_large_finite_targets():
    dataset = SymbolicRegressionDataset("Korns-8")

    assert dataset.X_train.shape == (100, 5)
    assert np.isfinite(dataset.y_train).all()
    assert np.max(np.abs(dataset.y_train)) > 100


# ---------------------------------------------------------------------------
# ODEDataset / load_ball_drop_dataset
# ---------------------------------------------------------------------------


@pytest.mark.external_data(
    "kd/dataset/discovery-of-physics-from-data/data/Ball_drops_data.xls",
    "kd/dataset/discovery-of-physics-from-data/data/balls.txt",
)
def test_load_ball_drop_dataset_shapes():
    pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")

    dataset = load_ball_drop_dataset()
    assert isinstance(dataset, ODEDataset)

    # 9 balls (11 sheets minus Bowling Ball + Volleyball) x 2 drops each
    assert dataset.n_traj == 18
    assert dataset.state_vars == ["h", "v"]
    assert set(dataset.param_names) == {"mass", "diameter"}

    for traj in dataset.trajectories:
        assert traj["t"].shape[0] == traj["state"].shape[1]
        assert traj["state"].shape[0] == dataset.n_state
        assert traj["params"]["mass"] > 0
        assert traj["params"]["diameter"] > 0


@pytest.mark.external_data(
    "kd/dataset/discovery-of-physics-from-data/data/Ball_drops_data.xls",
    "kd/dataset/discovery-of-physics-from-data/data/balls.txt",
)
def test_ball_drop_falls_downward():
    pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")

    dataset = load_ball_drop_dataset()

    # A dropped ball's height should mostly decrease, i.e. velocity is
    # predominantly negative, over the course of each trajectory.
    for traj in dataset.trajectories:
        v = traj["state"][1]
        assert (v < 0).mean() > 0.5


@pytest.mark.external_data(
    "kd/dataset/discovery-of-physics-from-data/data/Ball_drops_data.xls",
    "kd/dataset/discovery-of-physics-from-data/data/balls.txt",
)
def test_ode_dataset_to_regression_arrays_shapes():
    pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")

    dataset = load_ball_drop_dataset()
    total_points = sum(traj["t"].shape[0] for traj in dataset.trajectories)

    X, y, variable_names = dataset.to_regression_arrays()
    assert X.shape == (total_points, dataset.n_state + len(dataset.param_names))
    assert y.shape == (total_points, dataset.n_state)
    assert variable_names == ["h", "v", "mass", "diameter"]

    X_no_params, y_no_params, names_no_params = dataset.to_regression_arrays(include_params=False)
    assert X_no_params.shape == (total_points, dataset.n_state)
    assert names_no_params == ["h", "v"]


@pytest.mark.external_data(
    "kd/dataset/discovery-of-physics-from-data/data/Ball_drops_data.xls",
    "kd/dataset/discovery-of-physics-from-data/data/balls.txt",
)
def test_ode_dataset_sample():
    pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")

    dataset = load_ball_drop_dataset()
    sampled_t, sampled_state = dataset.sample(20, seed=0)
    assert sampled_t.shape == (20,)
    assert sampled_state.shape == (20, dataset.n_state)


# ---------------------------------------------------------------------------
# TabularRegressionDataset / load_rubber_dataset
# ---------------------------------------------------------------------------


@pytest.mark.external_data(
    "kd/dataset/Discovery_of_soild_consititutive/data/data_rubber/train/*.xlsx"
)
def test_load_rubber_dataset_shapes():
    pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")

    dataset = load_rubber_dataset("train")
    assert isinstance(dataset, TabularRegressionDataset)

    data = dataset.get_data()
    assert data["variable_names"] == ["lambda", "C", "T"]
    assert data["n_input_dim"] == 3
    assert data["X"].shape[0] == data["y"].shape[0]
    assert data["groups"].shape[0] == data["X"].shape[0]

    # lambda = 1 + nominal strain, and nominal strain should be >= 0 for this data
    lam = data["X"][:, 0]
    assert np.all(lam >= 1.0)


@pytest.mark.external_data(
    "kd/dataset/Discovery_of_soild_consititutive/data/data_rubber/train/*.xlsx"
)
def test_rubber_dataset_train_test_split():
    pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")

    dataset = load_rubber_dataset("train")
    n_total = dataset.X.shape[0]

    dataset.train_test_split(test_size=0.2, seed=0)

    assert dataset.X_train.shape[0] + dataset.X_test.shape[0] == n_total
    assert dataset.X_train.shape[1] == dataset.X_test.shape[1] == 3


@pytest.mark.external_data(
    "kd/dataset/Discovery_of_soild_consititutive/data/data_rubber/train/*.xlsx",
    "kd/dataset/Discovery_of_soild_consititutive/data/data_rubber/test/*.xlsx",
)
def test_rubber_dataset_test_split_loads_separately():
    pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")

    train_dataset = load_rubber_dataset("train")
    test_dataset = load_rubber_dataset("test")

    assert train_dataset.X.shape[0] != test_dataset.X.shape[0]
    assert set(np.unique(test_dataset.groups)).isdisjoint(set(np.unique(train_dataset.groups)))


# ---------------------------------------------------------------------------
# Unified catalog: load_dataset / list_datasets / get_dataset_category
# ---------------------------------------------------------------------------


def test_list_datasets_covers_all_categories():
    all_names = list_datasets()

    assert "kdv" in all_names
    assert "ball_drop" in all_names
    assert "rubber_train" in all_names

    pde_names = list_datasets("pde")
    ode_names = list_datasets("ode")
    regression_names = list_datasets("regression")

    assert set(pde_names) | set(ode_names) | set(regression_names) == set(all_names)
    assert set(pde_names) & set(ode_names) == set()
    assert set(pde_names) & set(regression_names) == set()

    assert "ball_drop" in ode_names
    assert "rubber_train" in regression_names
    assert "rubber_test" in regression_names
    assert "kdv" in pde_names
    assert "chafee-infante" in pde_names


def test_get_dataset_category():
    assert get_dataset_category("kdv") == "pde"
    assert get_dataset_category("ball_drop") == "ode"
    assert get_dataset_category("rubber_train") == "regression"

    with pytest.raises(ValueError):
        get_dataset_category("non-existent-dataset")


def test_load_dataset_unknown_name_raises():
    with pytest.raises(ValueError):
        load_dataset("non-existent-dataset")


def test_load_dataset_pde_registry_matches_load_pde_grid():
    via_catalog = load_dataset("kdv")
    via_direct = load_pde_grid("kdv")

    assert isinstance(via_catalog, GridPDEDataset)
    assert via_catalog.usol.shape == via_direct.usol.shape
    np.testing.assert_allclose(via_catalog.usol, via_direct.usol)


@pytest.mark.external_data("kd/dataset/TLC/heat/heat_complex.csv")
def test_load_dataset_tlc_returns_scatter_dataset():
    dataset = load_dataset("tlc_heat_complex")
    assert isinstance(dataset, ScatterPDEDataset)
    assert dataset.n_times > 1


@pytest.mark.external_data("kd/dataset/WDwake/TI8_U.npy", "kd/dataset/WDwake/TI8_V.npy")
def test_load_dataset_wdwake_returns_grid_dataset():
    dataset = load_dataset("wdwake")
    assert isinstance(dataset, GridPDEDataset)
    assert dataset.n_response == 2


@pytest.mark.external_data(
    "kd/dataset/discovery-of-physics-from-data/data/Ball_drops_data.xls",
    "kd/dataset/discovery-of-physics-from-data/data/balls.txt",
    "kd/dataset/Discovery_of_soild_consititutive/data/data_rubber/train/*.xlsx",
)
def test_load_dataset_ode_and_regression():
    pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")

    ode_dataset = load_dataset("ball_drop")
    assert isinstance(ode_dataset, ODEDataset)

    reg_dataset = load_dataset("rubber_train")
    assert isinstance(reg_dataset, TabularRegressionDataset)


# ---------------------------------------------------------------------------
# CYT RANS turbulence-closure dataset
# ---------------------------------------------------------------------------

_CYT_EXPECTED_COLUMNS = [
    "X",
    "Y",
    "U",
    "V",
    "Ru",
    "P",
    "Ux",
    "Uy",
    "Vx",
    "Vy",
    "Px",
    "Py",
    "T",
    "dis",
    "Mut",
    "Txx",
    "Txz",
    "Tzz",
    "Ma",
    "AoA",
    "Re",
    "ydudy",
    "vc",
    "conv",
    "prod",
    "diff",
    "destr",
    "Sup_var1",
    "Sup_var2",
    "Sup_var3",
    "Sup_var4",
    "Sup_var5",
]


@pytest.mark.external_data("kd/dataset/CYT/FlatPlate_lk0.215andPplus/Output/FlowFeature.dat")
def test_load_cyt_flowfeature_raw_columns():
    raw = load_cyt_flowfeature_raw("flatplate")
    assert set(raw.keys()) == set(_CYT_EXPECTED_COLUMNS)

    n_cells = raw["X"].shape[0]
    assert n_cells > 0
    for col in _CYT_EXPECTED_COLUMNS:
        assert raw[col].shape == (n_cells,)
        assert np.all(np.isfinite(raw[col]))


@pytest.mark.external_data("kd/dataset/CYT/FlatPlate_lk0.215andPplus/Output/FlowFeature.dat")
def test_load_cyt_dataset_default_target():
    dataset = load_cyt_dataset("flatplate")
    assert isinstance(dataset, TabularRegressionDataset)

    data = dataset.get_data()
    assert data["variable_names"] == [
        "U",
        "V",
        "Ru",
        "P",
        "Ux",
        "Uy",
        "Vx",
        "Vy",
        "Px",
        "Py",
        "T",
        "dis",
        "Ma",
        "AoA",
        "Re",
    ]
    assert data["X"].shape[0] == data["y"].shape[0]
    assert data["n_input_dim"] == 15

    # Eddy viscosity (Mut) should be non-negative everywhere by physical definition.
    assert np.all(data["y"] >= 0)


@pytest.mark.external_data(
    "kd/dataset/CYT/FlatPlate_lk0.215andPplus/Output/FlowFeature.dat",
    "kd/dataset/CYT/NACA0012_Re4e5_MLen_BEST/Output/FlowFeature.dat",
)
def test_load_cyt_dataset_alternate_target_and_case():
    dataset = load_cyt_dataset("naca0012", target="Txx")
    data = dataset.get_data()
    assert data["X"].shape[0] == data["y"].shape[0]

    # naca0012 should have a different cell count than flatplate.
    flatplate = load_cyt_dataset("flatplate")
    assert dataset.X.shape[0] != flatplate.X.shape[0]


@pytest.mark.external_data("kd/dataset/CYT/FlatPlate_lk0.215andPplus/Output/FlowFeature.dat")
def test_load_cyt_dataset_custom_feature_columns():
    dataset = load_cyt_dataset("flatplate", target="Mut", feature_columns=["X", "Y"])
    data = dataset.get_data()
    assert data["variable_names"] == ["X", "Y"]
    assert data["X"].shape[1] == 2


def test_load_cyt_dataset_unknown_columns_fail_before_reading_data(monkeypatch):
    def unexpected_read(*args, **kwargs):
        raise AssertionError("Invalid columns must be rejected before reading external data")

    monkeypatch.setattr("kd.dataset._base.load_cyt_flowfeature_raw", unexpected_read)
    with pytest.raises(ValueError, match="Unknown target column"):
        load_cyt_dataset("flatplate", target="not_a_real_column")
    with pytest.raises(ValueError, match="Unknown feature column"):
        load_cyt_dataset("flatplate", feature_columns=["not_a_real_column"])


def test_load_cyt_dataset_unknown_case_raises():
    with pytest.raises(FileNotFoundError):
        load_cyt_dataset("not_a_real_case")


@pytest.mark.external_data("kd/dataset/CYT/FlatPlate_lk0.215andPplus/Output/FlowFeature.dat")
def test_load_dataset_cyt_cases_registered():
    assert "cyt_flatplate" in list_datasets()
    assert "cyt_naca0012" in list_datasets()
    assert get_dataset_category("cyt_flatplate") == "regression"

    dataset = load_dataset("cyt_flatplate")
    assert isinstance(dataset, TabularRegressionDataset)


# ---------------------------------------------------------------------------
# Solid-constitutive strain-rate / hardening datasets
# ---------------------------------------------------------------------------


@pytest.mark.external_data("kd/dataset/Discovery_of_soild_consititutive/data/data_DIF/*.xlsx")
def test_load_solid_dif_dataset_shapes():
    pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")

    dataset = load_solid_dif_dataset()
    assert isinstance(dataset, TabularRegressionDataset)

    data = dataset.get_data()
    assert data["variable_names"] == ["strain_rate"]
    assert data["X"].shape[0] == data["y"].shape[0]
    assert data["n_input_dim"] == 1
    assert data["groups"].shape[0] == data["X"].shape[0]

    # strain rate is a physical rate, must be positive.
    assert np.all(data["X"][:, 0] > 0)
    # DIF (Dynamic Increase Factor) should be roughly >= 1 (allowing some
    # experimental scatter below 1 at very low/reference strain rates).
    assert data["y"].min() > 0.5


@pytest.mark.external_data(
    "kd/dataset/Discovery_of_soild_consititutive/data/data_strain_stress/*.xlsx"
)
def test_load_solid_strain_stress_dataset_shapes():
    pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")

    dataset = load_solid_strain_stress_dataset()
    data = dataset.get_data()

    assert data["variable_names"] == ["strain", "strain_rate"]
    assert data["X"].shape[0] == data["y"].shape[0]
    assert data["groups"].shape[0] == data["X"].shape[0]

    # true plastic strain and strain rate must both be non-negative.
    assert np.all(data["X"][:, 0] >= 0)
    assert np.all(data["X"][:, 1] > 0)

    # The one anomalous filename ("...1-e4.xlsx") must have been parsed as
    # 1e-4, not dropped or misparsed.
    assert np.any(np.isclose(data["X"][:, 1], 1e-4))


@pytest.mark.external_data(
    "kd/dataset/Discovery_of_soild_consititutive/data/saved_data_hardening_strain_rate/*.pkl"
)
def test_load_solid_hardening_dataset_shapes():
    pytest.importorskip("pandas")

    dataset = load_solid_hardening_dataset()
    data = dataset.get_data()

    assert data["variable_names"] == ["strain", "strain_rate", "DIF"]
    assert data["X"].shape[0] == data["y"].shape[0]
    assert data["groups"].shape[0] == data["X"].shape[0]
    assert np.all(data["X"][:, 1] > 0)  # strain_rate


@pytest.mark.external_data("kd/dataset/Discovery_of_soild_consititutive/data/data_DIF/*.xlsx")
def test_load_dataset_solid_registered():
    for name in ("solid_dif", "solid_strain_stress", "solid_hardening"):
        assert name in list_datasets()
        assert get_dataset_category(name) == "regression"

    dataset = load_dataset("solid_dif")
    assert isinstance(dataset, TabularRegressionDataset)


# ---------------------------------------------------------------------------
# Viscous gravity current (VGS) proppant-transport PDE dataset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case,window",
    [
        ("I", "0-100"),
        ("I", "100-200"),
        ("II", "0-1000"),
        ("II", "1000-2000"),
    ],
)
@pytest.mark.external_data(
    "kd/dataset/ViscousGravityCurrent/data/vgs_I_0-100.dat",
    "kd/dataset/ViscousGravityCurrent/data/vgs_I_100-200.dat",
    "kd/dataset/ViscousGravityCurrent/data/vgs_II_0-1000.dat",
    "kd/dataset/ViscousGravityCurrent/data/vgs_II_1000-2000.dat",
)
def test_load_vgs_dataset_shapes(case, window):
    dataset = load_vgs_dataset(case=case, window=window)
    assert isinstance(dataset, GridPDEDataset)
    assert dataset.usol.shape == (500, 500)
    assert dataset.x.shape == (500,)
    assert dataset.t.shape == (500,)
    assert np.all(np.isfinite(dataset.usol))

    # time axis must start at 0 and be strictly increasing
    assert dataset.t[0] == 0.0
    assert np.all(np.diff(dataset.t) > 0)


@pytest.mark.external_data(
    "kd/dataset/ViscousGravityCurrent/data/vgs_I_0-100.dat",
    "kd/dataset/ViscousGravityCurrent/data/vgs_I_100-200.dat",
    "kd/dataset/ViscousGravityCurrent/data/vgs_II_0-1000.dat",
    "kd/dataset/ViscousGravityCurrent/data/vgs_II_1000-2000.dat",
)
def test_load_vgs_dataset_default_window():
    default_I = load_vgs_dataset(case="I")
    explicit_I = load_vgs_dataset(case="I", window="0-100")
    np.testing.assert_array_equal(default_I.usol, explicit_I.usol)

    default_II = load_vgs_dataset(case="II")
    explicit_II = load_vgs_dataset(case="II", window="0-1000")
    np.testing.assert_array_equal(default_II.usol, explicit_II.usol)


@pytest.mark.external_data("kd/dataset/ViscousGravityCurrent/data/vgs_I_0-100.dat")
def test_load_vgs_dataset_case_aliases():
    via_alias = load_vgs_dataset(case="1")
    via_name = load_vgs_dataset(case="I")
    np.testing.assert_array_equal(via_alias.usol, via_name.usol)


def test_load_vgs_dataset_unknown_case_raises():
    with pytest.raises(ValueError):
        load_vgs_dataset(case="III")


def test_load_vgs_dataset_unknown_window_raises():
    with pytest.raises(ValueError):
        load_vgs_dataset(case="I", window="not_a_window")


@pytest.mark.external_data(
    "kd/dataset/ViscousGravityCurrent/data/vgs_I_0-100.dat",
    "kd/dataset/ViscousGravityCurrent/data/vgs_I_100-200.dat",
)
def test_load_vgs_dataset_windows_are_distinct():
    # Different windows of the same case should not hold identical data.
    a = load_vgs_dataset(case="I", window="0-100")
    b = load_vgs_dataset(case="I", window="100-200")
    assert not np.array_equal(a.usol, b.usol)


@pytest.mark.external_data("kd/dataset/ViscousGravityCurrent/data/vgs_I_0-100.dat")
def test_load_dataset_vgs_registered():
    vgs_names = [n for n in list_datasets() if n.startswith("vgs_")]
    assert set(vgs_names) == {
        "vgs_I_0-100",
        "vgs_I_100-200",
        "vgs_II_0-1000",
        "vgs_II_1000-2000",
    }
    for name in vgs_names:
        assert get_dataset_category(name) == "pde"

    dataset = load_dataset("vgs_I_0-100")
    assert isinstance(dataset, GridPDEDataset)


# ---------------------------------------------------------------------------
# PDEformer-1D 'sinus' random-PDE benchmark generator
# ---------------------------------------------------------------------------

_SINUS_KWARGS = dict(n_pde=2, n_x=64, n_t=21)


def test_load_pdeformer_sinus_benchmark_shapes():
    datasets = load_pdeformer_sinus_benchmark(seed=0, **_SINUS_KWARGS)
    assert len(datasets) == 2

    for dataset in datasets:
        assert isinstance(dataset, GridPDEDataset)
        assert dataset.usol.shape == (64, 21)
        assert np.all(np.isfinite(dataset.usol))
        assert dataset.x.min() == pytest.approx(-1.0)
        assert dataset.t.min() == 0.0
        assert dataset.t.max() == pytest.approx(1.0)

        assert isinstance(dataset.sym_true, str) and len(dataset.sym_true) > 0
        assert set(dataset.coef_dict) == {
            "f0_poly",
            "f1_poly",
            "s",
            "s_is_field",
            "kappa",
            "kappa_is_field",
            "ic",
        }


def test_load_pdeformer_sinus_benchmark_reproducible():
    ds1 = load_pdeformer_sinus_benchmark(seed=7, **_SINUS_KWARGS)
    ds2 = load_pdeformer_sinus_benchmark(seed=7, **_SINUS_KWARGS)

    assert len(ds1) == len(ds2)
    for a, b in zip(ds1, ds2):
        np.testing.assert_array_equal(a.usol, b.usol)
        assert a.sym_true == b.sym_true


def test_load_pdeformer_sinus_benchmark_different_seeds_differ():
    ds1 = load_pdeformer_sinus_benchmark(seed=1, **_SINUS_KWARGS)
    ds2 = load_pdeformer_sinus_benchmark(seed=2, **_SINUS_KWARGS)
    assert not np.array_equal(ds1[0].usol, ds2[0].usol)


def test_load_pdeformer_sinus_benchmark_sym_true_reflects_field_coefs():
    # A large enough batch should include both scalar/zero and field-valued
    # s/kappa draws (field_prob=1/3 default), and the sym_true string must
    # use the symbolic placeholder exactly when the coefficient is a field.
    datasets = load_pdeformer_sinus_benchmark(n_pde=8, n_x=32, n_t=11, seed=3)
    for dataset in datasets:
        coef_dict = dataset.coef_dict
        assert ("s(x)" in dataset.sym_true) == coef_dict["s_is_field"]
        assert ("kappa(x)" in dataset.sym_true) == coef_dict["kappa_is_field"]


def test_load_pdeformer_sinus_benchmark_via_catalog():
    assert "pdeformer_sinus" in list_datasets()
    assert get_dataset_category("pdeformer_sinus") == "pde"

    datasets = load_dataset("pdeformer_sinus", **_SINUS_KWARGS, seed=0)
    assert isinstance(datasets, list)
    assert len(datasets) == 2
    assert isinstance(datasets[0], GridPDEDataset)


def test_dedalus_backend_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        load_pdeformer_sinus_benchmark(n_pde=1, n_x=16, backend=DedalusBackend())
