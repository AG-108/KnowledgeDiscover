import os
import pickle

import numpy as np
from _common import bootstrap_project_root, build_wave_dataset_for_dscv, point_cloud_to_regular_grid

project_root = bootstrap_project_root()


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="numpy.*")
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow.*")
from kd.data import RegularData
from kd.model import KD_DSCV


def attach_regression_data_to_model(
    model,
    X,
    y,
    variable_names=None,
    dataset_name="regression_dataset",
    task_type="symbolic_regression",
    sym_true="",
):
    """
    Attach ordinary regression data to KD_DSCV without modifying KD_DSCV class.

    The target is:
        y ≈ Θ(X) w
    """

    X_arr = np.asarray(X)

    if X_arr.ndim == 1:
        n_input_dim = 1
    else:
        n_input_dim = X_arr.shape[1]

    model.data_class = RegularData(dataset_name)

    model.data_class.load_regression_data(
        X=X,
        y=y,
        variable_names=variable_names,
        sym_true=sym_true,
        n_input_dim=n_input_dim,
    )

    model.dataset = dataset_name

    model.config_task["task_type"] = task_type
    model.config_task["dataset"] = dataset_name

    if model.out_path is not None:
        model.out_path = os.path.join(model.out_path, f"discover_{dataset_name}_{model.seed}.csv")

    model.setup()

    return model


exp_name = "wave_breaking"  # "sc_rubber" or "wave_breaking"

if exp_name == "sc_rubber":

    def load_rubber_excel_dataset(
        data_dir,
        include_conditions=True,
        normalize_conditions=True,
        single_file=None,
    ):
        import os
        import re

        import numpy as np
        import pandas as pd

        X_list = []
        y_list = []
        curve_info = []

        for filename in sorted(os.listdir(data_dir)):
            if not filename.endswith(".xlsx"):
                continue

            if filename.startswith("~$"):
                continue

            if single_file is not None and filename != single_file:
                continue

            match = re.match(r"C(\d+)_(\d+)\.xlsx$", filename)
            if match is None:
                print(f"Skip unexpected file name: {filename}")
                continue

            C_value = float(match.group(1))
            T_value = float(match.group(2))

            file_path = os.path.join(data_dir, filename)
            df = pd.read_excel(file_path, engine="openpyxl")

            strain = df["nominal strain"].to_numpy(dtype=float)
            stress = df["nominal stress"].to_numpy(dtype=float)

            lam = 1.0 + strain

            if include_conditions:
                if normalize_conditions:
                    C_feature = C_value / 100.0
                    T_feature = (T_value - 273.15) / 100.0
                    variable_names = ["lambda", "C100", "T100"]
                else:
                    C_feature = C_value
                    T_feature = T_value
                    variable_names = ["lambda", "C", "T"]

                X_file = np.column_stack(
                    [
                        lam,
                        np.full_like(lam, C_feature),
                        np.full_like(lam, T_feature),
                    ]
                )
            else:
                X_file = lam.reshape(-1, 1)
                variable_names = ["lambda"]

            X_list.append(X_file)
            y_list.append(stress.reshape(-1, 1))

            curve_info.append(
                {
                    "filename": filename,
                    "C": C_value,
                    "T": T_value,
                    "n_points": len(lam),
                    "lambda_min": float(np.min(lam)),
                    "lambda_max": float(np.max(lam)),
                    "stress_min": float(np.min(stress)),
                    "stress_max": float(np.max(stress)),
                }
            )

        if len(X_list) == 0:
            raise RuntimeError(f"No valid .xlsx files found in {data_dir}")

        X = np.vstack(X_list)
        y = np.vstack(y_list).ravel()

        return X, y, variable_names, curve_info

    np.random.seed(42)

    X, y, variable_names, curve_info = load_rubber_excel_dataset(
        data_dir=os.path.join(
            project_root, "kd/dataset/Discovery_of_soild_consititutive-main/data/data_rubber/train"
        ),
        include_conditions=True,
        single_file=None,
    )

    print("Loaded rubber data:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("variables:", variable_names)
    print("curve info:", curve_info)

    model = KD_DSCV(
        binary_operators=["add", "sub", "mul", "div"],
        unary_operators=["n2", "n3", "n4"],
        n_samples_per_batch=500,
        n_iterations=100,
        seed=42,
    )

    attach_regression_data_to_model(
        model,
        X=X,
        y=y,
        variable_names=variable_names,
        dataset_name="rubber_C40_313",
        task_type="symbolic_regression",
    )

    step_output = model.train(n_epochs=50, verbose=True)

    print("\nBest result:")
    print(f"Expression: {step_output['expression']}")
    print(f"Reward: {step_output['r']}")

    best_program = step_output["program"]

    print("\nSTRidge expression:")
    print(best_program.str_expression)

    print("\nEvaluation:")
    print(best_program.evaluate)
elif exp_name == "wave_breaking":
    file_path = os.path.join(project_root, "kd/dataset/WaveBreaking.pkl")

    print("PKL path:", file_path)
    print("exists:", os.path.exists(file_path))
    print("isfile:", os.path.isfile(file_path))

    with open(file_path, "rb") as f:
        obj = pickle.load(f)

    key = "L_G2Tp12A080_broad"
    arr = obj[key]

    # arr is a raw [t, x, u] point cloud, not a regular grid, so it must go
    # through point_cloud_to_regular_grid() before build_wave_dataset_for_dscv().
    t_grid, x_grid, u_grid = point_cloud_to_regular_grid(
        arr,
        nx=512,
        x_margin_ratio=0.02,
    )

    dt = float(t_grid[1] - t_grid[0])
    dx = float(x_grid[1] - x_grid[0])

    dataset = build_wave_dataset_for_dscv(u_grid, dx=dx, dt=dt)

    print("dataset['u'] shape:", dataset["u"].shape)
    print("dataset['X'] shape:", dataset["X"][0].shape)
    print("dataset['ut'] shape:", dataset["ut"].shape)
    print("n_input_dim:", dataset["n_input_dim"])
    print("dt:", dt)
    print("dx:", dx)

    print("u min/max:", dataset["u"].min(), dataset["u"].max())
    print("ut min/max:", dataset["ut"].min(), dataset["ut"].max())

    class DictData:
        def __init__(self, dataset):
            self.dataset = dataset

        def get_data(self):
            return self.dataset

    # Build the discovery model.
    model = KD_DSCV(
        binary_operators=["add", "sub", "mul"],
        unary_operators=["n2"],
        n_samples_per_batch=1000,
        n_iterations=50,
        seed=42,
    )

    model.data_class = DictData(dataset)

    model.dataset = "wave_breaking_L_G2Tp12A080"
    model.config_task["task_type"] = "pde"
    model.config_task["dataset"] = "wave_breaking_L_G2Tp12A080"
    model.config_task["metric"] = "pde_reward"
    model.config_task["metric_params"] = [0.01]

    # Initialize the model and inspect the prepared task data.
    model.setup()

    from kd.model.discover.program import Program

    print("task u shape:", Program.task.u[0].shape)
    print("task x shape:", Program.task.x[0].shape)
    print("task ut shape:", Program.task.ut.shape)
    print("n_input_var:", Program.task.n_input_var)

    # Run a short training job for this example.
    step_output = model.train(n_epochs=10, verbose=True)

    print(step_output)
