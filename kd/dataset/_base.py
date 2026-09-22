"""
Base IO code for all datasets
"""

import ast
import csv
import itertools
import logging
import os
import pickle
import re
import zlib
from abc import ABC, abstractmethod
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.interpolate import interp2d

from ._info import DatasetInfo

DATA_MODULE = "kd.dataset.data"
GAMMA = 0.57721566490153286060651209008240243104215933593992


def _harmonic(x1):
    if all(val.is_integer() for val in x1):
        return np.array(
            [sum(1 / d for d in range(1, int(val) + 1)) for val in x1], dtype=np.float32
        )
    else:
        return GAMMA + np.log(x1) + 0.5 / x1 - 1.0 / (12 * x1**2) + 1.0 / (120 * x1**4)


_function_map = {
    "pi": np.pi,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "div": np.divide,
    "harmonic": _harmonic,
}


def _convert_data_dataframe(data, target, feature_names, target_names, sparse_data=False):
    # If the data is not sparse, create a regular DataFrame for features.
    if not sparse_data:
        data_df = pd.DataFrame(data, columns=feature_names, copy=False)
    else:
        # If the data is sparse, create a sparse DataFrame for features.
        data_df = pd.DataFrame.sparse.from_spmatrix(data, columns=feature_names)

    # Create a DataFrame for the target variable with appropriate column names.
    target_df = pd.DataFrame(target, columns=target_names)

    # Concatenate the data and target DataFrames along columns (axis=1) to create a combined DataFrame.
    combined_df = pd.concat([data_df, target_df], axis=1)

    # Separate the feature columns (X) and the target columns (y) from the combined DataFrame.
    X = combined_df[feature_names]
    y = combined_df[target_names]

    # If there is only one target variable (i.e., y has only one column), simplify y to a 1D Series.
    if y.shape[1] == 1:
        y = y.iloc[:, 0]

    # Return the combined DataFrame, features (X), and target (y).
    return combined_df, X, y


def load_csv_data(
    data_file_path: str, encoding: str = "utf-8", has_header: bool = True
) -> np.ndarray:
    """
    Reads a CSV file and returns the data as a NumPy array, automatically determining whether there is a header.

    Depending on the value of `has_header`, the function either skips the first row (if it is a header)
    or includes it as data.

    Args:
        data_file_path (str): The path to the CSV file.
        encoding (str): The file encoding, default is 'utf-8'.
        has_header (bool): Whether the CSV file contains a header row, default is True.

    Returns:
        np.ndarray: A NumPy array containing the data. If `has_header` is True, the first row will be excluded.

    Example:
        >>> data = load_csv_data('example.csv')
        >>> print(data)
        [[25. 85.]
         [22. 90.]
         [23. 88.]]
    """
    # Create a Path object for the file path
    data_path = Path(data_file_path)

    # Open the CSV file and read the data
    with data_path.open("r", encoding=encoding) as f:
        data = csv.reader(f)

        # Read all rows from the CSV
        rows = list(data)

        if has_header:
            # Exclude the header row from the numeric data.
            data_rows = rows[1:]
        else:
            # If no header, use all rows as data
            data_rows = rows

        # Convert the data into a NumPy array
        data_array = np.array(
            data_rows, dtype=np.float32
        )  # Assuming the data is numeric; handle further conversions if needed

    return data_array


def load_mat_file(file_path: str) -> Dict[str, Any]:
    """
    Parses a .mat file (MATLAB format) and returns its content as a Python dictionary.
    Supports both older .mat files (MATLAB 5) and newer ones (MATLAB 7.3 or HDF5 format).

    Args:
        file_path (str): The path to the .mat file to be loaded.

    Returns:
        Dict[str, Any]: A dictionary where keys are variable names and values are corresponding data arrays.

    Raises:
        ValueError: If the file format is not supported or if there's an error in reading the file.
    """
    data_path = Path(file_path)
    # Attempt to load the file as a standard .mat (MATLAB 5) file using scipy.io
    try:
        # Try loading with scipy (for MATLAB version 5 and below)
        mat_data = sio.loadmat(data_path)
        # Remove MATLAB-specific metadata (keys like __header__, __version__, __globals__)
        mat_data_clean = {key: value for key, value in mat_data.items() if not key.startswith("__")}
        return mat_data_clean
    except NotImplementedError:
        # This error will occur if scipy cannot handle the file (e.g., MATLAB version > 7.3)
        raise ValueError(
            "The .mat file is of an unsupported format (likely version 7.3 or higher)."
        )


def load_numpy_data(file_path: str) -> Union[np.ndarray, dict]:
    """
    Loads NumPy data from a `.npy` or `.npz` file and returns it as a NumPy array or a dictionary (for `.npz` files).

    Args:
        file_path (str): The path to the `.npy` or `.npz` file.

    Returns:
        np.ndarray or dict: If the file is a `.npy` file, a NumPy array is returned.
                             If the file is a `.npz` file, a dictionary of arrays is returned.

    Example:
        >>> data = load_numpy_data('data.npy')
        >>> print(data)
        [1. 2. 3. 4.]

        >>> data = load_numpy_data('data.npz')
        >>> print(data['arr_0'])
        [1. 2. 3. 4.]
    """
    if file_path.endswith(".npy"):
        # Load a single NumPy array from a .npy file
        return np.load(file_path)
    elif file_path.endswith(".npz"):
        # Load a NumPy compressed archive (.npz) and return as a dictionary of arrays
        return np.load(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


_TIME_COL_RE = re.compile(
    r"""^\s*(?P<var>.*?)\s*@\s*t\s*=\s*(?P<t>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)\s*$"""
)

_VAR_CLEAN_RE = re.compile(r"\s+")


def _clean_var_name(s: str) -> str:
    s = s.strip()
    s = _VAR_CLEAN_RE.sub(" ", s)
    return s


def load_csv_tlc(
    csv_path: str,
    *,
    equation_name: str = "unknown",
    spatial_vars: Optional[List[str]] = None,
    time_var: str = "t",
    domain: Optional[Dict[str, Tuple[float, float]]] = None,
    epi: float = 0.0,
    return_dataset: bool = False,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
    force_3d_usol: bool = True,
    encoding: Optional[str] = None,
) -> Union[Dict[str, Any], "ScatterPDEDataset"]:
    """Load time-stamped scattered PDE observations from a CSV file."""
    df = pd.read_csv(csv_path, encoding=encoding)

    if df.shape[1] < 2:
        raise ValueError(f"CSV 列数过少：{df.shape[1]}，至少应包含坐标列 + 数据列。")

    cols = list(df.columns)

    # Split coordinate columns from observations at the first time-stamped column.
    data_start = None
    parsed = []  # list of (col_name, var_name, t_value)
    for i, c in enumerate(cols):
        m = _TIME_COL_RE.match(str(c))
        if m:
            data_start = i
            break

    if data_start is None:
        raise ValueError(
            "未找到任何形如 '... @ t=...' 的数据列名。请检查表头格式，例如：'u (1) @ t=0.1'。"
        )

    coord_cols = cols[:data_start]
    data_cols = cols[data_start:]

    if len(coord_cols) == 0:
        raise ValueError(
            "未检测到坐标列（前 n 列）。请确认 CSV 前几列是坐标，并且后续列名含 '@ t='。"
        )

    # Parse each observation column into a state-variable and time pair.
    for c in data_cols:
        mm = _TIME_COL_RE.match(str(c))
        if not mm:
            raise ValueError(
                f"数据区列名不符合 '... @ t=...' 格式：{c!r}。"
                f"（提示：坐标列必须全部在前面，数据列必须全部带 '@ t='）"
            )
        var = _clean_var_name(mm.group("var"))
        t_val = float(mm.group("t"))
        parsed.append((c, var, t_val))

    # Store the leading columns as point coordinates.
    points = df[coord_cols].to_numpy(dtype=float)
    N, d = points.shape

    # Default spatial variable names to the coordinate column labels.
    if spatial_vars is None:
        spatial_vars = [str(c) for c in coord_cols]
    else:
        if len(spatial_vars) != d:
            raise ValueError(f"spatial_vars 长度 {len(spatial_vars)} 必须等于坐标维度 d={d}")

    # Preserve state and time order from their first appearance.
    var_names = sorted({v for _, v, _ in parsed})
    t_values = sorted({t for _, _, t in parsed})

    n_state = len(var_names)
    T = len(t_values)

    # Require exactly one column for every time and state pair.
    # The mapping also defines the final tensor column order.
    mapping: Dict[Tuple[float, str], str] = {}
    for col, var, t in parsed:
        key = (t, var)
        if key in mapping:
            raise ValueError(
                f"检测到重复列：t={t}, var={var}，列名至少重复两次：{mapping[key]!r} 和 {col!r}"
            )
        mapping[key] = col

    missing = []
    for t in t_values:
        for v in var_names:
            if (t, v) not in mapping:
                missing.append((t, v))
    if missing:
        # Limit diagnostics so malformed wide files remain readable.
        preview = ", ".join([f"(t={tv}, var={vv})" for tv, vv in missing[:8]])
        raise ValueError(f"缺少某些 (t,var) 列，示例：{preview}（共缺 {len(missing)} 个）")

    # Assemble usol with shape (n_points, n_times, n_states).
    usol = np.empty((N, T, n_state), dtype=float)
    for ti, t in enumerate(t_values):
        for si, v in enumerate(var_names):
            col = mapping[(t, v)]
            usol[:, ti, si] = df[col].to_numpy(dtype=float)

    t = np.asarray(t_values, dtype=float)

    # Optionally squeeze a single state to shape (n_points, n_times).
    if (not force_3d_usol) and n_state == 1:
        usol_out: np.ndarray = usol[:, :, 0]
    else:
        usol_out = usol

    payload: Dict[str, Any] = {
        "equation_name": equation_name,
        "points": points,
        "t": t,
        "usol": usol_out,
        "spatial_vars": spatial_vars,
        "time_var": time_var,
        "state_vars": var_names,
        "domain": domain,
        "epi": float(epi),
    }

    if not return_dataset:
        return payload

    # Resolve the dataset class at call time to avoid definition-order issues.
    if dataset_kwargs is None:
        dataset_kwargs = {}

    return ScatterPDEDataset(
        equation_name=equation_name,
        points=points,
        t=t,
        usol=usol_out,
        domain=domain,
        epi=float(epi),
        spatial_vars=spatial_vars,
        time_var=time_var,
        **dataset_kwargs,
    )


def load_wake_equation(data_dir: str, file_names: List[str]):
    U = []
    shape = None
    for file_name in file_names:
        data_path = os.path.join(data_dir, file_name)
        data = np.load(data_path).transpose([1, 2, 0])  # shape: (nt, nx, ny)
        if shape is None:
            shape = data.shape
        else:
            assert (
                shape == data.shape
            ), "All data files must have the same shape, but got {} and {}".format(
                shape, data.shape
            )
        U.append(data)  # shape: (nt, nx, ny)
    x = np.linspace(-5, 5, U[0].shape[0])
    y = np.linspace(-5, 5, U[0].shape[1])
    t = np.linspace(0, 20, U[0].shape[2])
    coords = {"x": x, "y": y, "t": t}
    usol = np.stack(U, axis=0)  # shape: (2, nx, ny, nt)
    return GridPDEDataset(
        equation_name="Wake",
        pde_data={"coords": coords, "usol": usol},
        domain={"x": (x.min(), x.max()), "y": (y.min(), y.max()), "t": (t.min(), t.max())},
        epi=0.0,
    )


class BaseDataLoader(ABC):
    """
    Abstract base class defining the interface for data loaders.
    """

    @abstractmethod
    def load_data(self):
        pass


class PDEDataLoader(BaseDataLoader):
    def __init__(self, data_dir: str):
        """
        Initializes the data loader.

        :param data_dir: Directory where data files are stored.
        """
        self.data_dir = Path(data_dir)

    def load_data(
        self, equation_name: str = None, file: str = None
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Loads PDE-related data from different file formats (CSV, MAT, NPY, NPZ).

        :param equation_name: The equation name (file name prefix) if file path is not provided.
        :param file: The full file path to load.
        :return: The loaded data as a NumPy array or a dictionary.
        :raises FileNotFoundError: If no matching data file is found.
        :raises ValueError: If the file format is unsupported.
        """
        if file:
            file_path = Path(file)
            if not file_path.exists():
                raise FileNotFoundError(f"Specified file does not exist: {file}")
        elif equation_name:
            file_path = self._find_file(equation_name)
            if file_path is None:
                raise FileNotFoundError(f"No data file found for equation: {equation_name}")
        else:
            raise ValueError("Either 'equation_name' or 'file' must be provided.")

        # Load the file based on its extension
        if file_path.suffix == ".csv":
            return load_csv_data(str(file_path))
        elif file_path.suffix == ".mat":
            return load_mat_file(str(file_path))
        elif file_path.suffix in [".npy", ".npz"]:
            return load_numpy_data(str(file_path))
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def _find_file(self, equation_name: str) -> Union[Path, None]:
        """
        Searches for a file that matches the given equation name in the data directory.

        :param equation_name: The equation name (file name prefix).
        :return: The path to the matching file, or None if no file is found.
        """
        for ext in [".csv", ".mat", ".npy", ".npz"]:
            file_path = self.data_dir / f"{equation_name}{ext}"
            if file_path.exists():
                return file_path
        return None


class MetaBase(type):
    """
    Metaclass to enforce that subclasses implement required methods
    and contain necessary attributes, ensuring a consistent interface.
    """

    required_methods = {"get_data"}
    required_attributes = ("x", "t", "usol")

    def __new__(cls, name, bases, dct):
        """
        Overrides class creation to enforce method and attribute requirements.
        """

        if not isinstance(cls.required_attributes, (set, list, tuple)):
            raise TypeError(f"{name}.required_attributes must be a set, list, or tuple.")

        # Ensure required methods are implemented
        for method in cls.required_methods:
            if method not in dct:
                raise TypeError(f"{name} must implement the method: {method}")

        # Ensure required attributes exist
        for attr in cls.required_attributes:
            if attr not in dct and not any(attr in base.__dict__ for base in bases):
                raise TypeError(f'{name} must contain the attribute: "{attr}"')

        return super().__new__(cls, name, bases, dct)


class MetaData(metaclass=MetaBase):
    """Base class to store metadata of a Partial Differential Equation (PDE) dataset."""

    x = None
    t = None
    usol = None

    def __init__(self, info: Any):
        """
        Initialize the metadata for a PDE dataset.

        :param info: Description of the equation.
        """
        self.info = info

    def __getitem__(self, key: str) -> Any:
        """
        Allow dictionary-like access to attributes.
        """
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Attribute '{key}' not found in {self.__class__.__name__}.")

    @abstractmethod
    def get_data(self) -> Dict[str, Any]:
        """
        Retrieve all PDE dataset information.
        """
        pass


class GridPDEDataset(MetaData):
    """
    A class representing a Partial Differential Equation (PDE) dataset, providing data access
    and analysis functionality.

    Internal storage:
      - self._usol: (n_response, N_x1, ..., N_t) with time last.
    Public-facing:
      - self.usol:
          * legacy=True  -> (N_x1, ..., N_t)   (response dim hidden; requires n_response==1)
          * legacy=False -> (n_response, N_x1, ..., N_t)
    """

    def __init__(
        self,
        equation_name: str,
        pde_data: Optional[Dict[str, Any]],
        domain: Optional[Dict[str, Tuple[float, float]]],
        epi: float,
        x: Optional[np.ndarray] = None,
        t: Optional[np.ndarray] = None,
        usol: Optional[np.ndarray] = None,
        descr: Optional["DatasetInfo"] = None,
        coords: Optional[Dict[str, np.ndarray]] = None,
        time_var: str = "t",
        legacy: bool = False,
    ):
        """
        Initializes the PDE dataset, supporting two input methods:
        1. Providing data through the `pde_data` dictionary.
        2. Directly passing `x`, `t`, and `usol` arrays.

        :param equation_name: Name of the PDE.
        :param descr: Metadata containing information about the PDE.
        :param pde_data: Optional dictionary containing 'x', 't', and 'usol' data.
                        New style supports 'coords' + 'usol'.
        :param x: Optional, spatial coordinate array (legacy).
        :param t: Optional, temporal coordinate array (legacy).
        :param usol: Optional, solution array u(X, t).
        :param domain: Dictionary defining the domain {variable: (min_value, max_value)}.
        :param epi: Additional parameter.
        :param coords: Optional[recommended], dictionary of coordinate arrays for each variable.
        :param time_var: Name of the time variable in coords (default is 't').
        :param legacy: If True, hide n_response dimension from public usol and enforce n_response==1.
        """
        super().__init__(equation_name)

        self.equation_name = equation_name
        self.domain = domain
        self.epi = epi
        self.descr = descr
        self.time_var = time_var
        self.legacy = bool(legacy)

        # ---- Load data into coords + _usol (unified internal representation) ----
        if pde_data is not None:
            if "coords" in pde_data:
                self.coords = {
                    k: np.asarray(v, dtype=float).reshape(-1) for k, v in pde_data["coords"].items()
                }
                self._usol = np.real(np.asarray(pde_data["usol"]))
            else:
                # legacy dict: x,t,usol
                self.coords = {
                    "x": np.asarray(pde_data.get("x"), dtype=float).reshape(-1),
                    self.time_var: np.asarray(pde_data.get("t"), dtype=float).reshape(-1),
                }
                self._usol = np.real(np.asarray(pde_data.get("usol")))
        elif coords is not None and usol is not None:
            self.coords = {k: np.asarray(v, dtype=float).reshape(-1) for k, v in coords.items()}
            self._usol = np.real(np.asarray(usol))
        elif x is not None and t is not None and usol is not None:
            # legacy direct input
            self.coords = {
                "x": np.asarray(x, dtype=float).reshape(-1),
                self.time_var: np.asarray(t, dtype=float).reshape(-1),
            }
            self._usol = np.real(np.asarray(usol))
        else:
            raise ValueError(
                "Provide either `pde_data`, or (`coords` & `usol`), or (`x`,`t`,`usol`)."
            )

        # ---- Validate coords ----
        if self.time_var not in self.coords:
            raise ValueError(f"coords must contain time variable '{self.time_var}'")

        self._vars: List[str] = list(self.coords.keys())
        # stable ordering: spatial vars in insertion order, then time last
        self._spatial_vars: List[str] = [v for v in self._vars if v != self.time_var]
        self._ordered_vars: List[str] = self._spatial_vars + [self.time_var]

        # ---- Normalize _usol to (n_response, *field_shape) ----
        expected_field_shape = tuple(len(self.coords[v]) for v in self._ordered_vars)

        if self._usol.shape == expected_field_shape:
            # backward compatible scalar response
            self._usol = self._usol[np.newaxis, ...]
        elif (
            self._usol.ndim == len(expected_field_shape) + 1
            and self._usol.shape[1:] == expected_field_shape
        ):
            pass
        else:
            raise ValueError(
                f"usol shape {self._usol.shape} does not match coords lengths {expected_field_shape} "
                f"or (n_response, {expected_field_shape}) for vars {self._ordered_vars} (time last)."
            )

        self.n_response: int = int(self._usol.shape[0])

        if self.legacy and self.n_response != 1:
            raise ValueError(f"legacy=True requires n_response==1, got {self.n_response}")

        # legacy alias (public-facing)
        self.u = self.usol

    # -------------------------
    # Public-facing usol property
    # -------------------------
    @property
    def usol(self) -> np.ndarray:
        """Public usol. In legacy mode, response dimension is hidden."""
        return self._usol[0] if self.legacy else self._usol

    @usol.setter
    def usol(self, value: np.ndarray) -> None:
        """Set usol; stores internally as _usol with response dim."""
        arr = np.real(np.asarray(value))
        expected_field_shape = tuple(len(self.coords[v]) for v in self._ordered_vars)

        if arr.shape == expected_field_shape:
            arr = arr[np.newaxis, ...]
        elif arr.ndim == len(expected_field_shape) + 1 and arr.shape[1:] == expected_field_shape:
            pass
        else:
            raise ValueError(
                f"usol shape {arr.shape} does not match expected {expected_field_shape} "
                f"or (n_response, {expected_field_shape})."
            )

        self._usol = arr
        self.n_response = int(self._usol.shape[0])

        if self.legacy and self.n_response != 1:
            raise ValueError(f"legacy=True requires n_response==1, got {self.n_response}")

        self.u = self.usol

    # -------------------------
    # Coords helpers
    # -------------------------
    @property
    def coords_spatial(self) -> Dict[str, np.ndarray]:
        return {k: self.coords[k] for k in self._spatial_vars}

    # For better clarity, we provide coords_time as well
    @property
    def coords_time(self) -> np.ndarray:
        return self.coords[self.time_var]

    @property
    def t(self) -> np.ndarray:
        return self.coords[self.time_var]

    @property
    def x(self) -> np.ndarray:
        # If x is multidimensional, return the first spatial variable
        if len(self._spatial_vars) == 0:
            raise AttributeError("No spatial variables found in coords.")
        return self.coords[self._spatial_vars[0]]

    @property
    def spatial_vars(self) -> List[str]:
        return list(self._spatial_vars)

    @property
    def vars(self) -> List[str]:
        return list(self._ordered_vars)

    # -------------------------
    # Core API
    # -------------------------
    def get_datapoint(self, *ids: int):
        """
        - 1D legacy: get_datapoint(x_id, t_id) -> ((x,t), u_scalar) when legacy=True
        - ND: get_datapoint(i1, i2, ..., it)

        Return:
          - legacy=True  -> (coords_tuple, float)
          - legacy=False -> (coords_tuple, u_vec) where u_vec has shape (n_response,)
        """
        if len(ids) == 1 and isinstance(ids[0], (tuple, list)):
            ids = tuple(ids[0])  # type: ignore

        if len(ids) != len(self._ordered_vars):
            raise ValueError(f"Expected {len(self._ordered_vars)} indices, got {len(ids)}")

        # bounds check
        for k, v in enumerate(self._ordered_vars):
            n = len(self.coords[v])
            if not (0 <= ids[k] < n):
                raise IndexError(f"Index out of range for '{v}': {ids[k]} not in [0, {n})")

        coord_vals = tuple(float(self.coords[v][ids[k]]) for k, v in enumerate(self._ordered_vars))

        if self.legacy:
            u_val = float(self._usol[(0,) + tuple(ids)])
            return coord_vals, u_val

        u_vec = self._usol[(slice(None),) + tuple(ids)]  # (n_response,)
        return coord_vals, np.asarray(u_vec, dtype=float)

    def get_data(self) -> Dict[str, Any]:
        """
        Backward-compatible:
          - legacy=True: returns usol without response dim (old shape)
          - legacy=False: returns usol with response dim

        Always returns coords. Also returns x,t legacy keys when available.
        """
        data = {"coords": self.coords, "usol": self.usol}
        if len(self._spatial_vars) >= 1 and self._spatial_vars[0] == "x":
            data["x"] = self.x
        data["t"] = self.t

        # only expose these when not legacy to avoid surprising old code
        if not self.legacy:
            data["n_response"] = self.n_response

        return data

    def get_size(self) -> Tuple[int, ...]:
        """
        legacy=True  -> (N_x1, ..., N_t)
        legacy=False -> (n_response, N_x1, ..., N_t)
        """
        return self.usol.shape

    def mesh(self, indexing: str = "ij") -> np.ndarray:
        """
        ND mesh: returns array of shape (prod(N_dims), n_dims) over coords only.
        """
        grids = np.meshgrid(*(self.coords[v] for v in self._ordered_vars), indexing=indexing)
        flat = [g.reshape(-1) for g in grids]
        return np.stack(flat, axis=1)

    def mesh_bounds(self, indexing: str = "ij") -> Tuple[np.ndarray, np.ndarray]:
        m = self.mesh(indexing=indexing)
        return m.min(0), m.max(0)

    def get_boundaries(self) -> Dict[str, Tuple[float, float]]:
        return {
            v: (float(self.coords[v].min()), float(self.coords[v].max()))
            for v in self._ordered_vars
        }

    def get_domain(self) -> Optional[Dict[str, Tuple[float, float]]]:
        return self.domain

    def get_derivative(self, axis: str = "x") -> np.ndarray:
        """
        Legacy: axis in {'x','t'}.
        ND: axis can be any variable name in coords (e.g. 'x','y','z','t').

        Returns:
          - legacy=True  -> same shape as field (no response dim)
          - legacy=False -> same shape as _usol (with response dim)
        """
        if axis == "t":
            axis = self.time_var

        if axis not in self._ordered_vars:
            raise ValueError(f"Invalid axis '{axis}'. Available: {self._ordered_vars}")

        ax = 1 + self._ordered_vars.index(axis)  # +1 because _usol has leading response dim
        grad = np.gradient(self._usol, axis=ax)  # (n_response, ...)
        return grad[0] if self.legacy else grad

    def get_range(
        self,
        x_range: Optional[Tuple[float, float]] = None,
        t_range: Optional[Tuple[float, float]] = None,
        ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Backward compatible:
          - legacy 1D: get_range(x_range=(..), t_range=(..))
        ND:
          - get_range(ranges={'x':(..), 'y':(..), 't':(..)})

        Output:
          - legacy=True: usol returned without response dim
          - legacy=False: usol returned with response dim
        """
        if ranges is None:
            if x_range is None or t_range is None:
                raise ValueError("Provide either `ranges` or both `x_range` and `t_range`.")
            if len(self._spatial_vars) != 1:
                raise ValueError("Legacy (x_range,t_range) only works for 1D spatial datasets.")
            ranges = {self._spatial_vars[0]: x_range, self.time_var: t_range}

        slicers = []
        out_coords = {}
        for v in self._ordered_vars:
            if v not in ranges:
                slicers.append(slice(None))
                out_coords[v] = self.coords[v]
                continue
            lo, hi = ranges[v]
            arr = self.coords[v]
            i0, i1 = np.searchsorted(arr, [lo, hi])
            slicers.append(slice(i0, i1))
            out_coords[v] = arr[i0:i1]

        sub = self._usol[(slice(None),) + tuple(slicers)]  # (n_response, ...)
        sub_out = sub[0] if self.legacy else sub

        result: Dict[str, Any] = {"coords": out_coords, "usol": sub_out}

        if len(self._spatial_vars) == 1:
            result["x"] = out_coords[self._spatial_vars[0]]
            result["t"] = out_coords[self.time_var]

        if not self.legacy:
            result["n_response"] = self.n_response

        return result

    def sample(
        self, n_samples: Union[int, float], method: str = "random"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          - sampled_points: (n, n_dims) over coords (no response dim)
          - sampled_usol:
              legacy=True  -> (n, 1)
              legacy=False -> (n, n_response)

        Methods:
          - random, uniform: any ND
          - spline: only legacy 1D (x,t) and n_response==1
        """
        field_shape = self._usol.shape[1:]  # (*dims)
        total_points = int(np.prod(field_shape))
        n_dims = len(field_shape)

        if isinstance(n_samples, float) and 0 < n_samples < 1:
            n_samples = int(total_points * n_samples)

        n_samples = int(n_samples)

        if n_samples > total_points:
            raise ValueError(
                f"Requested {n_samples} samples, but only {total_points} points available."
            )

        if method == "random":
            flat_idx = np.random.choice(total_points, n_samples, replace=False)
            multi_idx = np.unravel_index(flat_idx, field_shape)  # tuple length n_dims

            pts_cols = []
            for dim, v in enumerate(self._ordered_vars):
                pts_cols.append(self.coords[v][multi_idx[dim]])
            sampled_points = np.stack(pts_cols, axis=1)

            if self.legacy:
                sampled_u = self._usol[(0,) + multi_idx]  # (n,)
                sampled_usol = sampled_u.reshape(-1, 1)  # (n,1)
            else:
                sampled_u = self._usol[(slice(None),) + multi_idx]  # (n_response, n)
                sampled_usol = np.moveaxis(sampled_u, 0, 1)  # (n, n_response)

            return sampled_points, sampled_usol

        if method == "uniform":
            per_dim = int(round(n_samples ** (1.0 / n_dims)))
            per_dim = max(per_dim, 1)

            dim_indices = []
            for v in self._ordered_vars:
                n = len(self.coords[v])
                dim_indices.append(np.linspace(0, n - 1, per_dim, dtype=int))

            grids = np.meshgrid(*dim_indices, indexing="ij")
            multi_idx = tuple(g.reshape(-1) for g in grids)

            pts_cols = []
            for dim, v in enumerate(self._ordered_vars):
                pts_cols.append(self.coords[v][multi_idx[dim]])
            sampled_points = np.stack(pts_cols, axis=1)

            if self.legacy:
                sampled_u = self._usol[(0,) + multi_idx].reshape(-1, 1)  # (n,1)
                sampled_usol = sampled_u
            else:
                sampled_u = self._usol[(slice(None),) + multi_idx]  # (n_response, n)
                sampled_usol = np.moveaxis(sampled_u, 0, 1)  # (n, n_response)

            if sampled_points.shape[0] > n_samples:
                sampled_points = sampled_points[:n_samples]
                sampled_usol = sampled_usol[:n_samples]

            return sampled_points, sampled_usol

        if method == "spline":
            # Keep behavior consistent with old implementation: only 1D (x,t) and single response
            if len(self._ordered_vars) != 2 or (
                len(self._spatial_vars) == 0 or self._spatial_vars[0] != "x"
            ):
                raise ValueError(
                    "spline sampling is only supported for 1D (x,t) datasets in this implementation."
                )
            if self.n_response != 1:
                raise ValueError("spline sampling currently only supports n_response=1.")
            if not self.legacy:
                # Restrict this adapter to legacy scalar fields to keep shapes unambiguous.
                raise ValueError(
                    "spline sampling is only supported when legacy=True (to keep old behavior)."
                )

            grid_size = int(np.sqrt(n_samples))
            x_new = np.linspace(self.x.min(), self.x.max(), grid_size)
            t_new = np.linspace(self.t.min(), self.t.max(), grid_size)

            spline = interp2d(self.t, self.x, self._usol[0], kind="cubic")
            usol_new = spline(t_new, x_new)

            x_samples, t_samples = np.meshgrid(x_new, t_new)
            sampled_points = np.column_stack((x_samples.flatten(), t_samples.flatten()))
            sampled_usol = np.asarray(usol_new).reshape(-1, 1)  # (n,1)
            return sampled_points, sampled_usol

        raise ValueError(f"Unsupported sampling method: {method}")

    def plot_solution(self) -> None:
        """
        Heatmap visualization for 1D spatial datasets:
          - legacy=True: expects usol is 2D (Nx, Nt)
          - legacy=False: only supported when n_response==1 and usol is (1, Nx, Nt)
        """
        if self.legacy:
            U = self.usol  # (Nx, Nt)
            if U.ndim != 2:
                raise ValueError(
                    "plot_solution is only supported for 1D spatial datasets (2D usol) in legacy mode."
                )
            plt.figure(figsize=(8, 6))
            plt.imshow(
                U,
                aspect="auto",
                origin="lower",
                extent=[self.t.min(), self.t.max(), self.x.min(), self.x.max()],
            )
            plt.colorbar(label="Solution u(x, t)")
            plt.xlabel("Time (t)")
            plt.ylabel("Space (x)")
            plt.title(f"Solution of {self.equation_name}")
            plt.show()
            return

        # non-legacy
        if self.n_response != 1:
            raise ValueError("plot_solution only supports n_response=1 when legacy=False.")
        if self._usol.ndim != 3:
            raise ValueError(
                "plot_solution is only supported for 1D spatial datasets (usol shape (1, Nx, Nt))."
            )

        plt.figure(figsize=(8, 6))
        plt.imshow(
            self._usol[0],
            aspect="auto",
            origin="lower",
            extent=[self.t.min(), self.t.max(), self.x.min(), self.x.max()],
        )
        plt.colorbar(label="Solution u(x, t)")
        plt.xlabel("Time (t)")
        plt.ylabel("Space (x)")
        plt.title(f"Solution of {self.equation_name}")
        plt.show()

    def __repr__(self) -> str:
        return (
            f"GridPDEDataset(equation='{self.equation_name}', legacy={self.legacy}, "
            f"n_response={self.n_response}, size={self.get_size()}, "
            f"boundaries={self.get_boundaries()})"
        )


class ScatterPDEDataset(MetaData):
    """
    Scatter (unstructured) PDE dataset.

    Internal representation:
      - points: (N, d) spatial coordinates (scattered)
      - t:      (T,) time coordinates
      - usol:   (N, T) or (N, T, n_state)

    Notes:
      - Unlike GridPDEDataset, scattered points do NOT form separable coordinate axes.
      - Therefore, no `coords` dict of per-axis 1D arrays is used.
    """

    def __init__(
        self,
        equation_name: str,
        points: np.ndarray,
        t: np.ndarray,
        usol: np.ndarray,
        *,
        domain: Optional[Dict[str, Tuple[float, float]]] = None,
        epi: float = 0.0,
        descr: Optional["DatasetInfo"] = None,
        spatial_vars: Optional[List[str]] = None,  # e.g. ["x","y"] or ["x","y","z"]
        time_var: str = "t",
        point_ids: Optional[np.ndarray] = None,  # optional IDs, shape (N,)
    ):
        super().__init__(equation_name)

        self.equation_name = equation_name
        self.domain = domain
        self.epi = float(epi)
        self.descr = descr

        self.time_var = str(time_var)
        self._spatial_vars = list(spatial_vars) if spatial_vars is not None else None

        self.points = np.asarray(points, dtype=float)
        self._t = np.asarray(t, dtype=float).reshape(-1)
        self.usol = np.real(np.asarray(usol))

        self.point_ids = None if point_ids is None else np.asarray(point_ids)

        # ---- Validate shapes ----
        if self.points.ndim != 2:
            raise ValueError(f"`points` must be 2D array of shape (N, d). Got {self.points.shape}")

        N, d = self.points.shape
        T = len(self._t)

        if self.usol.ndim == 2:
            expected = (N, T)
            if self.usol.shape != expected:
                raise ValueError(f"`usol` shape {self.usol.shape} != {expected} for (N,T)")
            self._n_state = 1
        elif self.usol.ndim == 3:
            expected = (N, T, self.usol.shape[2])
            if self.usol.shape[0] != N or self.usol.shape[1] != T:
                raise ValueError(
                    f"`usol` first two dims must be (N,T)=({N},{T}), got {self.usol.shape}"
                )
            self._n_state = int(self.usol.shape[2])
        else:
            raise ValueError("`usol` must be 2D (N,T) or 3D (N,T,n_state).")

        if self.point_ids is not None:
            if self.point_ids.shape != (N,):
                raise ValueError(f"`point_ids` must have shape (N,), got {self.point_ids.shape}")

        # stable variable names
        if self._spatial_vars is None:
            # default names: x0, x1, ...
            self._spatial_vars = [f"x{j}" for j in range(d)]
        else:
            if len(self._spatial_vars) != d:
                raise ValueError(
                    f"`spatial_vars` length {len(self._spatial_vars)} must match points dim d={d}"
                )

        # Keep the GridPDEDataset-style alias for compatibility.
        self.u = self.usol

    # -----------------
    # Properties
    # -----------------
    @property
    def t(self) -> np.ndarray:
        return self._t

    @property
    def spatial_dim(self) -> int:
        return int(self.points.shape[1])

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_times(self) -> int:
        return int(self._t.shape[0])

    @property
    def n_state(self) -> int:
        return int(self._n_state)

    @property
    def spatial_vars(self) -> List[str]:
        return list(self._spatial_vars)

    @property
    def vars(self) -> List[str]:
        # For scattered data, vars are semantic labels rather than meshable axes.
        return self.spatial_vars + [self.time_var]

    # -----------------
    # Core accessors
    # -----------------
    def get_datapoint(
        self, point_id: int, t_id: int, state: int = 0
    ) -> Tuple[Tuple[float, ...], float]:
        """
        Returns:
          - coords_tuple: (x1, x2, ..., xd, t)
          - u_value: float
        """
        if not (0 <= point_id < self.n_points):
            raise IndexError(f"point_id out of range: {point_id} not in [0,{self.n_points})")
        if not (0 <= t_id < self.n_times):
            raise IndexError(f"t_id out of range: {t_id} not in [0,{self.n_times})")
        if not (0 <= state < self.n_state):
            raise IndexError(f"state out of range: {state} not in [0,{self.n_state})")

        coords = tuple(float(v) for v in self.points[point_id]) + (float(self._t[t_id]),)

        if self.usol.ndim == 2:
            u_val = float(self.usol[point_id, t_id])
        else:
            u_val = float(self.usol[point_id, t_id, state])

        return coords, u_val

    def get_data(self) -> Dict[str, Any]:
        """
        New-style scattered data dict.
        """
        return {
            "points": self.points,  # (N, d)
            "t": self._t,  # (T,)
            "usol": self.usol,  # (N, T) or (N,T,n_state)
            "spatial_vars": self.spatial_vars,
            "time_var": self.time_var,
            "point_ids": self.point_ids,
        }

    def get_size(self) -> Tuple[int, ...]:
        """
        Returns (N, T) or (N, T, n_state).
        """
        return tuple(self.usol.shape)

    # -----------------
    # Geometry helpers
    # -----------------
    def mesh(self) -> np.ndarray:
        """
        Returns all (x..., t) pairs as a flat table:
          shape: (N*T, d+1)

        Order: time-major within each point:
          rows for point 0 across all t, then point 1, ...
        """
        N, d = self.points.shape
        T = self.n_times

        X_rep = np.repeat(self.points, repeats=T, axis=0)  # (N*T, d)
        t_tile = np.tile(self._t, reps=N).reshape(-1, 1)  # (N*T, 1)
        return np.hstack([X_rep, t_tile])

    def mesh_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        m = self.mesh()
        return m.min(axis=0), m.max(axis=0)

    def get_boundaries(self) -> Dict[str, Tuple[float, float]]:
        """
        Returns bounding box for each spatial variable + time.
        """
        bounds: Dict[str, Tuple[float, float]] = {}
        for j, name in enumerate(self._spatial_vars):
            col = self.points[:, j]
            bounds[name] = (float(col.min()), float(col.max()))
        bounds[self.time_var] = (float(self._t.min()), float(self._t.max()))
        return bounds

    def get_domain(self) -> Optional[Dict[str, Tuple[float, float]]]:
        return self.domain

    # -----------------
    # Derivatives(To be Implemented)
    # -----------------
    def get_derivative(self, axis: str = "t", state: int = 0) -> np.ndarray:
        raise NotImplementedError("get_derivative not implemented for ScatterPDEDataset.")

    # -----------------
    # Subsetting & sampling
    # -----------------
    def get_range(
        self,
        *,
        spatial_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        t_range: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """
        Subset by spatial bounding box + time interval.

        spatial_ranges example:
          {"x": (0,1), "y": (-1,1)}  or {"x0":(...), "x1":(...)} depending on spatial_vars

        Returns a dict (not a new dataset instance) for flexibility.
        """
        N, d = self.points.shape

        # spatial mask
        mask = np.ones(N, dtype=bool)
        if spatial_ranges is not None:
            for name, (lo, hi) in spatial_ranges.items():
                if name not in self._spatial_vars:
                    raise ValueError(
                        f"Unknown spatial var '{name}'. Available: {self._spatial_vars}"
                    )
                j = self._spatial_vars.index(name)
                col = self.points[:, j]
                mask &= (col >= lo) & (col <= hi)

        sub_points = self.points[mask]
        sub_point_ids = None if self.point_ids is None else self.point_ids[mask]

        # time mask
        tmask = np.ones(self.n_times, dtype=bool)
        if t_range is not None:
            lo, hi = t_range
            tmask = (self._t >= lo) & (self._t <= hi)

        sub_t = self._t[tmask]

        if self.usol.ndim == 2:
            sub_usol = self.usol[mask][:, tmask]
        else:
            sub_usol = self.usol[mask][:, tmask, :]

        return {
            "points": sub_points,
            "t": sub_t,
            "usol": sub_usol,
            "spatial_vars": self.spatial_vars,
            "time_var": self.time_var,
            "point_ids": sub_point_ids,
        }

    def sample(
        self,
        n_samples: Union[int, float],
        *,
        method: str = "random",
        state: int = 0,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          - sampled_points: (n, d+1) columns are [x..., t]
          - sampled_usol:   (n, 1)

        Sampling happens over the joint set of all (point, time) pairs.
        """
        if method != "random":
            raise ValueError(f"Unsupported sampling method: {method} (only 'random' supported).")

        if not (0 <= state < self.n_state):
            raise IndexError(f"state out of range: {state} not in [0,{self.n_state})")

        rng = np.random.default_rng(seed)

        N = self.n_points
        T = self.n_times
        total = N * T

        if isinstance(n_samples, float):
            if not (0 < n_samples < 1):
                raise ValueError("If n_samples is float, it must be in (0,1).")
            n = int(round(total * n_samples))
        else:
            n = int(n_samples)

        if n <= 0:
            raise ValueError("n_samples must be positive.")
        if n > total:
            raise ValueError(f"Requested {n} samples, but only {total} (point,time) pairs exist.")

        flat_idx = rng.choice(total, size=n, replace=False)
        p_idx = flat_idx // T
        t_idx = flat_idx % T

        sampled_xt = np.hstack([self.points[p_idx], self._t[t_idx].reshape(-1, 1)])

        if self.usol.ndim == 2:
            y = self.usol[p_idx, t_idx]
        else:
            y = self.usol[p_idx, t_idx, state]

        return sampled_xt, y.reshape(-1, 1)

    def __repr__(self) -> str:
        return (
            f"ScatterPDEDataset(equation='{self.equation_name}', "
            f"points={self.points.shape}, t={self._t.shape}, usol={self.usol.shape}, "
            f"boundaries={self.get_boundaries()})"
        )


class ODEDataset(MetaData):
    """
    A class representing an Ordinary Differential Equation (ODE) discovery dataset.

    Stores one or more observed trajectories of a dynamical system: state
    variables u(t) sampled over time, optionally paired with per-trajectory
    static parameters (e.g. mass, diameter) and an identifying label.

    Internal representation:
      - self.trajectories: List[Dict] with keys:
          "t"      -> (T_i,) ndarray
          "state"  -> (n_state, T_i) ndarray
          "params" -> Dict[str, float] or None (trajectory-level static covariates)
          "id"     -> str or None

    Unlike GridPDEDataset/ScatterPDEDataset, trajectories are NOT forced onto a
    shared time grid or padded to equal length: real multi-trial ODE data (e.g.
    repeated drop experiments) is commonly sampled at different rates/durations
    per trial.
    """

    def __init__(
        self,
        equation_name: str,
        trajectories: List[Dict[str, Any]],
        state_vars: List[str],
        *,
        time_var: str = "t",
        param_names: Optional[List[str]] = None,
        domain: Optional[Dict[str, Tuple[float, float]]] = None,
        epi: float = 0.0,
        descr: Optional["DatasetInfo"] = None,
    ):
        super().__init__(equation_name)

        self.equation_name = equation_name
        self.time_var = str(time_var)
        self.domain = domain
        self.epi = float(epi)
        self.descr = descr

        self._state_vars: List[str] = list(state_vars)
        n_state = len(self._state_vars)

        if len(trajectories) == 0:
            raise ValueError("`trajectories` must contain at least one trajectory.")

        cleaned: List[Dict[str, Any]] = []
        param_key_set = set()
        for i, traj in enumerate(trajectories):
            t = np.asarray(traj["t"], dtype=float).reshape(-1)
            state = np.asarray(traj["state"], dtype=float)
            if state.ndim == 1:
                state = state.reshape(1, -1)
            if state.shape[0] != n_state:
                raise ValueError(
                    f"Trajectory {i} state has {state.shape[0]} rows, expected n_state={n_state}"
                )
            if state.shape[1] != t.shape[0]:
                raise ValueError(
                    f"Trajectory {i}: state.shape[1]={state.shape[1]} does not match len(t)={t.shape[0]}"
                )
            params = traj.get("params")
            if params is not None:
                param_key_set.update(params.keys())
            cleaned.append(
                {
                    "t": t,
                    "state": state,
                    "params": params,
                    "id": traj.get("id", f"traj{i}"),
                }
            )

        self.trajectories = cleaned
        self._param_names: List[str] = (
            list(param_names) if param_names is not None else sorted(param_key_set)
        )

    # -------------------------
    # Properties
    # -------------------------
    @property
    def n_traj(self) -> int:
        return len(self.trajectories)

    @property
    def n_state(self) -> int:
        return len(self._state_vars)

    @property
    def state_vars(self) -> List[str]:
        return list(self._state_vars)

    @property
    def param_names(self) -> List[str]:
        return list(self._param_names)

    # -------------------------
    # Core accessors
    # -------------------------
    def get_datapoint(
        self, traj_id: int, t_id: int
    ) -> Tuple[float, np.ndarray, Optional[Dict[str, float]]]:
        """Returns (t_value, state_vector, params_dict_or_None) for one trajectory sample."""
        traj = self.trajectories[traj_id]
        t_val = float(traj["t"][t_id])
        state_val = traj["state"][:, t_id].copy()
        return t_val, state_val, traj["params"]

    def get_data(self) -> Dict[str, Any]:
        return {
            "trajectories": self.trajectories,
            "state_vars": self.state_vars,
            "param_names": self.param_names,
            "time_var": self.time_var,
            "equation_name": self.equation_name,
        }

    def get_size(self) -> Tuple[int, int, List[int]]:
        """Returns (n_traj, n_state, [T_i for each trajectory])."""
        return (self.n_traj, self.n_state, [traj["t"].shape[0] for traj in self.trajectories])

    def get_domain(self) -> Optional[Dict[str, Tuple[float, float]]]:
        return self.domain

    # -------------------------
    # Derivatives
    # -------------------------
    def get_derivative(self, order: int = 1) -> List[np.ndarray]:
        """
        Returns a list (length n_traj) of (n_state, T_i) arrays: the `order`-th
        time derivative of each trajectory's state, computed with np.gradient
        (supports non-uniform time steps, which real multi-trial data often has).
        """
        if order < 1:
            raise ValueError("order must be >= 1")

        derivs = []
        for traj in self.trajectories:
            d = traj["state"]
            for _ in range(order):
                d = np.gradient(d, traj["t"], axis=1)
            derivs.append(d)
        return derivs

    # -------------------------
    # Bridge to (X, y) regression arrays
    # -------------------------
    def to_regression_arrays(
        self,
        target_state: Optional[Union[str, List[str]]] = None,
        order: int = 1,
        include_params: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Flatten all (trajectory, time) samples into rows, producing a generic
        (X, y) regression dataset:

            X columns: state variables (+ static params if include_params and
                       available), one row per (trajectory, time) sample.
            y columns: the `order`-th time derivative of `target_state`
                       (defaults to all state vars).

        This lets ODE trajectory data be consumed directly by the existing
        `SymbolicRegressionTask` (which only needs a generic {"X":..., "y":...}
        dict), without requiring a dedicated ODE task.

        Returns
        -------
        X : (N, n_state [+ n_params]) ndarray
        y : (N, n_targets) ndarray
        variable_names : names of X columns, in order
        """
        if target_state is None:
            target_idx = list(range(self.n_state))
        else:
            names = [target_state] if isinstance(target_state, str) else list(target_state)
            target_idx = [self._state_vars.index(n) for n in names]

        derivs = self.get_derivative(order=order)

        use_params = include_params and len(self._param_names) > 0

        X_rows = []
        y_rows = []
        for traj, d in zip(self.trajectories, derivs):
            state = traj["state"]  # (n_state, T)
            T = state.shape[1]
            state_cols = state.T  # (T, n_state)
            if use_params:
                params = traj["params"] or {}
                param_cols = np.array([[params.get(p, np.nan) for p in self._param_names]] * T)
                cols = np.hstack([state_cols, param_cols])
            else:
                cols = state_cols
            X_rows.append(cols)
            y_rows.append(d[target_idx, :].T)  # (T, n_targets)

        X = np.vstack(X_rows)
        y = np.vstack(y_rows)

        variable_names = list(self._state_vars)
        if use_params:
            variable_names += list(self._param_names)

        return X, y, variable_names

    # -------------------------
    # Sampling
    # -------------------------
    def sample(
        self,
        n_samples: Union[int, float],
        *,
        method: str = "random",
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Randomly sample (trajectory, time) pairs across all trajectories.

        Returns
        -------
        sampled_t     : (n,) time values
        sampled_state : (n, n_state) state values
        """
        if method != "random":
            raise ValueError(f"Unsupported sampling method: {method} (only 'random' supported).")

        rng = np.random.default_rng(seed)

        all_traj_idx = []
        all_t_idx = []
        for ti, traj in enumerate(self.trajectories):
            T = traj["t"].shape[0]
            all_traj_idx.append(np.full(T, ti))
            all_t_idx.append(np.arange(T))
        all_traj_idx = np.concatenate(all_traj_idx)
        all_t_idx = np.concatenate(all_t_idx)

        total = all_traj_idx.shape[0]
        if isinstance(n_samples, float):
            if not (0 < n_samples < 1):
                raise ValueError("If n_samples is float, it must be in (0,1).")
            n = int(round(total * n_samples))
        else:
            n = int(n_samples)

        if n <= 0:
            raise ValueError("n_samples must be positive.")
        if n > total:
            raise ValueError(
                f"Requested {n} samples, but only {total} (trajectory,time) pairs exist."
            )

        flat_idx = rng.choice(total, size=n, replace=False)
        traj_sel = all_traj_idx[flat_idx]
        t_sel = all_t_idx[flat_idx]

        sampled_t = np.array([self.trajectories[ti]["t"][tid] for ti, tid in zip(traj_sel, t_sel)])
        sampled_state = np.array(
            [self.trajectories[ti]["state"][:, tid] for ti, tid in zip(traj_sel, t_sel)]
        )

        return sampled_t, sampled_state

    def __repr__(self) -> str:
        sizes = [traj["t"].shape[0] for traj in self.trajectories]
        return (
            f"ODEDataset(equation='{self.equation_name}', n_traj={self.n_traj}, "
            f"state_vars={self.state_vars}, param_names={self.param_names}, "
            f"traj_lengths={sizes})"
        )


class SymbolicRegressionDataset(MetaData):
    """
    Class used to generate (X, y) data from a named benchmark expression.

    Parameters
    ----------
    name : str
        Name of benchmark expression.

    benchmark_source : str, optional
        Filename of CSV describing benchmark expressions.

    root : str, optional
        Directory containing benchmark_source and function_sets.csv.

    noise : float, optional
        If not None, Gaussian noise is added to the y values with standard
        deviation = noise * RMS of the noiseless y training values.

    seed : int, optional
        Random number seed used to generate data. Checksum on name is added to
        seed.

    logdir : str, optional
        Directory where experiment logfiles are saved.

    backup : bool, optional
        Save generated dataset in logdir if logdir is provided.
    """

    def __init__(
        self,
        name,
        benchmark_source="benchmarks.csv",
        root=None,
        noise=0.0,
        seed=0,
        logdir=None,
        backup=False,
    ):
        # Set class variables
        super().__init__(name)
        self.name = name
        self.seed = seed
        self.noise = noise if noise is not None else 0.0

        # Set random number generator used for sampling X values
        seed += zlib.adler32(
            name.encode("utf-8")
        )  # Different seed for each name, otherwise two benchmarks with the same domain will always have the same X values
        self.rng = np.random.RandomState(seed)

        # Load benchmark data
        if root is None:
            root = resources.files(DATA_MODULE)
        benchmark_path = os.path.join(root, benchmark_source)
        benchmark_df = pd.read_csv(benchmark_path, index_col=0, encoding="ISO-8859-1")
        row = benchmark_df.loc[name]
        self.n_input_var = row["variables"]

        # Create symbolic expression
        self.numpy_expr = self.make_numpy_expr(row["expression"])

        # Get dataset specifications
        self.train_spec = self.extract_dataset_specs(row["train_spec"])
        self.test_spec = self.extract_dataset_specs(row["test_spec"])
        if self.test_spec is None:
            self.test_spec = self.train_spec

        # Create X, y values - Train set
        self.X_train, self.y_train = self.build_dataset(self.train_spec)
        self.y_train_noiseless = self.y_train.copy()

        # Create X, y values - Test set
        self.X_test, self.y_test = self.build_dataset(self.test_spec)
        self.y_test_noiseless = self.y_test.copy()

        # Add Gaussian noise
        if self.noise > 0:
            y_rms = np.sqrt(np.mean(self.y_train**2))
            scale = self.noise * y_rms
            self.y_train += self.rng.normal(loc=0, scale=scale, size=self.y_train.shape)
            self.y_test += self.rng.normal(loc=0, scale=scale, size=self.y_test.shape)
        elif self.noise < 0:
            print("WARNING: Ignoring negative noise value: {}".format(self.noise))

        # Load default function set
        function_set_path = os.path.join(root, "function_sets.csv")
        function_set_df = pd.read_csv(function_set_path, index_col=0)
        function_set_name = row["function_set"]
        self.function_set = function_set_df.loc[function_set_name].tolist()[0].strip().split(",")

        # Prepare status output
        output_message = "\n-- BUILDING DATASET START -----------\n"
        output_message += "Generated data for benchmark   : {}\n".format(name)
        output_message += "Benchmark path                 : {}\n".format(benchmark_path)
        output_message += "Function set                   : {} --> {}\n".format(
            function_set_name, self.function_set
        )
        output_message += "Function set path              : {}\n".format(function_set_path)
        test_spec_txt = (
            row["test_spec"]
            if row["test_spec"] != "None"
            else "{} (Copy from train!)".format(row["test_spec"])
        )
        output_message += (
            "Dataset specifications         : \n"
            + "        Train --> {}\n".format(row["train_spec"])
            + "        Test  --> {}\n".format(test_spec_txt)
        )
        random_choice_train = self.rng.randint(self.X_train.shape[0])
        random_sample_train = "[{}],[{}]".format(
            self.X_train[random_choice_train], self.y_train[random_choice_train]
        )
        output_message += (
            "Built data set                 : \n"
            + "        Train --> X:{}, y:{}, Sample: {}\n".format(
                self.X_train.shape, self.y_train.shape, random_sample_train
            )
        )
        if row["test_spec"] is not None:
            random_choice_test = self.rng.randint(self.X_test.shape[0])
            random_sample_test = "[{}],[{}]".format(
                self.X_test[random_choice_test], self.y_test[random_choice_test]
            )
            output_message += "        Test  --> X:{}, y:{}, Sample: {}\n".format(
                self.X_test.shape, self.y_test.shape, random_sample_test
            )
        if backup and logdir is not None:
            output_message += self.save(logdir)
        output_message += "-- BUILDING DATASET END -------------\n"
        print(output_message)

    def extract_dataset_specs(self, specs):
        if not isinstance(specs, str):
            if np.isnan(specs):
                return None
            else:
                assert False, "Dataset specifications should be a string or None: {}".format(specs)
        specs = ast.literal_eval(specs)
        if specs is not None:
            specs["distribution"] = list(list(specs.items())[0][1].items())[0][0]
            if specs["distribution"] == "E":
                sizes = []
                for i in range(1, self.n_input_var + 1):
                    input_var = "all" if "all" in specs else f"x{i}"
                    if input_var not in specs:
                        input_var = "x1"
                    sizes.append(len(self._equidistant_values(specs[input_var]["E"])))
                specs["dataset_size"] = int(np.prod(sizes))
            else:
                specs["dataset_size"] = list(list(specs.items())[0][1].items())[0][1][2]
        return specs

    @staticmethod
    def _equidistant_values(spec):
        """Build an equidistant axis from ``[start, stop, step_or_count]``."""
        start, stop, step_or_count = spec
        distance = stop - start
        if step_or_count <= 0 or distance < 0:
            raise ValueError(f"Invalid equidistant specification: {spec}")
        if step_or_count > distance:
            count = int(step_or_count)
        else:
            count = int(round(distance / step_or_count)) + 1
        if count < 1:
            raise ValueError(f"Equidistant specification produces no points: {spec}")
        return np.linspace(start=start, stop=stop, num=count, endpoint=True)

    def build_dataset(self, specs, max_iterations=1000, max_repeated_empty=100):
        """This function generates an (X,y) dataset by randomly sampling X
        values in a given range and calculating the corresponding y values.
        During generation it is checked that the generated datapoints are
        valid within the given range, removing nan and inf values. The
        generated dataset will be filled up to the desired dataset size or
        the function terminates with an error."""
        if specs["distribution"] == "E":
            X = self.make_X(specs, specs["dataset_size"])
            y = self.numpy_expr(X)
            X, y = self.remove_invalid(X, y)
            if len(X) == 0:
                raise ValueError(f"Equidistant specification produced no finite samples: {specs}")
            return X, y

        current_size = 0
        X_tmp = None
        y_tmp = None
        X = None
        y = None
        count_repeated_empty = 0
        count_iterations = 0
        while current_size < specs["dataset_size"]:
            if count_iterations > max_iterations:
                assert False, "Dataset creation taking too long. Got {} from {}".format(
                    X_tmp.shape, specs
                )
            missing_value_count = specs["dataset_size"] - current_size
            # Get all X values
            X = self.make_X(specs, missing_value_count)
            assert X.ndim == 2, "Dataset X has wrong dimension: {} != 2".format(X.ndim)
            # Calculate y values
            y = self.numpy_expr(X)
            # Sanity check
            X, y = self.remove_invalid(X, y)
            if y.shape[0] == 0:
                count_repeated_empty += 1
                if count_repeated_empty > max_repeated_empty:
                    assert False, "Dataset cannot be created in the given range: {}".format(specs)
            # Put old and new data together if available
            if X_tmp is not None:
                X = np.append(X, X_tmp, axis=0)
                y = np.append(y, y_tmp, axis=0)
            current_size = X.shape[0]
            X_tmp = X
            y_tmp = y
            count_iterations += 1
        assert X.shape[0] == specs["dataset_size"]
        if X.ndim == 1:
            X = X[:, np.newaxis]
        return X, y

    def get_data(self) -> Dict[str, Any]:
        return {
            "X_train": self.X_train,
            "y_train": self.y_train,
            "X_test": self.X_test,
            "y_test": self.y_test,
            "function_set": self.function_set,
        }

    def remove_invalid(self, X, y, y_limit=None):
        """Remove non-finite targets and optionally enforce a magnitude limit."""
        valid = np.isfinite(y)
        if y_limit is not None:
            valid &= np.logical_and(y > -y_limit, y < y_limit)
        y = y[valid]
        X = X[valid]
        assert X.shape[0] == y.shape[0]
        return X, y

    def make_X(self, spec, size):
        """Creates X values based on provided specification."""

        features = []
        for i in range(1, self.n_input_var + 1):

            # Hierarchy: "all" --> "x{}".format(i)
            input_var = "x{}".format(i)
            if "all" in spec:
                input_var = "all"
            elif input_var not in spec:
                input_var = "x1"

            if "U" in spec[input_var]:
                low, high, n = spec[input_var]["U"]
                feature = self.rng.uniform(low=low, high=high, size=size)
            elif "E" in spec[input_var]:
                feature = self._equidistant_values(spec[input_var]["E"])
            else:
                raise ValueError(
                    "Did not recognize specification for {}: {}.".format(input_var, spec[input_var])
                )
            features.append(feature)

        # Do multivariable combinations
        if "E" in spec[input_var] and self.n_input_var > 1:
            X = np.array(list(itertools.product(*features)))
        else:
            X = np.column_stack(features)

        return X

    def make_numpy_expr(self, s):
        """This isn't pretty, but unlike sympy's lambdify, this ensures we use
        our protected functions. Otherwise, some expressions may have large
        error even if the functional form is correct due to the training set
        not using protected functions."""
        for k in _function_map.keys():
            s = s.replace(k, f"_function_map['{k}']")
        for i in reversed(range(self.n_input_var)):
            s = s.replace(f"x{i + 1}", f"x[:, {i}]")
        # Return numpy expression
        # `s` is built above purely from internal token substitution (never from
        # external/user input), so eval() here only ever runs a controlled
        # arithmetic expression string.
        return lambda x: eval(s)

    def save(self, logdir="./"):
        """Saves the dataset to a specified location."""
        save_path = os.path.join(
            logdir, "data_{}_n{:.2f}_s{}.csv".format(self.name, self.noise, self.seed)
        )
        try:
            os.makedirs(logdir, exist_ok=True)
            np.savetxt(
                save_path,
                np.concatenate(
                    (
                        np.hstack((self.X_train, self.y_train[..., np.newaxis])),
                        np.hstack((self.X_test, self.y_test[..., np.newaxis])),
                    ),
                    axis=0,
                ),
                delimiter=",",
                fmt="%1.5f",
            )
            return "Saved dataset to               : {}\n".format(save_path)
        except Exception as e:
            logging.warning("Could not save dataset: %s", e)

    def plot(self, logdir="./"):
        """Plot Dataset with underlying ground truth."""
        if self.X_train.shape[1] == 1:
            from matplotlib import pyplot as plt

            save_path = os.path.join(
                logdir, "plot_{}_n{:.2f}_s{}.png".format(self.name, self.noise, self.seed)
            )

            # Draw ground truth expression
            bounds = list(list(self.train_spec.values())[0].values())[0][:2]
            x = np.linspace(bounds[0], bounds[1], endpoint=True, num=100)
            y = self.numpy_expr(x[:, None])
            plt.plot(x, y, color="red", linestyle="dashed")
            # Draw the actual points
            plt.scatter(self.X_train, self.y_train)
            # Add a title
            plt.title("{} N:{} S:{}".format(self.name, self.noise, self.seed), fontsize=7)
            try:
                os.makedirs(logdir, exist_ok=True)
                plt.savefig(save_path)
                print("Saved plot to                  : {}".format(save_path))
            except Exception as e:
                logging.warning("Could not plot dataset: %s", e)
            plt.close()
        else:
            print("WARNING: Plotting only supported for 2D datasets.")


class TabularRegressionDataset(MetaData):
    """
    Lightweight wrapper for real (X, y) tabular regression data with no time or
    spatial axis (e.g. strain -> stress constitutive curves).

    Unlike `SymbolicRegressionDataset` (which *generates* (X, y) from a named
    benchmark expression), this class simply wraps already-collected real data.
    Domain-specific parsing (e.g. extracting a condition such as strain rate or
    temperature from a filename) belongs in the loader function that builds
    X/y, not in this class.
    """

    def __init__(
        self,
        name: str,
        X: np.ndarray,
        y: np.ndarray,
        *,
        variable_names: Optional[List[str]] = None,
        groups: Optional[np.ndarray] = None,
        sym_true: str = "",
        domain: Optional[Dict[str, Tuple[float, float]]] = None,
        descr: Optional["DatasetInfo"] = None,
    ):
        super().__init__(name)

        self.name = name
        self.domain = domain
        self.descr = descr
        self.sym_true = sym_true

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples, "
                f"got X.shape[0]={X.shape[0]} and y.shape[0]={y.shape[0]}."
            )

        self.X = X
        self.y = y
        self.n_input_dim = X.shape[1]

        if variable_names is None:
            variable_names = [f"x{i + 1}" for i in range(self.n_input_dim)]
        elif len(variable_names) != self.n_input_dim:
            raise ValueError(
                f"len(variable_names)={len(variable_names)} must match X.shape[1]={self.n_input_dim}"
            )
        self.variable_names = list(variable_names)

        self.groups = None if groups is None else np.asarray(groups)
        if self.groups is not None and self.groups.shape[0] != X.shape[0]:
            raise ValueError("`groups` must have the same length as X/y.")

        # Populated by train_test_split(); default to using the full data for both.
        self.X_train, self.y_train = self.X, self.y
        self.X_test, self.y_test = self.X, self.y

    def train_test_split(self, test_size: float = 0.2, seed: int = 0) -> None:
        """Populate X_train/y_train/X_test/y_test with a random split."""
        if not (0.0 < test_size < 1.0):
            raise ValueError("test_size must be in (0,1).")

        n = self.X.shape[0]
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        n_test = max(1, int(round(n * test_size)))
        test_idx, train_idx = idx[:n_test], idx[n_test:]

        self.X_train, self.y_train = self.X[train_idx], self.y[train_idx]
        self.X_test, self.y_test = self.X[test_idx], self.y[test_idx]

    def get_data(self) -> Dict[str, Any]:
        return {
            "X": self.X,
            "y": self.y,
            "variable_names": self.variable_names,
            "n_input_dim": self.n_input_dim,
            "sym_true": self.sym_true,
            "groups": self.groups,
        }

    def sample(
        self, n_samples: Union[int, float], *, seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        n_total = self.X.shape[0]

        if isinstance(n_samples, float):
            if not (0 < n_samples < 1):
                raise ValueError("If n_samples is float, it must be in (0,1).")
            n = int(round(n_total * n_samples))
        else:
            n = int(n_samples)

        if n <= 0 or n > n_total:
            raise ValueError(f"Requested {n} samples, but {n_total} available.")

        idx = rng.choice(n_total, size=n, replace=False)
        return self.X[idx], self.y[idx]

    def __repr__(self) -> str:
        return (
            f"TabularRegressionDataset(name='{self.name}', X={self.X.shape}, y={self.y.shape}, "
            f"variable_names={self.variable_names})"
        )


def load_burgers_equation():
    descr = DatasetInfo(
        description="""
        Dataset for high-viscosity Burgers equation 
        ut=-uux+0.1uxx
        x∈[-8.0,8.0), t∈[0,10]
        nx=256, nt=201, u.shape=(256,201)
        Resource:DLGA-PDE: Discovery of PDEs with incomplete candidate library via combination of deep learning and genetic algorithm
        """
    )

    file_path = resources.files(DATA_MODULE) / "burgers2.mat"
    pde_data = load_mat_file(file_path)
    return GridPDEDataset(
        equation_name="burgers equation",
        descr=descr,
        pde_data=pde_data,
        domain={"x": (-7.0, 7.0), "t": (1, 9)},
        epi=1e-3,
    )


def load_kdv_equation():
    descr = DatasetInfo(
        description="""
        Dataset for Korteweg-De Vries (KdV) equation with sin initial condition, actually a standardized form of Kdv_equation dataset
        ut=-uux-uxxx
        x∈[-20,20), t∈[0,40]
        nx=256, nt=201, u.shape=(256,201)
        Resource: PDE-READ: Human-readable Partial Differential Equation Discovery using Deep Learning, pp20
        """
    )

    file_path = resources.files(DATA_MODULE) / "KdV_equation.mat"
    pde_data = load_mat_file(file_path)

    return GridPDEDataset(
        equation_name="kdv equation",
        descr=descr,
        pde_data=None,
        x=pde_data["x"],
        t=pde_data["tt"],
        usol=pde_data["uu"],
        domain={"x": (-16, 16), "t": (5, 35)},
        epi=1e-3,
        legacy=True,
    )


def load_pde_dataset(
    filename: str,
    equation_name: str = "PDE Dataset",
    x_key: str = "x",
    t_key: str = "t",
    u_key: str = "usol",
    domain: dict = None,
    epi: float = 1e-3,
    data_dir_module: str = "kd.dataset.data",
):
    """Load a regular PDE grid from a MAT file using explicit array keys."""
    try:
        # Resolve the data file through the requested package.
        file_path = resources.files(data_dir_module) / filename

        # Read the MAT file with the low-level loader.
        pde_data = load_mat_file(file_path)

        # Extract coordinates and field values through the configured keys.
        x_data = np.asarray(pde_data[x_key], dtype=float).flatten()
        t_data = np.asarray(pde_data[t_key], dtype=float).flatten()
        u_data = np.asarray(pde_data[u_key])

        # Align the solution shape with (len(x), len(t)).
        if u_data.shape == (len(t_data), len(x_data)):
            u_data = u_data.T

        # Normalize the extracted arrays through GridPDEDataset.
        dataset = GridPDEDataset(
            equation_name=equation_name,
            pde_data=None,
            x=x_data,
            t=t_data,
            usol=u_data,
            domain=domain,
            epi=epi,
            legacy=True,
        )
        print(f"成功加载数据集: {equation_name}，文件: {filename}")
        return dataset

    except FileNotFoundError:
        print(f"错误: 在默认数据目录中未找到文件 {filename}")
        return None
    except KeyError as e:
        print(
            f"错误: 文件 {filename} 中缺少必需的键: {e}。请检查您传入的 x_key, t_key, u_key 参数是否正确。"
        )
        return None
    except Exception as e:
        print(f"错误: 加载或处理文件时发生未知错误: {e}")
        return None


# -------------------------
# Ball-drop ODE discovery dataset
# -------------------------

_BALL_NAME_MAP = {
    "Baseball": "baseball",
    "Blue Basketball": "bluebasketball",
    "Green Basketball": "greenbasketball",
    "Volleyball": "volleyball",
    "Bowling Ball": "bowlingball",
    "Golf Ball": "golfball",
    "Tennis Ball": "tennisball",
    "Whiffle Ball 1": "wiffleball1",
    "Whiffle Ball 2": "wiffleball2",
    "Yellow Whiffle Ball": "yellowWiffle",
    "Orange Whiffle Ball": "orangeWiffle",
}


def _load_ball_params(balls_txt_path: Path) -> Dict[str, Dict[str, float]]:
    """Parse `balls.txt` into {ball_key: {"mass": kg, "diameter": m}}, skipping entries with missing ("??") values."""
    params: Dict[str, Dict[str, float]] = {}
    with balls_txt_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines[1:]:  # skip header row
        parts = line.split()
        if len(parts) < 3:
            continue
        name, weight_oz, circumference_cm = parts[0], parts[1], parts[2]
        try:
            weight_oz = float(weight_oz)
            circumference_cm = float(circumference_cm)
        except ValueError:
            continue  # missing data marked as "??"

        mass_kg = weight_oz * 0.0283495
        circumference_m = circumference_cm / 100.0
        diameter_m = circumference_m / np.pi
        params[name] = {"mass": mass_kg, "diameter": diameter_m}

    return params


def load_ball_drop_dataset(
    exclude: Tuple[str, ...] = ("Bowling Ball", "Volleyball"),
    include_params: bool = True,
) -> "ODEDataset":
    """
    Load the falling-ball ODE discovery dataset from
    `discovery-of-physics-from-data/data/Ball_drops_data.xls` (+ `balls.txt`).

    Each (ball, drop) pair becomes one trajectory with state variables
    ["h", "v"] (height in m, velocity in m/s), optionally paired with static
    per-trajectory parameters {"mass": kg, "diameter": m} parsed from
    `balls.txt`.

    Resource: de Silva et al., "Discovery of Physics From Data: Universal Laws
    and Discrepancies", Frontiers in Artificial Intelligence (2020).
    """
    base_dir = Path(__file__).resolve().parent / "discovery-of-physics-from-data" / "data"
    xls_path = base_dir / "Ball_drops_data.xls"
    balls_txt_path = base_dir / "balls.txt"

    if not xls_path.exists():
        raise FileNotFoundError(f"Ball drop data file not found: {xls_path}")

    ball_params = _load_ball_params(balls_txt_path) if include_params else {}

    sheets = pd.read_excel(xls_path, sheet_name=None)

    descr = DatasetInfo(
        description="""
        Falling-ball ODE discovery dataset: height & velocity vs time for
        repeated drop experiments of various balls (baseball, basketballs,
        golf ball, etc). state_vars = ["h" (m), "v" (m/s)]; optional
        per-trajectory params {"mass" (kg), "diameter" (m)} parsed from ball
        weight/circumference.
        Resource: de Silva et al., Discovery of Physics From Data: Universal
        Laws and Discrepancies, Frontiers in Artificial Intelligence (2020).
        """
    )

    trajectories = []
    for ball_name, df in sheets.items():
        if ball_name in exclude:
            continue

        ball_key = _BALL_NAME_MAP.get(ball_name)
        params = ball_params.get(ball_key) if (include_params and ball_key) else None
        if include_params and params is None:
            continue  # missing mass/diameter for this ball

        for drop_id in sorted(df["Drop #"].unique()):
            sub = df[df["Drop #"] == drop_id].sort_values("Time (s)")
            t = sub["Time (s)"].to_numpy(dtype=float)
            state = np.vstack(
                [
                    sub["Height (m)"].to_numpy(dtype=float),
                    sub["Velocity (m/s)"].to_numpy(dtype=float),
                ]
            )
            trajectories.append(
                {
                    "t": t,
                    "state": state,
                    "params": params,
                    "id": f"{ball_name}-drop{int(drop_id)}",
                }
            )

    if len(trajectories) == 0:
        raise ValueError("No trajectories loaded; check `exclude` and balls.txt parsing.")

    return ODEDataset(
        equation_name="ball_drop",
        trajectories=trajectories,
        state_vars=["h", "v"],
        param_names=["mass", "diameter"] if include_params else None,
        descr=descr,
    )


# -------------------------
# Filled-rubber constitutive regression dataset
# -------------------------

_RUBBER_FILENAME_RE = re.compile(r"^C(?P<compound>\d+)_(?P<temperature>\d+)\.xlsx$", re.IGNORECASE)


def load_rubber_dataset(split: str = "train") -> "TabularRegressionDataset":
    """
    Load the filled-rubber constitutive-model regression dataset from
    `Discovery_of_soild_consititutive/data/data_rubber/{train,test}/*.xlsx`.

    Each file `C{compound}_{temperature}.xlsx` has columns "nominal
    strain"/"nominal stress" for one (compound, temperature) condition.

    Output schema (matches `kd.data.RegularData.load_regression_data`
    docstring):
        X = [lambda = 1 + nominal strain, compound, temperature]
        y = nominal stress

    Resource: "Beyond empirical models: Discovering constitutive laws in
    solids with graph-based equation discovery".
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    data_dir = (
        Path(__file__).resolve().parent
        / "Discovery_of_soild_consititutive"
        / "data"
        / "data_rubber"
        / split
    )
    if not data_dir.exists():
        raise FileNotFoundError(f"Rubber data directory not found: {data_dir}")

    file_paths = sorted(data_dir.glob("*.xlsx"))
    if len(file_paths) == 0:
        raise FileNotFoundError(f"No .xlsx files found in {data_dir}")

    X_rows = []
    y_rows = []
    groups = []
    for file_path in file_paths:
        m = _RUBBER_FILENAME_RE.match(file_path.name)
        if not m:
            continue
        compound = float(m.group("compound"))
        temperature = float(m.group("temperature"))

        df = pd.read_excel(file_path)
        strain = df["nominal strain"].to_numpy(dtype=float)
        stress = df["nominal stress"].to_numpy(dtype=float)
        valid = np.isfinite(strain) & np.isfinite(stress)
        strain, stress = strain[valid], stress[valid]

        lam = 1.0 + strain
        n = lam.shape[0]
        X_rows.append(np.column_stack([lam, np.full(n, compound), np.full(n, temperature)]))
        y_rows.append(stress)
        groups.extend([file_path.name] * n)

    if len(X_rows) == 0:
        raise ValueError(
            f"No files in {data_dir} matched the C<compound>_<temperature>.xlsx pattern."
        )

    X = np.vstack(X_rows)
    y = np.concatenate(y_rows)

    descr = DatasetInfo(
        description="""
        Filled-rubber constitutive-model regression dataset (nominal stress
        vs stretch ratio, parametric in compound and temperature).
        X = [lambda=1+nominal strain, compound, temperature (K)], y = nominal
        stress.
        Resource: Beyond empirical models: Discovering constitutive laws in
        solids with graph-based equation discovery.
        """
    )

    return TabularRegressionDataset(
        name=f"rubber_{split}",
        X=X,
        y=y,
        variable_names=["lambda", "C", "T"],
        groups=np.array(groups),
        descr=descr,
    )


# -------------------------
# CYT RANS turbulence-closure regression dataset
# -------------------------

_CYT_COLUMNS = [
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

_CYT_CASE_ALIASES = {
    "flatplate": "FlatPlate_lk0.215andPplus",
    "naca0012": "NACA0012_Re4e5_MLen_BEST",
}

_CYT_DEFAULT_FEATURE_COLUMNS = [
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

# Fortran free-format write can underflow the exponent field width for very
# small magnitudes and silently drop the 'E' (e.g. "0.103578010-307" instead
# of "0.103578010E-307"); repair that here.
_CYT_FORTRAN_FLOAT_RE = re.compile(r"^([+-]?\d*\.\d+)([+-]\d{2,3})$")


def _parse_fortran_float(tok: str) -> float:
    try:
        return float(tok)
    except ValueError:
        m = _CYT_FORTRAN_FLOAT_RE.match(tok)
        if m:
            return float(m.group(1) + "E" + m.group(2))
        raise


def _resolve_cyt_case_dir(case: str) -> Path:
    case_dir_name = _CYT_CASE_ALIASES.get(case, case)
    case_dir = Path(__file__).resolve().parent / "CYT" / case_dir_name
    if not case_dir.exists():
        raise FileNotFoundError(
            f"CYT case not found: {case!r} (resolved to {case_dir}). "
            f"Known aliases: {list(_CYT_CASE_ALIASES)}"
        )
    return case_dir


def load_cyt_flowfeature_raw(case: str) -> Dict[str, np.ndarray]:
    """
    Parse `Output/FlowFeature.dat` for a CYT RANS-turbulence CFD case into a
    dict of {column_name: (N,) ndarray}, one entry per mesh cell.

    Each record in the file is a Fortran free-format write split across two
    physical text lines (27 values on line 1, 5 on line 2 -- 32 columns
    total); see `_CYT_COLUMNS` for the column order.

    `case` may be a full CYT case directory name (e.g.
    "FlatPlate_lk0.215andPplus") or a short alias ("flatplate", "naca0012").
    """
    case_dir = _resolve_cyt_case_dir(case)
    file_path = case_dir / "Output" / "FlowFeature.dat"
    if not file_path.exists():
        raise FileNotFoundError(f"FlowFeature.dat not found: {file_path}")

    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        header = f.readline().split()
        if header != _CYT_COLUMNS:
            raise ValueError(f"Unexpected FlowFeature.dat header for case {case!r}: {header}")

        rows = []
        for line1 in f:
            line2 = f.readline()
            tokens = line1.split() + line2.split()
            if len(tokens) != len(_CYT_COLUMNS):
                raise ValueError(
                    f"Malformed FlowFeature.dat record in {file_path} "
                    f"(expected {len(_CYT_COLUMNS)} values, got {len(tokens)})"
                )
            rows.append([_parse_fortran_float(tok) for tok in tokens])

    arr = np.asarray(rows, dtype=float)
    return {name: arr[:, i] for i, name in enumerate(_CYT_COLUMNS)}


def load_cyt_dataset(
    case: str = "flatplate",
    target: str = "Mut",
    feature_columns: Optional[List[str]] = None,
) -> "TabularRegressionDataset":
    """
    Load a CYT RANS turbulence-closure regression dataset.

    Per-cell flow features (velocity, pressure, their gradients, wall
    distance, Mach/AoA/Re) become X, and a turbulence-closure quantity
    (default eddy viscosity "Mut") becomes y -- a constitutive-law discovery
    problem ("given local flow invariants, discover the closure relation"),
    analogous to `load_rubber_dataset`.

    Parameters
    ----------
    case : str
        CYT case name or alias ("flatplate" / "naca0012", or the full CYT
        directory name).
    target : str
        Column name (see `_CYT_COLUMNS`) to use as the regression target y.
        Default "Mut" (eddy viscosity). Other closure quantities such as
        "Txx"/"Txz"/"Tzz" (Reynolds stress components) or "prod"/"diff"/
        "destr" (turbulence budget terms) can be used instead.
    feature_columns : list[str], optional
        Column names to use as X. Defaults to velocity/density/pressure/
        gradients/wall-distance/Ma/AoA/Re (excludes raw coordinates X,Y and
        other closure-output columns).
    """
    columns = load_cyt_flowfeature_raw(case)

    if target not in columns:
        raise ValueError(f"Unknown target column {target!r}. Available: {_CYT_COLUMNS}")

    feats = (
        list(feature_columns) if feature_columns is not None else list(_CYT_DEFAULT_FEATURE_COLUMNS)
    )
    for feat in feats:
        if feat not in columns:
            raise ValueError(f"Unknown feature column {feat!r}. Available: {_CYT_COLUMNS}")

    X = np.column_stack([columns[feat] for feat in feats])
    y = columns[target]

    case_dir_name = _CYT_CASE_ALIASES.get(case, case)

    descr = DatasetInfo(
        description=f"""
        CYT RANS turbulence-closure dataset, case '{case_dir_name}'.
        Per-cell flow features (from Output/FlowFeature.dat, {X.shape[0]}
        cells) for turbulence closure / constitutive-law discovery:
        X = {feats}, y = '{target}'.
        """
    )

    return TabularRegressionDataset(
        name=f"cyt_{case}",
        X=X,
        y=y,
        variable_names=feats,
        descr=descr,
    )


# -------------------------
# Solid-constitutive strain-rate / hardening regression datasets
# -------------------------

_SOLID_DATA_DIR = Path(__file__).resolve().parent / "Discovery_of_soild_consititutive" / "data"

# Fortran/Excel-authoring quirk: a strain-rate filename suffix like "1-e4"
# (instead of the standard "1e-4") appears once in data_strain_stress; repair it.
_SOLID_STRAIN_RATE_FIX_RE = re.compile(r"^(\d+(?:\.\d+)?)-e(\d+)$")


def _parse_strain_rate_token(tok: str) -> float:
    try:
        return float(tok)
    except ValueError:
        m = _SOLID_STRAIN_RATE_FIX_RE.match(tok)
        if m:
            return float(m.group(1)) * (10.0 ** -int(m.group(2)))
        raise ValueError(f"Cannot parse strain rate from filename token: {tok!r}")


def load_solid_dif_dataset() -> "TabularRegressionDataset":
    """
    Load the Dynamic-Increase-Factor (DIF) vs strain-rate dataset from
    `Discovery_of_soild_consititutive/data/data_DIF/*.xlsx` (63 files, one
    material/loading-condition curve each, columns "strain rate"/"DIF",
    compiled from 18 published studies).

    X = [strain_rate], y = DIF, one row per (file, sample) pair; `groups`
    holds the source filename so curves from different materials/studies can
    be told apart (a single global X->y expression is not expected to fit
    perfectly across materials, mirroring how the original repository code
    fits each file's curve separately).

    Resource: "Beyond empirical models: Discovering constitutive laws in
    solids with graph-based equation discovery".
    """
    data_dir = _SOLID_DATA_DIR / "data_DIF"
    if not data_dir.exists():
        raise FileNotFoundError(f"data_DIF directory not found: {data_dir}")

    file_paths = sorted(data_dir.glob("*.xlsx"))
    if len(file_paths) == 0:
        raise FileNotFoundError(f"No .xlsx files found in {data_dir}")

    X_rows = []
    y_rows = []
    groups = []
    for file_path in file_paths:
        df = pd.read_excel(file_path)
        strain_rate = df["strain rate"].to_numpy(dtype=float)
        dif = df["DIF"].to_numpy(dtype=float)
        valid = np.isfinite(strain_rate) & np.isfinite(dif)
        strain_rate, dif = strain_rate[valid], dif[valid]

        X_rows.append(strain_rate.reshape(-1, 1))
        y_rows.append(dif)
        groups.extend([file_path.stem] * strain_rate.shape[0])

    X = np.vstack(X_rows)
    y = np.concatenate(y_rows)

    descr = DatasetInfo(
        description="""
        Dynamic Increase Factor (DIF) vs strain-rate dataset, compiled from
        40 materials across 18 published studies.
        X = [strain_rate (1/s)], y = DIF (dimensionless).
        Resource: Beyond empirical models: Discovering constitutive laws in
        solids with graph-based equation discovery.
        """
    )

    return TabularRegressionDataset(
        name="solid_dif",
        X=X,
        y=y,
        variable_names=["strain_rate"],
        groups=np.array(groups),
        descr=descr,
    )


def load_solid_strain_stress_dataset() -> "TabularRegressionDataset":
    """
    Load the strain-hardening dataset from
    `Discovery_of_soild_consititutive/data/data_strain_stress/*.xlsx` (64
    files, columns "true plastic strain"/"true plastic stress"; the strain
    rate for each file's curve is encoded in its filename, e.g.
    "Chen_2017_4.8.xlsx" -> 4.8/s, "Gao(22MnB5)_2020_1-e4.xlsx" -> 1e-4/s).

    X = [true_plastic_strain, strain_rate], y = true_plastic_stress, one row
    per (file, sample) pair; `groups` holds the source filename.

    Resource: "Beyond empirical models: Discovering constitutive laws in
    solids with graph-based equation discovery".
    """
    data_dir = _SOLID_DATA_DIR / "data_strain_stress"
    if not data_dir.exists():
        raise FileNotFoundError(f"data_strain_stress directory not found: {data_dir}")

    file_paths = sorted(data_dir.glob("*.xlsx"))
    if len(file_paths) == 0:
        raise FileNotFoundError(f"No .xlsx files found in {data_dir}")

    X_rows = []
    y_rows = []
    groups = []
    for file_path in file_paths:
        rate_token = file_path.stem.rsplit("_", 1)[-1]
        strain_rate = _parse_strain_rate_token(rate_token)

        df = pd.read_excel(file_path)
        strain = df["true plastic strain"].to_numpy(dtype=float)
        stress = df["true plastic stress"].to_numpy(dtype=float)
        valid = np.isfinite(strain) & np.isfinite(stress)
        strain, stress = strain[valid], stress[valid]

        n = strain.shape[0]
        X_rows.append(np.column_stack([strain, np.full(n, strain_rate)]))
        y_rows.append(stress)
        groups.extend([file_path.stem] * n)

    X = np.vstack(X_rows)
    y = np.concatenate(y_rows)

    descr = DatasetInfo(
        description="""
        Strain-hardening (true plastic stress vs true plastic strain)
        dataset, parametric in strain rate (encoded per-file in the source
        filename).
        X = [true_plastic_strain, strain_rate (1/s)], y = true_plastic_stress.
        Resource: Beyond empirical models: Discovering constitutive laws in
        solids with graph-based equation discovery.
        """
    )

    return TabularRegressionDataset(
        name="solid_strain_stress",
        X=X,
        y=y,
        variable_names=["strain", "strain_rate"],
        groups=np.array(groups),
        descr=descr,
    )


def load_solid_hardening_dataset() -> "TabularRegressionDataset":
    """
    Load the pre-combined strain-hardening + strain-rate-effect dataset from
    `Discovery_of_soild_consititutive/data/saved_data_hardening_strain_rate/*.pkl`
    (10 materials). Each pickle is a dict of parallel lists
    {"strain": [...], "stress": [...], "strain_rate": [...], "DIF": [...]},
    one entry per (material, rate) curve, with DIF already matched from the
    corresponding data_DIF file -- the "integrated model" data used to
    discover a combined stress = f(strain) * g(strain_rate) constitutive law.

    X = [strain, strain_rate, DIF], y = stress, one row per (file, curve,
    sample) triple; `groups` holds "{filename}_curve{i}".

    Resource: "Beyond empirical models: Discovering constitutive laws in
    solids with graph-based equation discovery".
    """
    data_dir = _SOLID_DATA_DIR / "saved_data_hardening_strain_rate"
    if not data_dir.exists():
        raise FileNotFoundError(f"saved_data_hardening_strain_rate directory not found: {data_dir}")

    file_paths = sorted(data_dir.glob("*.pkl"))
    if len(file_paths) == 0:
        raise FileNotFoundError(f"No .pkl files found in {data_dir}")

    X_rows = []
    y_rows = []
    groups = []
    for file_path in file_paths:
        with file_path.open("rb") as f:
            data = pickle.load(f)

        for i, (strain, stress, strain_rate, dif) in enumerate(
            zip(data["strain"], data["stress"], data["strain_rate"], data["DIF"])
        ):
            strain = np.asarray(strain, dtype=float)
            stress = np.asarray(stress, dtype=float)
            n = strain.shape[0]

            X_rows.append(
                np.column_stack(
                    [
                        strain,
                        np.full(n, float(strain_rate)),
                        np.full(n, float(dif)),
                    ]
                )
            )
            y_rows.append(stress)
            groups.extend([f"{file_path.stem}_curve{i}"] * n)

    X = np.vstack(X_rows)
    y = np.concatenate(y_rows)

    descr = DatasetInfo(
        description="""
        Pre-combined strain-hardening + strain-rate-effect ("integrated
        model") dataset, with DIF pre-matched per (material, strain-rate)
        curve from the corresponding data_DIF file.
        X = [strain, strain_rate (1/s), DIF], y = true_plastic_stress.
        Resource: Beyond empirical models: Discovering constitutive laws in
        solids with graph-based equation discovery.
        """
    )

    return TabularRegressionDataset(
        name="solid_hardening",
        X=X,
        y=y,
        variable_names=["strain", "strain_rate", "DIF"],
        groups=np.array(groups),
        descr=descr,
    )


# -------------------------
# Viscous gravity current (VGS) proppant-transport PDE dataset
# -------------------------

_VGS_ROOT = Path(__file__).resolve().parent / "ViscousGravityCurrent" / "data"

_VGS_NX = 500
_VGS_DX = 2e-2

# Canonical raw-data file per physical case and time window. The imported
# upstream repository contained several pipeline-stage copies; only the copy
# consumed by the original training entry point is retained here.
_VGS_CASE_CONFIG = {
    ("I", "0-100"): {
        "file": _VGS_ROOT / "vgs_I_0-100.dat",
        "t_arange": (0, 100, 0.2),
        "t_coeff": 1.22625e-1,
    },
    ("I", "100-200"): {
        "file": _VGS_ROOT / "vgs_I_100-200.dat",
        "t_arange": (0, 100, 0.2),
        "t_coeff": 1.22625e-1,
    },
    ("II", "0-1000"): {
        "file": _VGS_ROOT / "vgs_II_0-1000.dat",
        "t_arange": (0, 1000, 2),
        "t_coeff": 3.310149e-3,
    },
    ("II", "1000-2000"): {
        "file": _VGS_ROOT / "vgs_II_1000-2000.dat",
        "t_arange": (0, 1000, 2),
        "t_coeff": 3.310149e-3,
    },
}

_VGS_CASE_ALIASES = {"I": "I", "1": "I", "II": "II", "2": "II"}
_VGS_DEFAULT_WINDOW = {"I": "0-100", "II": "0-1000"}


def load_vgs_dataset(case: str = "I", window: Optional[str] = None) -> "GridPDEDataset":
    """
    Load a viscous-gravity-current / proppant-transport PDE discovery
    dataset from `ViscousGravityCurrent/data/`.

    The raw data is the free-surface height h(x, t) of a spreading
    two-phase (proppant-laden slurry vs ambient fluid) gravity current,
    extracted from a 2D Stokes/level-set simulation, on a fixed 500 (space)
    x 500 (time) grid, x in [0, 9.98]. Values are
    the raw, unnormalized interface height as written by the simulation
    (roughly in [0.5, 49.5]); the original ML pipeline additionally divided
    by 50 before feeding it to a neural surrogate, which this loader does
    not do.

    Two physical cases are available, each split into two time windows
    (different segments of the same underlying simulation). The exposed
    time axis is reset to start at 0 for each window, following the
    original pipeline's own convention -- reasonable for autonomous PDE
    discovery, where only relative time spacing matters, not absolute
    offset:
        case="I":  windows "0-100" (default) / "100-200"
        case="II": windows "0-1000" (default) / "1000-2000"

    No confirmed ground-truth governing equation (`sym_true`) exists
    anywhere in the source repository -- this is left unset.

    Parameters
    ----------
    case : str
        "I" (or "1") / "II" (or "2").
    window : str, optional
        Time window for the chosen case (see above). Defaults to the
        first/earlier window for that case.

    Resource: DLGA-PDE (deep-learning + genetic-algorithm PDE discovery)
    applied to viscous gravity currents / proppant transport.
    """
    case_key = _VGS_CASE_ALIASES.get(case, case)
    if case_key not in _VGS_DEFAULT_WINDOW:
        raise ValueError(f"Unknown VGS case: {case!r}. Available: {sorted(_VGS_DEFAULT_WINDOW)}")

    if window is None:
        window = _VGS_DEFAULT_WINDOW[case_key]

    config = _VGS_CASE_CONFIG.get((case_key, window))
    if config is None:
        available = [w for (c, w) in _VGS_CASE_CONFIG if c == case_key]
        raise ValueError(
            f"Unknown window {window!r} for case {case_key!r}. Available windows: {available}"
        )

    file_path = config["file"]
    if not file_path.exists():
        raise FileNotFoundError(f"VGS raw data file not found: {file_path}")

    raw = np.loadtxt(str(file_path))  # (nt, nx), Fortran/Python-order: rows=time, cols=space
    usol = raw.T  # (nx, nt) to match GridPDEDataset's legacy convention

    nx = usol.shape[0]
    x = np.arange(0, nx, 1) * _VGS_DX

    start, stop, step = config["t_arange"]
    t = np.arange(start, stop, step) * config["t_coeff"]

    descr = DatasetInfo(
        description=f"""
        Viscous gravity current / proppant transport PDE discovery dataset,
        case {case_key}, window {window} (segment of the underlying
        simulation). h(x, t): free-surface height of a spreading two-phase
        gravity current, from a 2D Stokes/level-set simulation. x in
        [0, {x.max():.4g}], t in [0, {t.max():.4g}] (locally reset per
        window). No confirmed ground-truth sym_true available.
        Resource: DLGA-PDE (deep-learning genetic-algorithm PDE discovery)
        applied to viscous gravity currents / proppant transport.
        """
    )

    return GridPDEDataset(
        equation_name=f"vgs_case{case_key}_{window}",
        pde_data=None,
        x=x,
        t=t,
        usol=usol,
        domain={"x": (float(x.min()), float(x.max())), "t": (float(t.min()), float(t.max()))},
        epi=1e-3,
        descr=descr,
        legacy=True,
    )
