"""
Base IO code for all datasets
"""
import ast
import csv
import itertools
import os
import zlib

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from typing import Any, Dict, Union, Optional, Tuple
from pathlib import Path
from importlib import resources
from abc import ABC, abstractmethod

from ._info import DatasetInfo
from scipy.interpolate import interp2d

DATA_MODULE = "kd.dataset.data"
GAMMA = 0.57721566490153286060651209008240243104215933593992

def _harmonic(x1):
    if all(val.is_integer() for val in x1):
        return np.array([sum(1/d for d in range(1, int(val)+1)) for val in x1], dtype=np.float32)
    else:
        return GAMMA + np.log(x1) + 0.5/x1 - 1./(12*x1**2) + 1./(120*x1**4)

_function_map = {
    'pi': np.pi,
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan,
    'exp': np.exp,
    'log': np.log,
    'sqrt': np.sqrt,
    'div': np.divide,
    'harmonic': _harmonic,
}

def _convert_data_dataframe(
    data, target, feature_names, target_names, sparse_data=False
):
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


def load_csv_data(data_file_path: str, encoding: str = "utf-8", has_header: bool = True) -> np.ndarray:
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
            # If the file has a header, remove the first row
            header = rows[0]  # Store header if needed (can be returned or processed)
            data_rows = rows[1:]
        else:
            # If no header, use all rows as data
            data_rows = rows
        
        # Convert the data into a NumPy array
        data_array = np.array(data_rows, dtype=np.float32)  # Assuming the data is numeric; handle further conversions if needed
        
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
        mat_data_clean = {key: value for key, value in mat_data.items() if not key.startswith('__')}
        return mat_data_clean
    except NotImplementedError:
        # This error will occur if scipy cannot handle the file (e.g., MATLAB version > 7.3)
        raise ValueError("The .mat file is of an unsupported format (likely version 7.3 or higher).")

  
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
    if file_path.endswith('.npy'):
        # Load a single NumPy array from a .npy file
        return np.load(file_path)
    elif file_path.endswith('.npz'):
        # Load a NumPy compressed archive (.npz) and return as a dictionary of arrays
        return np.load(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


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

    def load_data(self, equation_name: str = None, file: str = None) -> Union[np.ndarray, Dict[str, Any]]:
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
    required_methods = {'get_data'}
    required_attributes = ('x', 't', 'usol')
    
    def __new__(cls, name, bases, dct):
        """
        Overrides class creation to enforce method and attribute requirements.
        """
                
        if not isinstance(cls.required_attributes, (set, list, tuple)):
            raise TypeError(f"{name}.required_attributes must be a set, list, or tuple.")
        
        # Ensure required methods are implemented
        for method in cls.required_methods:
            if method not in dct:
                raise TypeError(f'{name} must implement the method: {method}')
        
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


class PDEDataset(MetaData):
    """ 
    A class representing a Partial Differential Equation (PDE) dataset, providing data access 
    and analysis functionality.
    """
    
    def __init__(self, 
                 equation_name: str,
                 pde_data: Optional[Dict[str, Any]],
                 domain: Optional[Dict[str, Tuple[float, float]]],
                 epi: float,
                 x: Optional[np.ndarray] = None, 
                 t: Optional[np.ndarray] = None, 
                 usol: Optional[np.ndarray] = None,
                 descr: Optional[DatasetInfo] = None):
        """
        Initializes the PDE dataset, supporting two input methods:
        1. Providing data through the `pde_data` dictionary.
        2. Directly passing `x`, `t`, and `usol` arrays.

        :param equation_name: Name of the PDE.
        :param descr: Metadata containing information about the PDE.
        :param pde_data: Optional dictionary containing 'x', 't', and 'usol' data.
        :param x: Optional, spatial coordinate array.
        :param t: Optional, temporal coordinate array.
        :param usol: Optional, solution array u(x, t).
        :param domain: Dictionary defining the domain {variable: (min_value, max_value)}.
        :param epi: Additional parameter.
        """
        super().__init__(equation_name)

        if pde_data is not None:
            # Assign data from the provided dictionary
            self.x = np.asarray(pde_data.get('x'), dtype=float).flatten()
            self.t = np.asarray(pde_data.get('t'), dtype=float).flatten()
            self.usol = np.real(np.asarray(pde_data.get('usol')))
        elif x is not None and t is not None and usol is not None:
            # Directly assign provided x, t, and usol arrays
            self.x = np.asarray(x, dtype=float).flatten()
            self.t = np.asarray(t, dtype=float).flatten()
            self.usol = np.real(np.asarray(usol))
        else:
            raise ValueError("Either `pde_data` or `x`, `t`, and `usol` must be provided.")

        # Ensure consistency of data dimensions
        if self.usol.shape != (len(self.x), len(self.t)):
            raise ValueError(f"usol dimensions {self.usol.shape} do not match x {len(self.x)} and t {len(self.t)}")
        
        self.u = self.usol
        self.equation_name = equation_name
        self.domain = domain
        self.epi = epi 
        
    def get_datapoint(self, x_id: int, t_id: int) -> Tuple[float, float, float]:
        """
        Retrieves the (x, t) coordinates and the corresponding solution value.

        :param x_id: Index in the spatial dimension.
        :param t_id: Index in the temporal dimension.
        :return: A tuple (x, t, usol) representing the coordinates and solution.
        """
        if not (0 <= x_id < len(self.x)) or not (0 <= t_id < len(self.t)):
            raise IndexError("Index out of range.")
        return self.x[x_id], self.t[t_id], self.usol[x_id, t_id]

    def get_data(self) -> Dict[str, Any]:
        """
        Returns the dataset information as a dictionary.
        """
        return {'x': self.x, 't': self.t, 'usol': self.usol}
        
    def sample(self, n_samples: Union[int, float], method: str = 'random') -> Tuple[np.ndarray, np.ndarray]:
        """
        Samples a subset of the data points using different sampling methods.

        :param n_samples: Number of samples to draw. If < 1, treated as a ratio.
        :param method: Sampling method to use ('random', 'uniform', 'spline').
        :return: A tuple (sampled_points, sampled_usol), where:
                - sampled_points is an array of shape (n, 2) with (x, t) pairs
                - sampled_usol is an array of shape (n,) with corresponding solution values
        """
        total_points = len(self.x) * len(self.t)
        
        # Convert ratio to absolute count if n_samples is a float < 1
        if isinstance(n_samples, float) and 0 < n_samples < 1:
            n_samples = int(total_points * n_samples)

        if n_samples > total_points:
            raise ValueError(f"Requested {n_samples} samples, but only {total_points} points available.")

        if method == 'random':
            indices = np.random.choice(total_points, n_samples, replace=False)
            x_samples, t_samples = np.unravel_index(indices, self.usol.shape)
            sampled_points = np.column_stack((self.x[x_samples], self.t[t_samples]))
            sampled_usol = self.usol[x_samples, t_samples]

        elif method == 'uniform':
            grid_size = int(np.sqrt(n_samples))
            x_indices = np.linspace(0, len(self.x) - 1, grid_size, dtype=int)
            t_indices = np.linspace(0, len(self.t) - 1, grid_size, dtype=int)
            x_samples, t_samples = np.meshgrid(x_indices, t_indices)
            sampled_points = np.column_stack((
                self.x[x_samples.flatten()],
                self.t[t_samples.flatten()]
            ))
            sampled_usol = self.usol[x_samples.flatten(), t_samples.flatten()]

        elif method == 'spline':
            grid_size = int(np.sqrt(n_samples))
            x_new = np.linspace(self.x.min(), self.x.max(), grid_size)
            t_new = np.linspace(self.t.min(), self.t.max(), grid_size)
            spline = interp2d(self.t, self.x, self.usol, kind='cubic')
            usol_new = spline(t_new, x_new)
            x_samples, t_samples = np.meshgrid(x_new, t_new)
            sampled_points = np.column_stack((x_samples.flatten(), t_samples.flatten()))
            sampled_usol = usol_new.flatten()

        else:
            raise ValueError(f"Unsupported sampling method: {method}")

        return sampled_points, sampled_usol.reshape(-1, 1)
    
    def mesh(self, indexing='ij') -> np.ndarray:
        """
        Returns a 2D coordinate grid where each row is a combination of (x, t).
        Equivalent to np.meshgrid(x, t) but flattened into a 2D array.

        :return: A 2D array of shape (len(x) * len(t), 2) where each row is (x, t).
        """
        X, T = np.meshgrid(self.x, self.t, indexing=indexing)
        return np.column_stack([X.ravel(), T.ravel()])

    def mesh_bounds(self, indexing='ij') -> np.ndarray:
        X, T = np.meshgrid(self.x, self.t, indexing=indexing)
        mesh_data = np.column_stack([X.ravel(), T.ravel()])
        return mesh_data.min(0), mesh_data.max(0)
    
    def get_boundaries(self) -> Dict[str, Tuple[float, float]]:
        """ Returns the minimum and maximum values for x and t. """
        return {'x': (self.x.min(), self.x.max()), 't': (self.t.min(), self.t.max())}

    def get_domain(self) -> Dict[str, Tuple[float, float]]:
        """ Retrieves the domain definition. """
        return self.domain

    def get_derivative(self, axis: str = 'x') -> np.ndarray:
        """
        Computes the derivative of `usol` along the spatial or temporal axis.

        :param axis: Direction of differentiation ('x' or 't').
        :return: The computed derivative array.
        """
        if axis == 'x':
            return np.gradient(self.usol, axis=0)
        elif axis == 't':
            return np.gradient(self.usol, axis=1)
        else:
            raise ValueError("Invalid axis name, must be 'x' or 't'.")

    def get_range(self, x_range: Tuple[float, float], t_range: Tuple[float, float]) -> Dict[str, np.ndarray]:
        """ 
        Retrieves subsets of x, t, and the corresponding solution within the specified range.

        :param x_range: Tuple representing the range of x values (min_x, max_x).
        :param t_range: Tuple representing the range of t values (min_t, max_t).
        :return: A dictionary containing subsets of x, t, and usol.
        """
        x_start, x_end = np.searchsorted(self.x, [x_range[0], x_range[1]])
        t_start, t_end = np.searchsorted(self.t, [t_range[0], t_range[1]])

        return {'x': self.x[x_start:x_end], 
                't': self.t[t_start:t_end], 
                'usol': self.usol[x_start:x_end, t_start:t_end]}

    def get_size(self) -> Tuple[int, int]:
        """ Returns the dimensions of the solution array (x dimension, t dimension). """
        return self.usol.shape

    def plot_solution(self) -> None:
        """
        Generates a heatmap visualization of the solution `usol` over (x, t) dimensions.
        """
        plt.figure(figsize=(8, 6))
        plt.imshow(self.usol, aspect='auto', origin='lower',
                   extent=[self.t.min(), self.t.max(), self.x.min(), self.x.max()])
        plt.colorbar(label='Solution u(x, t)')
        plt.xlabel('Time (t)')
        plt.ylabel('Space (x)')
        plt.title(f"Solution of {self.equation_name}")
        plt.show()

    def __repr__(self) -> str:
        """ Returns a string representation of the dataset. """
        return (f"PDEDataset(equation='{self.equation_name}', size={self.get_size()}, "
                f"boundaries={self.get_boundaries()})")
    
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

    def __init__(self, name, benchmark_source="benchmarks.csv", root=None, noise=0.0,
                 seed=0, logdir=None, backup=False):
        # Set class variables
        super().__init__(name)
        self.name = name
        self.seed = seed
        self.noise = noise if noise is not None else 0.0

        # Set random number generator used for sampling X values
        seed += zlib.adler32(name.encode("utf-8")) # Different seed for each name, otherwise two benchmarks with the same domain will always have the same X values
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
            print('WARNING: Ignoring negative noise value: {}'.format(self.noise))

        # Load default function set
        function_set_path = os.path.join(root, "function_sets.csv")
        function_set_df = pd.read_csv(function_set_path, index_col=0)
        function_set_name = row["function_set"]
        self.function_set = function_set_df.loc[function_set_name].tolist()[0].strip().split(',')

        # Prepare status output
        output_message = '\n-- BUILDING DATASET START -----------\n'
        output_message += 'Generated data for benchmark   : {}\n'.format(name)
        output_message += 'Benchmark path                 : {}\n'.format(benchmark_path)
        output_message += 'Function set                   : {} --> {}\n'.format(function_set_name, self.function_set)
        output_message += 'Function set path              : {}\n'.format(function_set_path)
        test_spec_txt = row["test_spec"] if row["test_spec"] != "None" else "{} (Copy from train!)".format(row["test_spec"])
        output_message += 'Dataset specifications         : \n' \
                          + '        Train --> {}\n'.format(row["train_spec"]) \
                          + '        Test  --> {}\n'.format(test_spec_txt)
        random_choice_train = self.rng.randint(self.X_train.shape[0])
        random_sample_train = "[{}],[{}]".format(self.X_train[random_choice_train], self.y_train[random_choice_train])
        output_message += 'Built data set                 : \n' \
                          + '        Train --> X:{}, y:{}, Sample: {}\n'.format(self.X_train.shape, self.y_train.shape, random_sample_train)
        if row["test_spec"] is not None:
            random_choice_test = self.rng.randint(self.X_test.shape[0])
            random_sample_test = "[{}],[{}]".format(self.X_test[random_choice_test], self.y_test[random_choice_test])
            output_message += '        Test  --> X:{}, y:{}, Sample: {}\n'.format(self.X_test.shape, self.y_test.shape, random_sample_test)
        if backup and logdir is not None:
            output_message += self.save(logdir)
        output_message += '-- BUILDING DATASET END -------------\n'
        print(output_message)

    def extract_dataset_specs(self, specs):
        if not isinstance(specs, str):
            if np.isnan(specs):
                return None
            else:
                assert False, "Dataset specifications should be a string or None: {}".format(specs)
        specs = ast.literal_eval(specs)
        if specs is not None:
            specs['distribution'] = list(list(specs.items())[0][1].items())[0][0]
            if specs['distribution'] == "E":
                lower = list(list(specs.items())[0][1].items())[0][1][0]
                upper = list(list(specs.items())[0][1].items())[0][1][1]
                distance = upper - lower
                specs['dataset_size'] = int(distance / list(list(specs.items())[0][1].items())[0][1][2]) + 1
            else:
                specs['dataset_size'] = list(list(specs.items())[0][1].items())[0][1][2]
        return specs

    def build_dataset(self, specs, max_iterations=1000, max_repeated_empty=100):
        """This function generates an (X,y) dataset by randomly sampling X
        values in a given range and calculating the corresponding y values.
        During generation it is checked that the generated datapoints are
        valid within the given range, removing nan and inf values. The
        generated dataset will be filled up to the desired dataset size or
        the function terminates with an error."""
        current_size = 0
        X_tmp = None
        y_tmp = None
        X = None
        y = None
        count_repeated_empty = 0
        count_iterations = 0
        while current_size < specs["dataset_size"]:
            if count_iterations > max_iterations:
                assert False, "Dataset creation taking too long. Got {} from {}".format(X_tmp.shape, specs)
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
            if not X_tmp is None:
                X = np.append(X, X_tmp, axis=0)
                y = np.append(y, y_tmp, axis=0)
            current_size = X.shape[0]
            # Handle "E" distributions
            if X.shape[0] != specs["dataset_size"] and specs['distribution'] == "E":
                assert False, "Equal distant data points cannot be created in the given range: {}".format(specs)
            X_tmp = X
            y_tmp = y
            count_iterations +=1
        assert X.shape[0] == specs["dataset_size"]
        if X.ndim == 1:
            X = X[:, np.newaxis]
        return X, y

    def get_data(self) -> Dict[str, Any]:
        return {'X_train':self.X_train, 'y_train':self.y_train, 'X_test':self.X_test, 'y_test':self.y_test, 'function_set':self.function_set}

    def remove_invalid(self, X, y, y_limit=100):
        """Removes nan, infs, and out of range datapoints from a dataset."""
        valid = np.logical_and(y > -y_limit, y < y_limit)
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
                start, stop, step = spec[input_var]["E"]
                if step > stop - start:
                    n = step
                else:
                    n = int((stop - start)/step) + 1
                feature = np.linspace(start=start, stop=stop, num=n, endpoint=True)
            else:
                raise ValueError("Did not recognize specification for {}: {}.".format(input_var, spec[input_var]))
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
        #Return numpy expression
        return lambda x : eval(s)

    def save(self, logdir='./'):
        """Saves the dataset to a specified location."""
        save_path = os.path.join(logdir,'data_{}_n{:.2f}_s{}.csv'.format(
                self.name, self.noise, self.seed))
        try:
            os.makedirs(logdir, exist_ok=True)
            np.savetxt(
                save_path,
                np.concatenate(
                    (
                        np.hstack((self.X_train, self.y_train[..., np.newaxis])),
                        np.hstack((self.X_test, self.y_test[..., np.newaxis]))
                    ), axis=0),
                delimiter=',', fmt='%1.5f'
            )
            return 'Saved dataset to               : {}\n'.format(save_path)
        except:
            import sys
            e = sys.exc_info()[0]
            print("WARNING: Could not save dataset: {}".format(e))

    def plot(self, logdir='./'):
        """Plot Dataset with underlying ground truth."""
        if self.X_train.shape[1] == 1:
            from matplotlib import pyplot as plt
            save_path = os.path.join(logdir,'plot_{}_n{:.2f}_s{}.png'.format(
                    self.name, self.noise, self.seed))

            # Draw ground truth expression
            bounds = list(list(self.train_spec.values())[0].values())[0][:2]
            x = np.linspace(bounds[0], bounds[1], endpoint=True, num=100)
            y = self.numpy_expr(x[:, None])
            plt.plot(x, y, color='red', linestyle='dashed')
            # Draw the actual points
            plt.scatter(self.X_train, self.y_train)
            # Add a title
            plt.title(
                "{} N:{} S:{}".format(
                    self.name, self.noise, self.seed),
                fontsize=7)
            try:
                os.makedirs(logdir, exist_ok=True)
                plt.savefig(save_path)
                print('Saved plot to                  : {}'.format(save_path))
            except:
                import sys
                e = sys.exc_info()[0]
                print("WARNING: Could not plot dataset: {}".format(e))
            plt.close()
        else:
            print("WARNING: Plotting only supported for 2D datasets.")
        
        
def load_burgers_equation():    
    descr = DatasetInfo(
        description = """
        Dataset for high-viscosity Burgers equation 
        ut=-uux+0.1uxx
        x∈[-8.0,8.0), t∈[0,10]
        nx=256, nt=201, u.shape=(256,201)
        Resource:DLGA-PDE: Discovery of PDEs with incomplete candidate library via combination of deep learning and genetic algorithm
        """
    )

    file_path = resources.files(DATA_MODULE) / "burgers2.mat"
    pde_data = load_mat_file(file_path)
    return PDEDataset(
        equation_name = 'burgers equation',
        descr = descr,
        pde_data = pde_data,
        domain = {'x': (-7.0, 7.0), 't': (1, 9)},
        epi = 1e-3
    )
    
def load_kdv_equation():
    descr = DatasetInfo(
        description = """
        Dataset for Korteweg-De Vries (KdV) equation with sin initial condition, actually a standardized form of Kdv_equation dataset
        ut=-uux-uxxx
        x∈[-20,20), t∈[0,40]
        nx=256, nt=201, u.shape=(256,201)
        Resource: PDE-READ: Human-readable Partial Differential Equation Discovery using Deep Learning, pp20
        """
    )

    file_path = resources.files(DATA_MODULE) / "KdV_equation.mat"
    pde_data = load_mat_file(file_path)

    return PDEDataset(
        equation_name = 'kdv equation',
        descr = descr,
        pde_data = None,
        x = pde_data['x'],
        t = pde_data['tt'],
        usol = pde_data['uu'],
        domain = {'x': (-16, 16), 't': (5, 35)},
        epi = 1e-3
    )


def load_pde_dataset(
    filename: str,
    equation_name: str = 'PDE Dataset',
    x_key: str = 'x',
    t_key: str = 't',
    u_key: str = 'usol',
    domain: dict = None,
    epi: float = 1e-3,
    data_dir_module: str = "kd.dataset.data"
):
    """
    一个通用的、用户友好的数据加载器，可以从指定的.mat文件加载PDE数据，
    并允许用户自定义所有关键元信息。

    Args:
        filename (str): 要加载的 .mat 文件的名称 (例如: "my_data.mat")。
        equation_name (str): 您为这个数据集赋予的名称 (例如: "My PDE")。
        x_key (str, optional): .mat 文件中代表空间坐标的键。默认为 'x'。
        t_key (str, optional): .mat 文件中代表时间坐标的键。默认为 't'。
        u_key (str, optional): .mat 文件中代表解的键。默认为 'usol'。
        domain (dict, optional): 定义分析子域，格式为 {'x':(min,max), 't':(min,max)}。默认为 None。
        epi (float, optional): 为该数据集推荐的稀疏性惩罚项。默认为 1e-3。
        data_dir_module (str, optional): 存储数据文件的模块路径。默认为 "kd.dataset.data"。

    Returns:
        一个功能完备的 PDEDataset 对象，可被所有 KD 模型使用。
    """
    try:
        # 1. 自动构建文件的完整路径
        file_path = resources.files(data_dir_module) / filename
        
        # 2. 使用底层的加载器读取 .mat 文件
        pde_data = load_mat_file(file_path)

        # 3. 使用用户指定的键名，从加载的字典中提取数据
        x_data = np.asarray(pde_data[x_key], dtype=float).flatten()
        t_data = np.asarray(pde_data[t_key], dtype=float).flatten()
        u_data = np.asarray(pde_data[u_key])

        # 确保 u_data 的形状与 (len(x), len(t)) 对齐
        if u_data.shape == (len(t_data), len(x_data)):
            u_data = u_data.T

        # 4. 将所有信息送入 PDEDataset 进行标准化封装
        dataset = PDEDataset(
            equation_name=equation_name,
            pde_data=None, 
            x=x_data,
            t=t_data,
            usol=u_data,
            domain=domain, 
            epi=epi
        )
        print(f"成功加载数据集: {equation_name}，文件: {filename}")
        return dataset

    except FileNotFoundError:
        print(f"错误: 在默认数据目录中未找到文件 {filename}")
        return None
    except KeyError as e:
        print(f"错误: 文件 {filename} 中缺少必需的键: {e}。请检查您传入的 x_key, t_key, u_key 参数是否正确。")
        return None
    except Exception as e:
        print(f"错误: 加载或处理文件时发生未知错误: {e}")
        return None
