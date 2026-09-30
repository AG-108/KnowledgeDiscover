"""Headerless, last-column-target CSVs with official pairs kept intact."""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PREFIX = "DeepSymbolicOptimization/dso/dso/task/regression/data/"
FILE_SR_NAMES = [f"Custom-{i}" for i in range(1, 24)] + ["Test-1", "PiecewiseFunction-1"]
FILE_GRAMMAR = ("add", "sub", "mul", "div", "sin", "cos", "exp", "log", "abs", "const")


class FileRegressionDataset:
    def __init__(self, name, train, test, metadata):
        self.registry_name = name
        self.X_train, self.y_train = train[:, :-1], train[:, -1]
        self.X_test, self.y_test = test[:, :-1], test[:, -1]
        self.variable_names = [f"x{i + 1}" for i in range(self.X_train.shape[1])]
        self.protocol_metadata = metadata
        self.target_name = "y"


def _read(path):
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 2 or not np.isfinite(data).all():
        raise ValueError("File SR requires >=2 finite numeric rows and >=2 columns")
    return data


def load_file_sr(name, *, root=None):
    label = name.removeprefix("dso_file_")
    if label not in FILE_SR_NAMES:
        raise ValueError("Unknown file SR candidate")
    root = Path(root) if root is not None else ROOT
    all_train = _read(root / (PREFIX + label + ".csv"))
    paths = [PREFIX + label + ".csv"]
    if label == "PiecewiseFunction-1":
        # Input-only interleaved 5 range blocks: block 2 held out; interpolation.
        order = np.argsort(all_train[:, 0], kind="stable")
        block = np.minimum(np.arange(len(order)) * 5 // len(order), 4)
        train, test = all_train[order[block != 2]], all_train[order[block == 2]]
        split = "input_rank_middle_fifth_holdout; interpolation"
    else:
        paths.append(PREFIX + label + "_test.csv")
        train, test = all_train, _read(root / paths[-1])
        split = "official_file_train_test_pair"
    if train.shape[1] != test.shape[1]:
        raise ValueError("Official CSV pair column counts differ")
    rows_train = {tuple(row) for row in train}
    overlap = sum(tuple(row) in rows_train for row in test)
    return FileRegressionDataset(
        name,
        train,
        test,
        {
            "source_paths": paths,
            "layout_evidence": PREFIX + "../regression.py: header=None, final column target",
            "split_protocol": split,
            "train_test_identical_rows": int(overlap),
            "overlap_policy": "retain official pair; disclose overlap, no independent-generalization claim if overlapping",
            "benchmark_function_set": list(FILE_GRAMMAR),
            "reference_status": "unknown",
            "units": "unknown",
            "target_definition": "final numeric CSV column",
            "train_input_ranges": np.column_stack(
                (train[:, :-1].min(0), train[:, :-1].max(0))
            ).tolist(),
            "test_input_ranges": np.column_stack(
                (test[:, :-1].min(0), test[:, :-1].max(0))
            ).tolist(),
        },
    )
