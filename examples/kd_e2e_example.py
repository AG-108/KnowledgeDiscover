"""Explicitly configured official-checkpoint example; never downloads assets."""

import argparse
import numpy as np
from kd.model import E2ETransformerModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--trust-checkpoint", action="store_true")
    args = parser.parse_args()
    X = np.random.default_rng(0).normal(size=(100, 2))
    y = np.cos(X[:, 0]) + X[:, 1] ** 2
    model = E2ETransformerModel(**vars(args)).fit(X, y)
    print(model.best_expression_)
    print(model.provenance_)


if __name__ == "__main__":
    main()
