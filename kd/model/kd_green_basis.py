"""NumPy-only finite kernel basis diagnostic."""

import numpy as np


class GreenBasis:
    """Fixed monomials plus |x-xi|; fit coefficients from responses jointly."""

    def __init__(self, degree=2, rcond=1e-10):
        if not isinstance(degree, int) or degree < 0 or degree > 5 or not 0 < rcond < 1:
            raise ValueError("Finite basis requires degree 0..5 and 0<rcond<1")
        self.degree, self.rcond = degree, rcond

    def basis(self, inputs):
        x, xi = np.asarray(inputs).T
        powers = [(i, j) for i in range(self.degree + 1) for j in range(self.degree + 1 - i)]
        self.terms_ = [f"x**{i}*xi**{j}" for i, j in powers] + ["abs(x-xi)"]
        return np.stack([x**i * xi**j for i, j in powers] + [abs(x - xi)], axis=-1)

    def fit(self, training_view):
        self.shape_ = tuple(training_view["kernel_shape"])
        basis = self.basis(training_view["kernel_inputs"]).reshape(*self.shape_, -1)
        design = np.einsum("xyb,yf->xfb", basis, training_view["weighted_forcing"]).reshape(
            -1, basis.shape[-1]
        )
        target = training_view["response"].ravel()
        self.coef_, _, rank, singular = np.linalg.lstsq(design, target, rcond=self.rcond)
        self.diagnostics_ = {
            "response_design_rank": int(rank),
            "basis_size": len(self.coef_),
            "finite_basis_identifiable": bool(rank == len(self.coef_)),
            "condition": float(singular[0] / singular[-1]) if singular[-1] > 0 else None,
            "nonparametric_kernel_unique": False,
        }
        self.expression_ = " + ".join(
            f"({c:.12g})*({t})" for c, t in zip(self.coef_, self.terms_) if abs(c) > 1e-12
        )
        self.stop_reason_ = "budget_exhausted"
        return self

    def kernel(self, inputs):
        return (self.basis(inputs) @ self.coef_).reshape(self.shape_)
