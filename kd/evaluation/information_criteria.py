"""Information criteria with explicit statistical provenance."""

import math


def information_criteria(*, n_observations, n_free_parameters, log_likelihood=None,
                         residual_sum_squares=None, likelihood_model=None):
    """Return AIC/BIC only when parameter and likelihood provenance is declared.

    ``likelihood_model='gaussian_mle_variance'`` permits the usual RSS-based
    maximized Gaussian likelihood (up to its data-independent constant).  It is
    deliberately opt-in; expression token counts are never parameter counts.
    """
    if isinstance(n_observations, bool) or not isinstance(n_observations, int) or n_observations <= 0:
        raise ValueError("n_observations must be a positive integer")
    if isinstance(n_free_parameters, bool) or not isinstance(n_free_parameters, int) or n_free_parameters < 0:
        raise ValueError("n_free_parameters must be a non-negative integer")
    if log_likelihood is None:
        if likelihood_model != "gaussian_mle_variance" or residual_sum_squares is None:
            raise ValueError("declare log_likelihood or an explicit supported likelihood_model")
        rss = float(residual_sum_squares)
        if not math.isfinite(rss) or rss <= 0:
            raise ValueError("residual_sum_squares must be finite and positive")
        log_likelihood = -0.5 * n_observations * (
            math.log(2 * math.pi) + 1 + math.log(rss / n_observations)
        )
    log_likelihood = float(log_likelihood)
    if not math.isfinite(log_likelihood):
        raise ValueError("log_likelihood must be finite")
    variance_parameters = 1 if likelihood_model == "gaussian_mle_variance" else 0
    k = n_free_parameters + variance_parameters
    return {
        "aic": 2 * k - 2 * log_likelihood,
        "bic": math.log(n_observations) * k - 2 * log_likelihood,
        "information_criterion_n": n_observations,
        "information_criterion_k": k,
        "information_criterion_structural_k": n_free_parameters,
        "information_criterion_variance_parameters": variance_parameters,
        "likelihood_model": likelihood_model or "externally_supplied_log_likelihood",
    }
