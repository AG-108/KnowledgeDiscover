import numpy as np
import pytest

from kd.metrics import AIC, BIC, MSE, AdditiveParsimonyScore, ParsimonyInformationCriterion


def test_vanilla_metrics():
    metric = MSE()
    metric.reset()

    # Test with simple data
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.5, 2.5])

    result = metric(y_true, y_pred)

    expected_mse = ((0.0**2) + (0.5**2) + (0.5**2)) / 3  # MSE calculation
    assert np.isclose(result, expected_mse), f"Expected {expected_mse}, got {result}"

    print("MSE test passed.")


def _mse_of(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def test_aic():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])
    num_params = 2

    result = AIC(num_params=num_params)(y_true, y_pred)

    n = len(y_true)
    mse = _mse_of(y_true, y_pred)
    expected = n * np.log(mse) + 2 * num_params
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"


def test_bic():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])
    num_params = 2

    result = BIC(num_params=num_params)(y_true, y_pred)

    n = len(y_true)
    mse = _mse_of(y_true, y_pred)
    expected = n * np.log(mse) + num_params * np.log(n)
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"


def test_bic_penalizes_params_more_than_aic_for_larger_n():
    # BIC's ln(n) penalty should exceed AIC's constant 2 penalty once n is
    # large enough (n > e^2 ~= 7.39), so BIC should score the same fit worse.
    y_true = np.linspace(0.0, 1.0, 50)
    y_pred = y_true + 0.1
    num_params = 3

    aic = AIC(num_params=num_params)(y_true, y_pred)
    bic = BIC(num_params=num_params)(y_true, y_pred)

    assert bic > aic


def test_parsimony_information_criterion():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])
    complexity = 5
    physics_penalty = 0.2
    lambda_complexity = 0.5
    lambda_physics = 2.0

    with pytest.warns(DeprecationWarning, match="legacy"):
        metric = ParsimonyInformationCriterion(
            complexity=complexity,
            physics_penalty=physics_penalty,
            lambda_complexity=lambda_complexity,
            lambda_physics=lambda_physics,
        )
    result = metric(y_true, y_pred)

    n = len(y_true)
    mse = _mse_of(y_true, y_pred)
    expected = n * np.log(mse) + lambda_complexity * complexity + lambda_physics * physics_penalty
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"


def test_additive_parsimony_score_has_honest_metric_name():
    metric = AdditiveParsimonyScore(complexity=1.0)
    assert metric.config.name == "additive_parsimony_v1"


def test_information_criteria_accumulate_across_update_calls():
    # AIC/BIC/PIC reuse MetaMetrics' streaming update()/compute(): the MSE
    # ingredient should accumulate across multiple update() calls just like
    # plain MSE does, before the complexity penalty is applied once at the end.
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 5.0])
    num_params = 1

    streaming = AIC(num_params=num_params)
    streaming.reset()
    streaming.update(y_true[:2], y_pred[:2])
    streaming.update(y_true[2:], y_pred[2:])
    streaming_result = streaming.compute()

    one_shot_result = AIC(num_params=num_params)(y_true, y_pred)

    assert np.isclose(streaming_result, one_shot_result)


def test_aic_and_bic_are_abstract_base_instances_rejected():
    # AIC/BIC/ParsimonyInformationCriterion all require model-complexity
    # info that only makes sense per fitted model, so the shared
    # _InformationCriterionBase must not be directly usable.
    from kd.metrics import _InformationCriterionBase

    with pytest.raises(TypeError):
        _InformationCriterionBase()
