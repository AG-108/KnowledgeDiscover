import pytest

from kd.evaluation.physical_credibility import (
    CheckDefinition,
    DatasetCard,
    build_judge_payload,
    evaluate_physical_credibility,
    run_mock_judge,
)


def _card(names=("v", "x", "t", "a")):
    v, x, t, a = names
    return DatasetCard(
        dataset_id="kinematics-demo",
        card_version="reviewed-1",
        physical_setting="one-dimensional motion on the declared interval",
        unit_basis=("L", "T"),
        symbol_units={v: (1, -1), x: (1, 0), t: (0, 1), a: (1, -2)},
        declared_assumptions=("forward_time",),
        provenance=("synthetic test fixture; not expert validation",),
        checks=(
            CheckDefinition(
                "units", "dimensional", "homogeneous equation", "dataset-card", "1",
                "problem", "units supplied for every symbol",
            ),
            CheckDefinition(
                "finite", "domain", "real finite expression", "dataset-card", "1",
                "problem", "fixed domain", domain={x: (1.0, 2.0), t: (0.1, 1.0)},
                parameters={"target": "rhs"}, variables=(x, t),
            ),
            CheckDefinition(
                "time-sign", "sign", "nonnegative time", "dataset-card", "1",
                "derived", "forward-time regime", assumptions=("forward_time",),
                domain={t: (0.0, 1.0)}, parameters={"expression": t, "relation": "nonnegative"},
            ),
        ),
    )


def test_dimensional_checks_pass_fail_and_unknown_coefficient_units():
    card = _card()
    assert evaluate_physical_credibility("v = x/t", card).evidence[0].status == "pass"
    assert evaluate_physical_credibility("v = x*t", card).evidence[0].status == "fail"
    report = evaluate_physical_credibility("v = c*x/t", card)
    assert report.evidence[0].status == "unknown"
    assert report.evidence[0].observations["unknown_units"] == ("c",)


def test_domain_and_sign_checks_are_conditional_and_real():
    domain = CheckDefinition(
        "log-domain", "domain", "real logarithm", "fixture", "1", "problem",
        "positive coordinate", domain={"x": (-1.0, 1.0)},
        parameters={"expression": "log(x)"},
    )
    report = evaluate_physical_credibility(
        "v = x/t", DatasetCard("d", "1", "setting", (domain,), symbol_units={})
    )
    assert report.evidence[0].status == "fail"
    assert report.evidence[0].observations["invalid_examples"]

    missing_assumption = DatasetCard(
        "d", "1", "setting",
        (CheckDefinition("s", "sign", "sign", "fixture", "1", "derived", "regime",
                         assumptions=("dissipative",), domain={"x": (0, 1)},
                         parameters={"expression": "x", "relation": "nonnegative"}),),
    )
    assert evaluate_physical_credibility("x = x", missing_assumption).evidence[0].status == "not_applicable"


def test_order_swap_and_consistent_variable_renaming_do_not_change_outcomes():
    first = _card()
    swapped = DatasetCard(first.dataset_id, first.card_version, first.physical_setting,
                          tuple(reversed(first.checks)), first.unit_basis, first.symbol_units,
                          first.declared_assumptions, first.provenance)
    a = evaluate_physical_credibility("v = x/t", first)
    b = evaluate_physical_credibility("v = x/t", swapped)
    assert {e.check_id: e.status for e in a.evidence} == {e.check_id: e.status for e in b.evidence}

    renamed = evaluate_physical_credibility("speed = distance/time", _card(("speed", "distance", "time", "accel")))
    assert [e.status for e in a.evidence] == [e.status for e in renamed.evidence]


def test_coverage_keeps_unknown_separate_from_failure():
    report = evaluate_physical_credibility("v = c*x/t", _card())
    assert report.coverage == {"pass": 1, "fail": 0, "unknown": 2,
                               "not_applicable": 0, "evaluation_error": 0,
                               "total": 3, "resolved": 1}


def test_judge_payload_is_blinded_hashed_frozen_and_cannot_override_evidence():
    report = evaluate_physical_credibility("v = x/t", _card())
    payload = build_judge_payload(report)
    assert "method_identity" not in payload and "ground_truth" not in payload
    assert len(payload["payload_hash"]) == 64
    with pytest.raises(TypeError):
        payload["equation"] = "tampered"

    judged = run_mock_judge(report, lambda _: {"assessment": "conditional-pass"}, "offline-mock", "1")
    assert judged["response"]["assessment"] == "conditional-pass"
    with pytest.raises(ValueError, match="override"):
        run_mock_judge(report, lambda _: {"evidence": []}, "offline-mock", "1")


def test_hypothesis_cannot_silently_become_a_gate():
    with pytest.raises(ValueError, match="hypotheses"):
        CheckDefinition("h", "sign", "guess", "author", "1", "hypothesis", "", gate=True)


def test_candidate_parser_rejects_python_calls_and_payload_has_physical_context():
    report = evaluate_physical_credibility("v = __import__('os').getcwd()", _card())
    assert all(record.status == "evaluation_error" for record in report.evidence)
    payload = build_judge_payload(evaluate_physical_credibility("v = x/t", _card()))
    assert "dataset_id" not in payload["card"]
    assert payload["physical_context"]["physical_setting"]
    assert len(payload["physical_context"]["check_definitions"]) == 3


def test_inputs_are_deeply_frozen_and_caller_mutation_does_not_change_hash():
    nested = {"target": "rhs", "metadata": {"reviewers": ["one"]}}
    domain = {"x": [0.0, 1.0]}
    checks = [CheckDefinition("d", "domain", "finite", "fixture", "1", "problem", "",
                              variables=["x"], domain=domain, parameters=nested)]
    units = {"x": [1.0]}
    card = DatasetCard("d", "1", "setting", checks, unit_basis=["L"], symbol_units=units)
    original = card.fingerprint
    nested["metadata"]["reviewers"].append("two")
    domain["x"][0] = -99
    units["x"][0] = 99
    checks.clear()
    assert card.fingerprint == original
    assert card.checks[0].parameters["metadata"]["reviewers"] == ("one",)
    with pytest.raises(TypeError):
        card.checks[0].parameters["new"] = 1


def test_known_dimensional_contradictions_fail_but_literal_policy_abstains():
    card = _card()
    assert evaluate_physical_credibility("v = x + t", card).evidence[0].status == "fail"
    assert evaluate_physical_credibility("v = sin(x)", card).evidence[0].status == "fail"
    unknown = evaluate_physical_credibility("v = 2*x/t", card)
    assert unknown.evidence[0].status == "unknown"
    dimensionless = DatasetCard(card.dataset_id, card.card_version, card.physical_setting,
                                card.checks, card.unit_basis, card.symbol_units,
                                card.declared_assumptions, card.provenance, "dimensionless")
    assert evaluate_physical_credibility("v = 2*x/t", dimensionless).evidence[0].status == "pass"


def test_rhs_targets_candidate_domain_and_sign_and_preserves_expression_holes():
    checks = (
        CheckDefinition("domain", "domain", "rhs finite", "fixture", "1", "problem", "",
                        variables=("x",), domain={"x": (0, 1)},
                        parameters={"target": "rhs"}),
        CheckDefinition("sign", "sign", "rhs nonpositive", "fixture", "1", "derived", "",
                        variables=("x",), domain={"x": (0, 1)},
                        parameters={"target": "rhs", "relation": "nonpositive"}),
    )
    card = DatasetCard("d", "1", "setting", checks, symbol_units={})
    good = evaluate_physical_credibility("y = -x", card)
    assert [item.status for item in good.evidence] == ["pass", "pass"]
    assert evaluate_physical_credibility("y = x", card).evidence[1].status == "fail"
    hole = evaluate_physical_credibility("y = x/x", card)
    assert hole.evidence[0].status == "fail"


def test_declared_check_symbol_can_be_absent_from_candidate_and_probes_are_bounded():
    check = CheckDefinition("d", "domain", "declared expression", "fixture", "1", "problem", "",
                            variables=("z", "x"), domain={"z": (0, 1), "x": (0, 1)}, budget=2,
                            parameters={"expression": "z + x"})
    report = evaluate_physical_credibility("y = x", DatasetCard("d", "1", "s", (check,)))
    assert report.evidence[0].status == "pass"
    assert report.evidence[0].observations["probes_evaluated"] == 2
    assert report.evidence[0].observations["continuous_domain_proof"] is False


def test_parse_failure_is_evidence_and_equivalent_candidate_order_is_invariant():
    broken = evaluate_physical_credibility("not an equation", _card())
    assert all(item.status == "evaluation_error" for item in broken.evidence)

    candidates = ("v = x/t", "t*v = x")
    forward = {candidate: [e.status for e in evaluate_physical_credibility(candidate, _card()).evidence]
               for candidate in candidates}
    reverse = {candidate: [e.status for e in evaluate_physical_credibility(candidate, _card()).evidence]
               for candidate in reversed(candidates)}
    assert forward == reverse
    assert forward[candidates[0]] == forward[candidates[1]]
