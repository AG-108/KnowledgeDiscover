"""Re-evaluate saved SR/ODE expressions with protocol 2.1 recovery semantics.

This reads immutable per-case ``result.json`` files and writes a separate report;
it never rewrites the archived benchmark outputs or reruns a baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_benchmark as benchmark  # noqa: E402


DEFAULT_SOURCES = (
    ("formal", "fast_cpu_baselines_20260918", {"gplearn", "pyoperon", "pysindy", "pysr", "sindy"}),
    ("formal", "dscv_reduced_20260918", {"dscv"}),
    ("formal", "physo_full_20260918", {"physo"}),
    ("formal_partial", "symbolicgpt_full_20260918", {"symbolicgpt"}),
    ("formal", "full_benchmark_20260917/llmsr", {"llmsr"}),
    ("pilot", "full_benchmark_no_llmsr_20260917", {"dso"}),
    ("v2_validation", "v2_validation_ode_core", {"sindy"}),
)

RECORD_FIELDS = (
    "scope", "source", "name", "task", "model", "dataset", "instance", "target",
    "seed", "status", "recovery_eligible", "recovery_evaluable", "structural_recovery",
    "algebraic_exact_recovery", "coefficient_error", "exact_recovery_semantics",
    "expression", "evaluation_expression", "recovery_expression_provenance",
    "ground_truth_expression",
)

AGGREGATE_FIELDS = (
    "scope", "task", "model", "n_result_rows", "n_recovery_eligible", "n_successful_eligible",
    "n_recovery_evaluable", "parse_coverage_success", "n_structural_recovery",
    "structural_recovery_conditional_rate", "structural_recovery_lower_bound",
    "n_algebraic_exact_recovery", "n_algebraic_evaluable",
    "algebraic_exact_conditional_rate", "n_coefficient_error",
    "coefficient_error_median", "coefficient_error_mean", "coefficient_error_p90",
    "coefficient_error_max", "sources",
)


def saved_rows(results_root: Path):
    """Yield selected saved result rows without changing the source artifacts."""
    seen = set()
    for scope, relative, models in DEFAULT_SOURCES:
        directory = results_root / relative
        summary = directory / "benchmark_summary.csv"
        details = {}
        for path in sorted(directory.glob("cases/*/result.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            rows = value if isinstance(value, list) else [value]
            for row in rows:
                details[(row.get("name"), row.get("instance"), row.get("target"))] = row
        if not summary.is_file():
            continue
        with summary.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for summary_row in rows:
            if summary_row.get("model") not in models or summary_row.get("task") not in {"sr", "ode"}:
                continue
            detail = details.get(
                (summary_row.get("name"), summary_row.get("instance"), summary_row.get("target")),
                {},
            )
            row = {**summary_row, **detail}
            # Final-summary membership and status are authoritative; detail
            # rows only supply fields (notably variable_names) absent from CSV.
            row["status"] = summary_row.get("status")
            row["expression"] = summary_row.get("expression")
            row["ground_truth_expression"] = summary_row.get("ground_truth_expression")
            if row.get("model") == "physo":
                row["physo_log_path"] = (
                    directory / "cases" / row["name"] / "instance_0000" /
                    "target_00" / "physo.log"
                )
            row["exact_recovery"] = (
                None if summary_row.get("exact_recovery") in (None, "")
                else float(summary_row["exact_recovery"])
            )
            key = (relative, row.get("name"), row.get("instance"), row.get("target"))
            if key in seen:
                continue
            seen.add(key)
            yield scope, relative, row


def matched_physo_log_expression(log_path, display):
    """Recover the selected PhySO program only if its rendering matches exactly.

    PhySO returns the last program on its Pareto front. Its logger saves every
    epoch's candidates, rewards, rounded complexities and infix programs.
    Replaying that selection avoids guessing the meaning of two-dimensional
    pretty-printed fractions in archived result files.
    """
    if not log_path or not Path(log_path).is_file() or not display:
        return None
    import sympy as sp
    from sympy.parsing.sympy_parser import parse_expr

    best_by_complexity = {}
    try:
        with Path(log_path).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    complexity = round(float(row["complexity"]))
                    reward = float(row["reward"])
                    epoch = int(row["epoch"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not math.isfinite(reward):
                    continue
                previous = best_by_complexity.get(complexity)
                if previous is None or reward > previous[0] or (
                    reward == previous[0] and epoch > previous[1]
                ):
                    best_by_complexity[complexity] = (reward, epoch, row.get("program"))
    except OSError:
        return None

    selected, best_reward = None, 0.0
    for complexity in sorted(best_by_complexity):
        reward, _, program = best_by_complexity[complexity]
        if reward > best_reward:
            selected, best_reward = program, reward
    if not selected:
        return None
    try:
        symbolic = parse_expr(selected, evaluate=False)
        return selected if sp.pretty(symbolic, use_unicode=True) == display else None
    except (TypeError, ValueError, SyntaxError, sp.SympifyError):
        return None


def reevaluate(scope, source, row):
    truth = row.get("ground_truth_expression")
    eligible = bool(truth)
    expression = row.get("expression")
    expression_provenance = "archived_expression"
    if eligible and row.get("model") == "physo" and expression and (
        benchmark.expression_sympy(expression, row.get("variable_names") or ()) is None
    ):
        recovered = matched_physo_log_expression(row.get("physo_log_path"), expression)
        if recovered is not None:
            expression = recovered
            expression_provenance = "exact_display_match_to_physo_log"
    metrics = benchmark.expression_metrics(
        expression, truth, row.get("variable_names") or (), evaluate_algebraic=False
    ) if eligible and expression else {
        "exact_recovery": None,
        "structural_recovery": None,
        "algebraic_exact_recovery": None,
        "coefficient_error": None,
        "exact_recovery_semantics": "additive_term_structure_ignoring_numeric_coefficients",
    }
    return {
        "scope": scope,
        "source": source,
        "name": row.get("name"),
        "task": row.get("task"),
        "model": row.get("model"),
        "dataset": row.get("dataset"),
        "instance": row.get("instance"),
        "target": row.get("target"),
        "seed": row.get("seed"),
        "status": row.get("status"),
        "recovery_eligible": eligible,
        "recovery_evaluable": metrics.get("structural_recovery") is not None,
        "structural_recovery": metrics.get("structural_recovery"),
        "algebraic_exact_recovery": (
            row.get("algebraic_exact_recovery")
            if row.get("algebraic_exact_recovery") is not None
            else row.get("exact_recovery")
        ),
        "coefficient_error": metrics.get("coefficient_error"),
        "exact_recovery_semantics": metrics.get("exact_recovery_semantics"),
        "expression": row.get("expression"),
        "evaluation_expression": expression if expression != row.get("expression") else None,
        "recovery_expression_provenance": expression_provenance,
        "ground_truth_expression": truth,
    }


def aggregate(records):
    groups = {}
    for row in records:
        groups.setdefault((row["scope"], row["task"], row["model"]), []).append(row)
    output = []
    for (scope, task, model), rows in sorted(groups.items()):
        eligible = [row for row in rows if row["recovery_eligible"]]
        successful = [row for row in eligible if row["status"] == "ok"]
        evaluable = [row for row in successful if row["recovery_evaluable"]]
        structural = [row for row in evaluable if row["structural_recovery"] == 1]
        algebraic_evaluable = [
            row for row in successful if row["algebraic_exact_recovery"] is not None
        ]
        algebraic = [row for row in algebraic_evaluable if row["algebraic_exact_recovery"] == 1]
        errors = [row["coefficient_error"] for row in structural if row["coefficient_error"] is not None]
        sorted_errors = sorted(errors)
        output.append({
            "scope": scope,
            "task": task,
            "model": model,
            "n_result_rows": len(rows),
            "n_recovery_eligible": len(eligible),
            "n_successful_eligible": len(successful),
            "n_recovery_evaluable": len(evaluable),
            "parse_coverage_success": len(evaluable) / len(successful) if successful else None,
            "n_structural_recovery": len(structural),
            "structural_recovery_conditional_rate": len(structural) / len(evaluable) if evaluable else None,
            "structural_recovery_lower_bound": len(structural) / len(eligible) if eligible else None,
            "n_algebraic_exact_recovery": len(algebraic),
            "n_algebraic_evaluable": len(algebraic_evaluable),
            "algebraic_exact_conditional_rate": (
                len(algebraic) / len(algebraic_evaluable) if algebraic_evaluable else None
            ),
            "n_coefficient_error": len(errors),
            "coefficient_error_median": statistics.median(errors) if errors else None,
            "coefficient_error_mean": statistics.fmean(errors) if errors else None,
            "coefficient_error_p90": (
                sorted_errors[math.ceil(0.9 * len(sorted_errors)) - 1] if sorted_errors else None
            ),
            "coefficient_error_max": max(errors) if errors else None,
            "sources": ";".join(sorted({row["source"] for row in rows})),
        })
    return output


def percent(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def number(value):
    if value is None:
        return "—"
    return f"{value:.4g}"


def markdown(aggregates):
    lines = [
        "# SR/ODE recovery re-analysis (metric protocol 2.1)",
        "",
        "`exact_recovery` now means equality of expanded additive-term structure while ignoring numeric coefficients. "
        "`algebraic_exact_recovery` preserves the previous strict SymPy equality. Coefficient error is aligned relative L2 error and is reported only for structurally recovered expressions.",
        "",
        "Rates labelled conditional use successfully parsed eligible outputs. The lower bound divides structural recoveries by all rows with trusted ground truth, including errors, timeouts, and parser failures. Archived experiment files were not modified.",
        "Previously recorded strict algebraic recovery uses its own evaluable denominator; newly parsed display expressions are not silently counted as strictly evaluated.",
        "Archived PhySO programs are recovered from candidate logs only when their Unicode rendering exactly matches the saved display expression.",
        "",
    ]
    for scope in ("formal", "formal_partial", "pilot", "v2_validation"):
        rows = [row for row in aggregates if row["scope"] == scope]
        if not rows:
            continue
        lines.extend([
            f"## {scope}", "",
            "| Task | Model | Rows | Eligible | Parsed successful | Structural recovery | Lower bound | Previous strict algebraic | Coefficient error median / P90 / max |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in rows:
            lines.append(
                f"| {row['task']} | {row['model']} | {row['n_result_rows']} | "
                f"{row['n_recovery_eligible']} | {row['n_recovery_evaluable']}/"
                f"{row['n_successful_eligible']} | {row['n_structural_recovery']}/"
                f"{row['n_recovery_evaluable']} ({percent(row['structural_recovery_conditional_rate'])}) | "
                f"{percent(row['structural_recovery_lower_bound'])} | "
                f"{row['n_algebraic_exact_recovery']}/{row['n_algebraic_evaluable']} "
                f"({percent(row['algebraic_exact_conditional_rate'])}) | "
                f"{number(row['coefficient_error_median'])} / "
                f"{number(row['coefficient_error_p90'])} / "
                f"{number(row['coefficient_error_max'])} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--output-prefix", type=Path,
        default=ROOT / "results" / "recovery_reanalysis_v2_1",
    )
    args = parser.parse_args()

    records = [reevaluate(*item) for item in saved_rows(args.results_root)]
    aggregates = aggregate(records)
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps({"records": records, "aggregates": aggregates}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with prefix.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AGGREGATE_FIELDS)
        writer.writeheader()
        writer.writerows(aggregates)
    prefix.with_suffix(".md").write_text(markdown(aggregates), encoding="utf-8")
    print(json.dumps({"records": len(records), "aggregates": len(aggregates), "output": str(prefix)}, indent=2))


if __name__ == "__main__":
    main()
