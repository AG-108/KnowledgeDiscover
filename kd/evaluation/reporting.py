"""Bounded summaries that retain their actual case and seed denominators."""

import math
from collections import defaultdict


def _finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def seed_family_summary(rows, metric, *, family_key="family"):
    """Summarize per-seed values by family without pooling targets as seeds."""
    buckets = defaultdict(list)
    for row in rows:
        if _finite(row.get(metric)) and row.get("seed") is not None:
            buckets[(row.get("model"), row.get(family_key, row.get("task", "unknown")),
                     row.get("metric_protocol_version", "legacy/unknown"), row.get("profile"),
                     row.get("regime"), row["seed"])].append(
                float(row[metric])
            )
    by_family = defaultdict(list)
    case_counts = defaultdict(int)
    for key, values in buckets.items():
        group = key[:-1]
        by_family[group].append(sum(values) / len(values))
        case_counts[group] += len(values)
    result = []
    for group, seed_values in sorted(by_family.items(), key=repr):
        model, family, protocol, profile, regime = group
        mean = sum(seed_values) / len(seed_values)
        variance = sum((value - mean) ** 2 for value in seed_values) / (len(seed_values) - 1) if len(seed_values) > 1 else None
        result.append({"model": model, "family": family, "metric_protocol_version": protocol,
                       "profile": profile, "regime": regime, "metric": metric, "mean": mean,
                       "seed_std": math.sqrt(variance) if variance is not None else None,
                       "n_seeds": len(seed_values), "n_target_rows": case_counts[group],
                       "coverage_semantics": "available_case_mean; not unbiased for missing cases"})
    return result


def paired_comparison(rows, metric, model_a, model_b, *, pair_keys=("dataset", "instance", "target", "seed", "metric_protocol_version", "profile", "regime")):
    """Compare only exact shared target/seed pairs and expose dropped denominators."""
    selected = {model_a: {}, model_b: {}}
    candidate = {model_a: 0, model_b: 0}
    for row in rows:
        model = row.get("model")
        if model in selected and _finite(row.get(metric)):
            candidate[model] += 1
            key = tuple(row.get(key) for key in pair_keys)
            if key in selected[model]:
                raise ValueError(f"duplicate pairing key for {model}: {key}")
            selected[model][key] = float(row[metric])
    keys = sorted(set(selected[model_a]) & set(selected[model_b]), key=repr)
    differences = [selected[model_a][key] - selected[model_b][key] for key in keys]
    return {"metric": metric, "model_a": model_a, "model_b": model_b,
            "mean_difference_a_minus_b": sum(differences) / len(differences) if differences else None,
            "n_pairs": len(keys), "n_model_a_evaluable": candidate[model_a],
            "n_model_b_evaluable": candidate[model_b], "pair_keys": list(pair_keys)}
