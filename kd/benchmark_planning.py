"""Conservative per-slot queue estimates, independent of fit quality."""

from __future__ import annotations

from collections import defaultdict

from kd.benchmark_resources import slot_for_model


def case_budget(case, n_targets, policy):
    if policy.get("timeout") is None:
        raise ValueError("Finite T is required to bound container chunks")
    target = policy["setup_timeout"] + policy["timeout"] + policy["timeout_grace"] + policy["score_timeout"]
    # Include preparation even when a precomputed plan is reused, and the
    # separate ODE system-scoring worker's setup/score caps.
    system = policy["setup_timeout"] + policy["score_timeout"] if case["task"] == "ode" else 0
    return policy["setup_timeout"] + n_targets * target + system


def plan_chunks(cases, target_counts, policy, *, max_hours=40.0, overhead_hours=0.5):
    if not 0 < max_hours <= 40 or not 0 <= overhead_hours < max_hours:
        raise ValueError("Chunks must be <=40 hours with a non-negative, smaller overhead reserve")
    groups = defaultdict(list)
    for case in cases:
        count = target_counts[case["name"]]
        if not isinstance(count, int) or count < 1:
            raise ValueError("Every case needs a complete positive target count")
        slot = slot_for_model(case["model"])
        queue = "gpu_llmsr" if case["model"] == "llmsr" else slot
        seconds = case_budget(case, count, policy)
        groups[queue].append((case["name"], count, seconds))
    limit = (max_hours - overhead_hours) * 3600
    chunks, totals = [], {}
    for queue, entries in sorted(groups.items()):
        totals[queue] = {"n_cases": len(entries), "n_targets": sum(row[1] for row in entries),
                         "phase_budget_hours": sum(row[2] for row in entries) / 3600}
        bins = []
        # Deterministic first-fit decreasing; a case is never split between shards.
        for name, count, seconds in sorted(entries, key=lambda item: (-item[2], item[0])):
            if seconds > limit:
                raise ValueError(f"One case exceeds the chunk bound: {name}; reduce T or split its declared instance matrix")
            selected = next((entry for entry in bins if entry["seconds"] + seconds <= limit), None)
            if selected is None:
                selected = {"cases": [], "n_targets": 0, "seconds": 0}
                bins.append(selected)
            selected["cases"].append(name)
            selected["n_targets"] += count
            selected["seconds"] += seconds
        for index, entry in enumerate(bins):
            chunks.append({"chunk_id": f"{queue}-{index:04d}", "queue": queue,
                           "cases": sorted(entry["cases"]), "n_targets": entry["n_targets"],
                           "phase_budget_hours": entry["seconds"] / 3600,
                           "reserved_overhead_hours": overhead_hours,
                           "planned_hours": entry["seconds"] / 3600 + overhead_hours})
    return {"schema": "benchmark-queue-plan-v1", "max_chunk_hours": max_hours,
            "container_limit_hours": 48, "container_recovery_margin_hours": 48 - max_hours,
            "estimate_kind": "phase caps plus declared overhead reserve; not observed runtime",
            "unmeasured": ["checkpoint/file-system overhead", "cold service startup", "server-specific peak memory"],
            "slots": totals, "n_chunks": len(chunks), "chunks": chunks}
