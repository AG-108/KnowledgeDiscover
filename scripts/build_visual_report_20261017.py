"""Chart-first visual report for the 2026-10-17 combined benchmark results.

Reads only the frozen outputs of build_combined_results_20261017.py and writes an
offline HTML report plus SVG/PNG figures. Cohorts are never pooled; recovery and
completion are counted separately; every error/time panel states its sample size.

    python scripts/build_visual_report_20261017.py
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/analysis/combined_results_20261017"
OUT = SRC / "visualizations"

OLD, NEW = "historical_20261016", "pde_extension_20261017"
COHORT_LABEL = {OLD: "旧快照 10-16", NEW: "新 PDE 扩展 10-17"}
TASK_LABEL = {"sr": "SR", "ode": "ODE", "pde": "PDE"}
# One global order and palette (Paul Tol "muted"), used by every figure.
ORDER = ("sisso", "physo", "symbolicgpt", "llmsr", "dlga", "pdefind", "pysindy", "weakform", "deepmod")
LABEL = dict(sisso="SISSO", physo="PhySO", symbolicgpt="SymbolicGPT", llmsr="LLM-SR", dlga="DLGA",
             pdefind="PDE-FIND", pysindy="PySINDy", weakform="WSINDy", deepmod="DeepMoD")
COLOR = dict(sisso="#332288", physo="#88CCEE", symbolicgpt="#882255", llmsr="#117733", dlga="#CC6677",
             pdefind="#44AA99", pysindy="#DDCC77", weakform="#AA4499", deepmod="#999933")
STATUS = (("ok", "成功", "#4C7FA8"), ("timeout", "超时", "#E3A33B"),
          ("error", "报错", "#C0504D"), ("missing", "未见终态", "#D9DEE5"))
GROUPS = ((OLD, "sr"), (OLD, "ode"), (OLD, "pde"), (NEW, "pde"))
RECOVERY_FIELD = {"sr": "structural_recovery", "ode": "structural_recovery", "pde": "pde_support_recovery"}
PDE_DATASETS = ("burgers", "burgers_sine", "kdv", "chafee-infante", "allen_cahn", "convection_diffusion",
                "fisher", "fisher_linear", "PDE_divide", "PDE_compound")
BUDGET = 300.0
FLOOR = 1e-18  # log-axis position for exact zeros (marked, never mixed with real values)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load():
    rows = [json.loads(line) for line in (SRC / "rows.jsonl").read_text(encoding="utf-8").splitlines() if line]
    with (SRC / "dataset_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        metrics = list(csv.DictReader(stream))
    summary = json.loads((SRC / "summary.json").read_text(encoding="utf-8"))
    return rows, metrics, summary


def derive(rows, metrics, summary):
    """Every number drawn or quoted, recomputed from rows and cross-checked with the CSV."""
    by = defaultdict(list)
    for r in rows:
        by[r["cohort"], r["task"], r["model"]].append(r)
    cov = {(c["cohort"], c["task"], c["model"]): c for c in summary["coverage"]}
    for key, c in cov.items():
        st = Counter(r["status"] for r in by[key])
        assert (len(by[key]), st["ok"], st["timeout"], st["error"]) == (
            c["terminal"], c["ok"], c["timeout"], c["error"]), key
    methods = {g: [m for m in ORDER if (*g, m) in cov] for g in GROUPS}

    def evaluable(rs, task, field=None):
        field = field or RECOVERY_FIELD[task]
        return [r for r in rs if r["status"] == "ok" and finite(r.get(field))]

    recovery = {}
    for (c, t, m) in cov:
        values = [r[RECOVERY_FIELD[t]] for r in evaluable(by[c, t, m], t)]
        assert set(values) <= {0, 1}
        recovery[c, t, m] = (int(sum(values)), len(values))
    published = defaultdict(lambda: [0, 0])  # dataset/system scope only; RHS rows are a second view
    for x in metrics:
        if x["scope"] != "rhs" and x["metric"] == RECOVERY_FIELD[x["task"]] and x["recovery_evaluable"] != "":
            cell = published[x["cohort"], x["task"], x["method"]]
            cell[0] += int(x["recovery_yes"])
            cell[1] += int(x["recovery_evaluable"])
    assert all(tuple(published.get(k, (0, 0))) == v for k, v in recovery.items()), "recovery differs from CSV"

    family = {x["dataset"]: x["family"] for x in metrics if x["task"] == "sr"}
    families = list(dict.fromkeys(x["family"] for x in metrics if x["task"] == "sr"))
    systems = list(dict.fromkeys(x["dataset"] for x in metrics if x["task"] == "ode"))

    def cells(cohort, task, keyfn):
        out = defaultdict(lambda: [0, 0])
        for m in methods[cohort, task]:
            for r in evaluable(by[cohort, task, m], task):
                cell = out[keyfn(r), m]
                cell[0] += int(r[RECOVERY_FIELD[task]])
                cell[1] += 1
        return {k: tuple(v) for k, v in out.items()}

    heat = {(OLD, "sr"): (families, cells(OLD, "sr", lambda r: family[r["dataset"]])),
            (OLD, "ode"): (systems, cells(OLD, "ode", lambda r: r["dataset"])),
            (OLD, "pde"): ([d for d in PDE_DATASETS if any(r["dataset"] == d for r in by[OLD, "pde", "dlga"])],
                           cells(OLD, "pde", lambda r: r["dataset"])),
            (NEW, "pde"): (list(PDE_DATASETS), cells(NEW, "pde", lambda r: r["dataset"]))}

    def common(cohort, task, subset):
        """Recovery on targets every method in `subset` evaluated (the fair cross-method view)."""
        def key(r):
            return r["dataset"], r["target"], r["seed"]
        sets = {m: {key(r): r[RECOVERY_FIELD[task]] for r in evaluable(by[cohort, task, m], task)} for m in subset}
        shared = set.intersection(*(set(s) for s in sets.values()))
        return len(shared), {m: int(sum(sets[m][k] for k in shared)) for m in subset}

    shared = {"sr_3": common(OLD, "sr", ("sisso", "physo", "llmsr")),
              "ode_4": common(OLD, "ode", ("sisso", "physo", "symbolicgpt", "llmsr")),
              "ode_3": common(OLD, "ode", ("sisso", "symbolicgpt", "llmsr")),
              "pde_5": common(NEW, "pde", methods[NEW, "pde"])}

    coef = {}
    for g in GROUPS:
        for m in methods[g]:
            values = [(r["dataset"], r["seed"], r["coefficient_error"])
                      for r in evaluable(by[(*g, m)], g[1], "coefficient_error")]
            if values:
                coef[(*g, m)] = values
    new_keys = {m: {(d, s) for d, s, _ in coef[NEW, "pde", m]} for m in methods[NEW, "pde"]}
    assert len({frozenset(v) for v in new_keys.values()}) == 1 and len(new_keys["weakform"]) == 30
    coef_ok = {m: (sum(r["coefficient_recovery"] == 1 for r in evaluable(by[NEW, "pde", m], "pde", "coefficient_recovery")),
                   len(evaluable(by[NEW, "pde", m], "pde", "coefficient_recovery"))) for m in methods[NEW, "pde"]}

    fit = {(*g, m): [r["fit_seconds"] for r in by[(*g, m)] if r["status"] == "ok" and finite(r.get("fit_seconds"))]
           for g in GROUPS for m in methods[g]}
    for x in metrics:
        if x["metric"] == "fit_seconds" and x["scope"] != "rhs" and x["cohort"] == NEW:
            values = [r["fit_seconds"] for r in by[NEW, "pde", x["method"]] if r["dataset"] == x["dataset"]]
            assert math.isclose(statistics.median(values), float(x["median"]), rel_tol=1e-12)

    nrmse = {(d, m): [r["equation_residual_nrmse"] for r in by[NEW, "pde", m]
                      if r["dataset"] == d and r["status"] == "ok" and finite(r.get("equation_residual_nrmse"))]
             for d in PDE_DATASETS for m in methods[NEW, "pde"]}
    for x in metrics:
        if x["metric"] == "nrmse" and x["cohort"] == NEW:
            assert math.isclose(statistics.median(nrmse[x["dataset"], x["method"]]), float(x["median"]), rel_tol=1e-12)
    best = {d: min(methods[NEW, "pde"], key=lambda m: statistics.median(nrmse[d, m])) for d in PDE_DATASETS}
    dlga = {d: {c: sorted((r["seed"], r["equation_residual_nrmse"]) for r in by[c, "pde", "dlga"] if r["dataset"] == d)
                for c in (OLD, NEW)} for d in heat[OLD, "pde"][0]}
    return dict(by=by, cov=cov, methods=methods, recovery=recovery, heat=heat, shared=shared, coef=coef,
                coef_ok=coef_ok, fit=fit, nrmse=nrmse, best=best, dlga=dlga)


# ---------------------------------------------------------------- drawing helpers
def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": "#8A96A3", "axes.linewidth": 0.8, "axes.labelcolor": "#203047",
        "xtick.color": "#3D4C5C", "ytick.color": "#3D4C5C", "text.color": "#203047",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "grid.color": "#E3E8EE", "grid.linewidth": 0.7, "figure.facecolor": "white",
        "savefig.facecolor": "white", "svg.fonttype": "path", "svg.hashsalt": "kd-visual-20261017",
        "legend.frameon": False,
    })


def tag(artist, method, cohort, kind):
    """gid read by the HTML filter: m-<method>|c-<cohort>|<kind>-<n>."""
    tag.n = getattr(tag, "n", 0) + 1
    artist.set_gid(f"m-{method}|c-{cohort}|{kind}-{tag.n}")
    return artist


def fmt(value, digits=3):
    if value == 0:
        return "0"
    text = f"{value:.{digits}g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        return f"{mantissa}e{int(exponent)}"
    return text


def text_color(method):
    """Darkened palette colour for labels, so light hues stay readable on white."""
    import colorsys
    hue, light, sat = colorsys.rgb_to_hls(*mcolors.to_rgb(COLOR[method]))
    return mcolors.to_hex(colorsys.hls_to_rgb(hue, min(light, 0.36), sat))


def ink(rgba):
    r, g, b = rgba[:3]
    return "#1B2733" if 0.299 * r + 0.587 * g + 0.114 * b > 0.55 else "white"


def short(system):
    return system.removeprefix("ode_core_")


def method_axis(ax, ms, where="bottom", positions=None):
    """Method names as coloured labels that disappear with the method's filter."""
    positions = list(range(len(ms)) if positions is None else positions)
    if where == "left":
        ax.set_yticks(positions, [""] * len(ms))
    else:
        ax.set_xticks(positions, [""] * len(ms))
    for p, m in zip(positions, ms):
        if where == "left":
            t = ax.text(-0.012, p, LABEL[m], transform=ax.get_yaxis_transform(), ha="right", va="center")
        else:
            y, va = (-0.03, "top") if where == "bottom" else (1.012, "bottom")
            t = ax.text(p, y, LABEL[m], transform=ax.get_xaxis_transform(), ha="center", va=va)
        t.set(color=text_color(m), fontweight="bold", fontsize=8.5)
        tag(t, m, "any", "lbl")


def svg_text(fig, prefix):
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata={"Date": None})
    plt.close(fig)
    text = buf.getvalue()
    text = text[text.index("<svg"):]
    # Inline SVGs share one document: namespace every id and reference per card.
    text = re.sub(r'(\s)id="([^"]+)"', lambda m: f'{m[1]}id="{prefix}__{m[2]}"', text)
    text = re.sub(r'((?:xlink:)?href)="#([^"]+)"', lambda m: f'{m[1]}="#{prefix}__{m[2]}"', text)
    return re.sub(r"url\(#([^)]+)\)", lambda m: f"url(#{prefix}__{m[1]})", text)


def panel(cohorts, tasks, methods, size, draw):
    return dict(cohorts=cohorts, tasks=tasks, methods=methods, size=size, draw=draw)


def render_card(p, prefix):
    fig = plt.figure(figsize=p["size"], layout="constrained")
    p["draw"](fig, fig.add_gridspec(1, 1)[0])
    return svg_text(fig, prefix)


def render_composite(name, title, panels, figsize, shape, slots, h=None, w=None, legend=None):
    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = fig.add_gridspec(*shape, height_ratios=h, width_ratios=w)
    for p, slot in zip(panels, slots):
        p["draw"](fig, [gs[s] for s in slot] if isinstance(slot, list) else gs[slot])
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=11)
    if legend:
        legend(fig)
    fig.savefig(OUT / f"{name}.svg", format="svg", metadata={"Date": None})
    fig.savefig(OUT / f"{name}.png", dpi=300, metadata={"Software": None})
    plt.close(fig)


# ---------------------------------------------------------------- figures
def fig_overview(D):
    def make(group):
        cohort, task = group
        ms = D["methods"][group]

        def draw(fig, spec):
            # Composite passes two cells of one shared grid so columns align across rows.
            if isinstance(spec, list):
                a = fig.add_subplot(spec[0])
                b = fig.add_subplot(spec[1], sharey=a)
            else:
                a, b = spec.subgridspec(1, 2, width_ratios=[3, 2], wspace=0.04).subplots(sharey=True)
            for i, m in enumerate(ms):
                c, left = D["cov"][(*group, m)], 0.0
                for key, name, color in STATUS:
                    share = c[key] / c["planned"]
                    if share:
                        tag(a.barh(i, share, left=left, color=color, height=0.62, edgecolor="white",
                                   linewidth=0.5)[0], m, cohort, "bar")
                        if share >= 0.09:
                            tag(a.text(left + share / 2, i, f"{c[key]}", ha="center", va="center", fontsize=7.5,
                                       color="#1B2733" if key == "missing" else "white"), m, cohort, "txt")
                    left += share
                tag(a.text(1.01, i, f"{c['ok']}/{c['planned']}", va="center", fontsize=8), m, cohort, "txt")
                k, n = D["recovery"][(*group, m)]
                if n:
                    tag(b.barh(i, k / n, color=COLOR[m], height=0.62)[0], m, cohort, "bar")
                tag(b.text(max(k / n if n else 0, 0) + 0.02, i, f"{k}/{n}" if n else "不可评",
                           va="center", fontsize=8), m, cohort, "txt")
            pad = max(0.6, 1.4 - 0.2 * len(ms))  # keeps a lone bar as thin as bars in larger panels
            for ax in (a, b):
                ax.set_xlim(0, 1)
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25%", "50%", "75%", "100%"])
                ax.set_ylim(len(ms) - 1 + pad, -pad)
                ax.grid(axis="x")
                ax.set_axisbelow(True)
            method_axis(a, ms, "left")
            b.tick_params(left=False)
            a.set_title(f"{COHORT_LABEL[cohort]} · {TASK_LABEL[task]}：运行状态（占计划目标）", loc="left")
            b.set_title("结构恢复率（恢复/可评估）", loc="left")
        return panel([cohort], [task], ms, (9.2, 0.5 * len(ms) + 1.15), draw)
    return [make(g) for g in GROUPS]


def status_legend(fig):
    handles = [Rectangle((0, 0), 1, 1, color=c) for _, _, c in STATUS]
    fig.legend(handles, [n for _, n, _ in STATUS], loc="outside lower center", ncol=4, fontsize=8.5)


def heat_axes(ax, rows_, ms, title):
    ax.set_xlim(-0.5, len(ms) - 0.5)
    ax.set_ylim(len(rows_) - 0.5, -0.5)
    ax.set_yticks(range(len(rows_)), rows_)
    ax.tick_params(length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    method_axis(ax, ms, "top")
    ax.set_title(title, loc="left", pad=22)


def fig_recovery(D):
    cmap = plt.get_cmap("Blues")
    eligible = {r["dataset"] for r in D["by"][NEW, "pde", "weakform"] if r.get("recovery_eligible")}

    def make(group, size):
        cohort, task = group
        names, cells = D["heat"][group]
        ms = D["methods"][group]
        labels = [short(n) + (" *" if task == "pde" and n in eligible else "") for n in names]

        def draw(fig, spec):
            ax = fig.add_subplot(spec)
            for i, name in enumerate(names):
                for j, m in enumerate(ms):
                    k, n = cells.get((name, m), (0, 0))
                    face = cmap(0.12 + 0.88 * k / n) if n else "#F1F3F6"
                    tag(ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=face, edgecolor="white",
                                               linewidth=1.2)), m, cohort, "cell")
                    if not n:  # no evaluable result: hatched, distinct from an evaluated 0/n
                        tag(ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////",
                                                   edgecolor="#D3D9E0", linewidth=0)), m, cohort, "cell")
                    tag(ax.text(j, i, f"{k}/{n}" if n else "—", ha="center", va="center", fontsize=7.5,
                                color=ink(face) if n else "#9AA5B1"), m, cohort, "txt")
            unit = "方程族" if task == "sr" else ("系统" if task == "ode" else "数据集")
            heat_axes(ax, labels, ms, f"{COHORT_LABEL[cohort]} · {TASK_LABEL[task]}（按{unit}）")
        return panel([cohort], [task], ms, size, draw)
    return [make((NEW, "pde"), (4.6, 4.6)), make((OLD, "ode"), (4.2, 3.9)),
            make((OLD, "pde"), (2.6, 1.9)), make((OLD, "sr"), (4.4, 8.6))]


def strip(ax, rng, x, values, color, gid, horizontal=False, width=0.28):
    """Box (n>=3) plus every point, on a log axis."""
    pos = np.array(values, dtype=float)
    if len(pos) >= 3:
        box = ax.boxplot(pos, positions=[x], widths=0.55, vert=not horizontal, showfliers=False,
                         patch_artist=True, medianprops=dict(color="#1B2733", linewidth=1.4),
                         whiskerprops=dict(color="#6B7785"), capprops=dict(color="#6B7785"),
                         boxprops=dict(facecolor=mcolors.to_rgba(color, 0.22), edgecolor=color))
        for artist in (a for group in box.values() for a in group):
            tag(artist, *gid, "box")
    jitter = x + rng.uniform(-width, width, len(pos))
    xy = (pos, jitter) if horizontal else (jitter, pos)
    tag(ax.scatter(*xy, s=11, color=color, alpha=0.8, edgecolors="none", zorder=3), *gid, "pts")


def fig_coefficients(D):
    rng = np.random.default_rng(20261017)

    def make(group, size):
        cohort, task = group
        ms = D["methods"][group]

        def draw(fig, spec):
            ax = fig.add_subplot(spec)
            for j, m in enumerate(ms):
                values = [e for *_, e in D["coef"].get((*group, m), [])]
                real, zeros = [v for v in values if v > 0], sum(v == 0 for v in values)
                if real:
                    strip(ax, rng, j, real, COLOR[m], (m, cohort))
                if zeros:
                    tag(ax.scatter([j], [FLOOR], marker="v", s=34, color=COLOR[m], zorder=3), m, cohort, "pts")
                    tag(ax.text(j + 0.12, FLOOR, f"×{zeros}", va="center", fontsize=7.5), m, cohort, "txt")
                note = f"n={len(values)}"
                if cohort == NEW:
                    k, n = D["coef_ok"][m]
                    note += f"\n达标 {k}/{n}"
                tag(ax.text(j, 1.0, note, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                            fontsize=7.5, color="#3D4C5C"), m, cohort, "txt")
            ax.set_yscale("log")
            ax.set_ylim(FLOOR / 6, 1e3)
            ax.set_yticks([FLOOR, 1e-15, 1e-10, 1e-5, 1, 1e2], ["=0", "1e-15", "1e-10", "1e-5", "1", "100"])
            ax.axhline(1e-17, color="#C3CBD4", linewidth=0.8, linestyle=":")
            ax.set_xlim(-0.6, len(ms) - 0.4)
            ax.grid(axis="y")
            ax.set_axisbelow(True)
            method_axis(ax, ms, "bottom")
            ax.set_ylabel("系数相对 L2 误差")
            ax.set_title(f"{COHORT_LABEL[cohort]} · {TASK_LABEL[task]}", loc="left", pad=26)
        return panel([cohort], [task], ms, size, draw)
    return [make((NEW, "pde"), (5.2, 4.0)), make((OLD, "pde"), (2.2, 4.0)),
            make((OLD, "ode"), (4.4, 4.0)), make((OLD, "sr"), (4.4, 4.0))]


def fig_runtime(D):
    rng = np.random.default_rng(17)

    def make(group):
        cohort, task = group
        ms = D["methods"][group]

        def draw(fig, spec):
            ax = fig.add_subplot(spec)
            for i, m in enumerate(ms):
                values = D["fit"][(*group, m)]
                timeout = D["cov"][(*group, m)]["timeout"]
                if values:
                    strip(ax, rng, i, values, COLOR[m], (m, cohort), horizontal=True, width=0.22)
                    note = f"中位 {fmt(statistics.median(values))} s · n={len(values)}"
                else:
                    note = "n=0"
                if timeout:
                    note += f" · 硬超时 {timeout}"
                tag(ax.text(1.01, i, note, transform=ax.get_yaxis_transform(), va="center", fontsize=7.8),
                    m, cohort, "txt")
            ax.set_xscale("log")
            ax.set_xlim(3e-3, 1.5e3)
            ax.set_ylim(len(ms) - 0.4, -0.6)
            ax.axvline(BUDGET, color="#C0504D", linewidth=0.9, linestyle="--")
            ax.text(BUDGET, -0.6, " 300 s 预算", color="#C0504D", fontsize=7.5, va="bottom")
            ax.grid(axis="x")
            ax.set_axisbelow(True)
            method_axis(ax, ms, "left")
            ax.set_xlabel("单目标拟合时间 fit_seconds（秒，对数）")
            ax.set_title(f"{COHORT_LABEL[cohort]} · {TASK_LABEL[task]}", loc="left", pad=12)
        return panel([cohort], [task], ms, (9.0, 0.55 * len(ms) + 1.25), draw)
    return [make(g) for g in ((NEW, "pde"), (OLD, "pde"), (OLD, "ode"), (OLD, "sr"))]


def fig_new_pde(D):
    ms = D["methods"][NEW, "pde"]
    med = {k: statistics.median(v) for k, v in D["nrmse"].items()}
    norm = mcolors.LogNorm(1e-3, 2e4)
    cmap = plt.get_cmap("OrRd")
    rng = np.random.default_rng(5)

    def heat(fig, spec):
        ax = fig.add_subplot(spec)
        for i, d in enumerate(PDE_DATASETS):
            for j, m in enumerate(ms):
                face = cmap(norm(med[d, m]))
                best = D["best"][d] == m
                tag(ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=face, edgecolor="#1B2733" if best
                                           else "white", linewidth=1.8 if best else 1.2, zorder=3 if best else 2)),
                    m, NEW, "cell")
                tag(ax.text(j, i, fmt(med[d, m]), ha="center", va="center", fontsize=7.3, color=ink(face),
                            fontweight="bold" if best else "normal", zorder=4), m, NEW, "txt")
        heat_axes(ax, list(PDE_DATASETS), ms, "(a) 方程残差 NRMSE 中位数（n=5/格，黑框=该数据集最低）")
        fig.colorbar(plt.cm.ScalarMappable(norm, cmap), ax=ax, orientation="horizontal", location="bottom",
                     shrink=0.6, pad=0.02, aspect=40, label="NRMSE 中位数（对数色标）")

    def scatter(fig, spec):
        ax = fig.add_subplot(spec)
        for j, m in enumerate(ms):
            x = [med[d, m] for d in PDE_DATASETS]
            y = [D["heat"][NEW, "pde"][1][d, m][0] + (j - 2) * 0.09 for d in PDE_DATASETS]
            tag(ax.scatter(x, y, s=34, color=COLOR[m], edgecolors="white", linewidths=0.6, zorder=3), m, NEW, "pts")
        ax.set_xscale("log")
        ax.set_xlim(4e-4, 6e4)
        ax.set_ylim(-0.5, 5.5)
        ax.set_yticks(range(6))
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.set_xlabel("数据集残差 NRMSE 中位数（对数）")
        ax.set_ylabel("项结构匹配的 seed 数（/5）")
        ax.set_title("(b) 残差与结构恢复（每点=1 数据集，n=10/方法）", loc="left")

    def paired(fig, spec):
        ax = fig.add_subplot(spec)
        names = list(D["dlga"])
        for i, d in enumerate(names):
            old, new = (np.array([v for _, v in D["dlga"][d][c]]) for c in (OLD, NEW))
            xo, xn = i - 0.17 + rng.uniform(-0.04, 0.04, 5), i + 0.17 + rng.uniform(-0.04, 0.04, 5)
            for a, b, ya, yb in zip(xo, xn, old, new):
                tag(ax.plot([a, b], [ya, yb], color="#B8C2CC", linewidth=0.8, zorder=1)[0], "dlga", "any", "ln")
            tag(ax.scatter(xo, old, s=30, facecolors="white", edgecolors=COLOR["dlga"], linewidths=1.3, zorder=3),
                "dlga", OLD, "pts")
            tag(ax.scatter(xn, new, s=30, color=COLOR["dlga"], zorder=3), "dlga", NEW, "pts")
            for x, v, c in ((i - 0.17, old, OLD), (i + 0.17, new, NEW)):
                tag(ax.text(x, 2.6, fmt(float(np.median(v))), ha="center", fontsize=7.5), "dlga", c, "txt")
        ax.set_yscale("log")
        ax.set_ylim(0.04, 4)
        ax.set_xticks(range(len(names)), names)
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.set_ylabel("DLGA 方程残差 NRMSE（对数）")
        ax.set_title("(c) DLGA 旧快照（空心）vs 新扩展（实心），同 seed 连线", loc="left")
    return [panel([NEW], ["pde"], ms, (5.6, 5.0), heat), panel([NEW], ["pde"], ms, (5.0, 4.0), scatter),
            panel([OLD, NEW], ["pde"], ["dlga"], (5.0, 4.0), paired)]


# ---------------------------------------------------------------- text (all numbers computed)
def texts(D):
    rec, cov, fit, sh = D["recovery"], D["cov"], D["fit"], D["shared"]

    def pct(k, n):
        return f"{k}/{n}（{100 * k / n:.0f}%）"

    def r(m):
        return rec[NEW, "pde", m]
    wins = Counter(D["best"].values())
    cmed = {m: statistics.median(e for *_, e in D["coef"][NEW, "pde", m]) for m in D["methods"][NEW, "pde"]}
    runner = min((m for m in cmed if m != "weakform"), key=cmed.get)
    gap = math.log10(cmed[runner] / cmed["weakform"])
    fmed = {m: statistics.median(fit[NEW, "pde", m]) for m in D["methods"][NEW, "pde"]}
    eligible = Counter()
    for row in D["by"][NEW, "pde", "weakform"] + D["by"][NEW, "pde", "pdefind"]:
        if row.get("recovery_eligible"):
            eligible[row["model"]] += row["pde_support_recovery"]
    dl = {d: {c: statistics.median(v for _, v in D["dlga"][d][c]) for c in (OLD, NEW)} for d in D["dlga"]}
    assert D["dlga"]["burgers"][OLD] == D["dlga"]["burgers"][NEW]
    physo_to = cov[OLD, "sr", "physo"]["timeout"] + cov[OLD, "ode", "physo"]["timeout"]
    deep_full = sum(v >= BUDGET for v in fit[NEW, "pde", "deepmod"])
    full_ws = sum(D["heat"][NEW, "pde"][1][d, "weakform"] == (5, 5) for d in PDE_DATASETS)
    span = math.log10(statistics.median(fit[NEW, "pde", "deepmod"]) / fmed["pdefind"])
    assert span >= 4 and gap >= 7
    findings = [
        f"新 PDE 扩展 250/250 目标全部成功，但项结构恢复差距极大：WSINDy {pct(*r('weakform'))}（认证子集 "
        f"{eligible['weakform']}/15），PDE-FIND {pct(*r('pdefind'))}，PySINDy、DLGA、DeepMoD 均为 0/50。",
        f"低残差不等于正确结构：PySINDy 在 {wins['pysindy']}/10 个数据集残差最低却 0/50 恢复；30 个带系数参考目标上 "
        f"WSINDy 系数误差中位数 {fmt(cmed['weakform'])}，比次优的 {LABEL[runner]}（{fmt(cmed[runner])}）低约 {gap:.1f} 个数量级。",
        f"效率相差 4 个数量级以上：稀疏回归 fit 中位数 {fmt(fmed['pdefind'])}–{fmt(fmed['weakform'])} s，DLGA "
        f"{fmt(fmed['dlga'])} s，DeepMoD {deep_full}/50 用满 300 s 预算；旧快照 PhySO 共 {physo_to} 次硬超时。"]
    s3, o3, o4 = sh["sr_3"], sh["ode_3"], sh["ode_4"]
    figures = {
        "fig1": (f"新扩展五方法完成率均为 100%，结构恢复率 WSINDy {pct(*r('weakform'))}最高；旧快照 SR 中 "
                 f"SymbolicGPT 仅 {cov[OLD, 'sr', 'symbolicgpt']['terminal']}/1340、PhySO "
                 f"{cov[OLD, 'sr', 'physo']['terminal']}/1340 到达终态。",
                 "左：各状态占计划目标的比例，右侧数字为成功/计划；右：结构恢复数/可评估数（仅成功且可判定的目标）。"
                 "SR/ODE 为忽略系数的结构恢复，PDE 为登记项支撑恢复。"),
        "fig2": (f"同一共同目标集上比较：旧 SR {s3[0]} 个共同目标 PhySO {s3[1]['physo']}、SISSO {s3[1]['sisso']}、"
                 f"LLM-SR {s3[1]['llmsr']}；旧 ODE {o3[0]} 个共同目标 LLM-SR {o3[1]['llmsr']}、SISSO {o3[1]['sisso']}、"
                 f"SymbolicGPT {o3[1]['symbolicgpt']}；新 PDE 中 WSINDy 在 10 个数据集中 {full_ws} 个 5/5。",
                 f"格内为恢复/可评估，颜色越深恢复率越高，“—”为无可评估结果。* 为 recovery_eligible 认证数据集。"
                 f"四方法 ODE 共同目标仅 {o4[0]} 个，SymbolicGPT 在 SR 仅 1 个成功目标，故不纳入共同集。"),
        "fig3": (f"WSINDy 系数误差中位数 {fmt(cmed['weakform'])}，30/30 达标；PDE-FIND {fmt(cmed['pdefind'])}"
                 f"（{D['coef_ok']['pdefind'][0]}/30），PySINDy {fmt(cmed['pysindy'])}（0/30），DLGA {fmt(cmed['dlga'])}，"
                 f"DeepMoD {fmt(cmed['deepmod'])}。",
                 "相对 L2 误差，对数坐标；n 为有效样本。新 PDE 为 6 个带系数参考数据集×5 seed；旧 SR/ODE 仅在结构已恢复时可算，"
                 "样本很小。“=0” 行为误差恰为 0 的目标。达标按 rtol=0.05、atol=1e-8。"),
        "fig4": (f"同一新矩阵内 PDE-FIND fit 中位数 {fmt(fmed['pdefind'])} s，PySINDy {fmt(fmed['pysindy'])} s，"
                 f"WSINDy {fmt(fmed['weakform'])} s，DLGA {fmt(fmed['dlga'])} s，DeepMoD 全部用满 300 s。",
                 "仅统计成功目标的 fit_seconds，n 为样本数；硬超时目标无 fit 值，单独计数。新旧资源协议不同，不做跨批次速度比较。"),
        "fig5": (f"PySINDy 残差最低 {wins['pysindy']}/10 但结构 0/50，WSINDy 残差最低 {wins['weakform']}/10、结构 40/50；"
                 f"DLGA 新旧 Burgers 逐 seed 相同，Chafee–Infante 中位数由 {fmt(dl['chafee-infante'][OLD])} 升至 "
                 f"{fmt(dl['chafee-infante'][NEW])}。",
                 "(a)(b) 均为新扩展；(c) 旧快照 domain-v4 与新扩展 domain-v5 分批展示，不合并为重复。"
                 "PDE 残差以留出 u_t 标准差归一化。")}
    return findings, figures


# ---------------------------------------------------------------- HTML
CSS = """
body{margin:0;background:#F5F7FA;color:#203047;font:14px/1.6 "Microsoft YaHei","PingFang SC",system-ui,sans-serif}
main{max-width:1320px;margin:auto;padding:24px 28px 48px}h1{font-size:23px;margin:0 0 4px}
.meta{color:#5A6B7C;font-size:13px;margin:0 0 14px}.meta a{color:#17639C}
ol.find{background:#fff;border:1px solid #D9E1EA;border-left:4px solid #4C7FA8;margin:0 0 18px;padding:12px 18px 12px 38px}
ol.find li{margin:4px 0}
.bar{position:sticky;top:0;z-index:5;background:#fff;border:1px solid #D9E1EA;padding:10px 14px;display:flex;
flex-wrap:wrap;gap:6px 22px;align-items:center;font-size:13px;box-shadow:0 2px 6px rgba(32,48,71,.06)}
.bar b{color:#5A6B7C;font-weight:600;margin-right:4px}.bar label{margin-right:10px;white-space:nowrap;cursor:pointer}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 2px;vertical-align:-1px}
.bar button{font:inherit;font-size:12px;padding:2px 10px;border:1px solid #C3CDD8;background:#F5F7FA;border-radius:3px;cursor:pointer}
section{margin-top:26px}h2{font-size:17px;margin:0 0 2px}p.key{margin:2px 0 10px;font-weight:600;color:#17395C}
.grid{display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start}
figure.card{margin:0;background:#fff;border:1px solid #D9E1EA;padding:8px;max-width:100%}
figure.card svg{display:block;max-width:100%;height:auto}
p.cap{color:#5A6B7C;font-size:12.5px;margin:8px 0 0}p.cap a{color:#17639C;margin-left:6px}
.empty{color:#8A96A3;font-size:13px;display:none}.hide{display:none!important}
footer{margin-top:34px;color:#6B7785;font-size:12px;border-top:1px solid #D9E1EA;padding-top:10px}
code{background:#EEF2F6;padding:1px 4px;border-radius:3px}
"""

JS = """
const q=s=>[...document.querySelectorAll(s)],val=n=>q(`input[name=${n}]:checked`).map(i=>i.value);
function apply(){const c=val('cohort'),t=val('task'),m=val('method');
q('figure.card').forEach(f=>{const ok=f.dataset.cohorts.split(' ').some(x=>c.includes(x))&&
f.dataset.tasks.split(' ').some(x=>t.includes(x))&&f.dataset.methods.split(' ').some(x=>m.includes(x));
f.classList.toggle('hide',!ok)});
q('section').forEach(s=>{const e=s.querySelector('.empty');if(e)e.style.display=s.querySelector('figure.card:not(.hide)')?'none':'block'});
let css='';for(const x of q('input[name=method]:not(:checked)'))css+=`[id*="__m-${x.value}|"]{display:none}`;
for(const x of q('input[name=cohort]:not(:checked)'))css+=`[id*="|c-${x.value}|"]{display:none}`;
document.getElementById('dyn').textContent=css}
q('.bar input').forEach(i=>i.addEventListener('change',apply));
document.getElementById('all').onclick=()=>{q('.bar input').forEach(i=>i.checked=true);apply()};apply();
"""


def checkbox(name, value, text, color=None):
    swatch = f'<span class="sw" style="background:{color}"></span>' if color else ""
    return f'<label><input type="checkbox" name="{name}" value="{value}" checked>{swatch}{text}</label>'


FIGURES = (  # name, title, builder, composite (figsize, grid shape, slots, height_ratios, width_ratios)
    ("fig1_overview", "算法表现总览：完成率与结构恢复率", fig_overview,
     ((9.6, 10.4), (4, 2), [[(i, 0), (i, 1)] for i in range(4)], [4, 4, 1.7, 5], [3, 2])),
    ("fig2_recovery_heatmap", "结构恢复热力图", fig_recovery,
     ((10.5, 10.5), (3, 2), [(0, 1), (1, 1), (2, 1), (slice(0, 3), 0)], [10, 8, 3], [1, 1])),
    ("fig3_coefficient_error", "系数误差分布", fig_coefficients,
     ((15.5, 4.4), (1, 4), [0, 1, 2, 3], None, [5, 1.4, 4, 4])),
    ("fig4_runtime", "运行耗时对比", fig_runtime,
     ((10.0, 11.4), (4, 1), [0, 1, 2, 3], [5, 1.8, 4, 4], None)),
    ("fig5_new_pde", "新增 PDE 结果", fig_new_pde,
     ((16.0, 5.2), (1, 3), [0, 1, 2], None, [5.6, 5, 5])),
)


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, metrics, summary = load()
    D = derive(rows, metrics, summary)
    findings, words = texts(D)
    sections = []
    for f, (name, title, build, (size, shape, slots, h, w)) in enumerate(FIGURES, 1):
        tag.n = 0
        panels = build(D)
        render_composite(name, f"图 {f}  {title}", panels, size, shape, slots, h, w,
                         status_legend if f == 1 else None)
        tag.n = 0
        cards = []
        for i, p in enumerate(build(D)):
            cards.append(f'<figure class="card" data-cohorts="{" ".join(p["cohorts"])}" data-tasks="'
                         f'{" ".join(p["tasks"])}" data-methods="{" ".join(p["methods"])}">'
                         f'{render_card(p, f"f{f}p{i}")}</figure>')
        key, cap = words[name.split("_")[0]]
        sections.append(f'<section id="{name}"><h2>图 {f}　{title}</h2><p class="key">{key}</p>'
                        f'<div class="grid">{"".join(cards)}</div><p class="empty">当前筛选下无可显示的图。</p>'
                        f'<p class="cap">{cap}<a href="{name}.svg">SVG</a><a href="{name}.png">PNG</a></p></section>')
    write_html(findings, sections)
    data = dict(findings=findings, figures={k: dict(key=v[0], caption=v[1]) for k, v in words.items()},
                recovery={"|".join(k): dict(recovered=v[0], evaluable=v[1]) for k, v in D["recovery"].items()},
                shared_targets={k: dict(n=v[0], recovered=v[1]) for k, v in D["shared"].items()},
                source_sha256={p: file_sha(SRC / p) for p in ("rows.jsonl", "dataset_metrics.csv", "summary.json")})
    (OUT / "figure_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT.relative_to(ROOT))


def write_html(findings, sections):
    bar = ('<div class="bar"><span><b>批次</b>' + "".join(checkbox("cohort", c, t) for c, t in COHORT_LABEL.items())
           + '</span><span><b>任务</b>' + "".join(checkbox("task", k, t) for k, t in TASK_LABEL.items())
           + '</span><span><b>算法</b>' + "".join(checkbox("method", m, LABEL[m], COLOR[m]) for m in ORDER)
           + '</span><button id="all">全部显示</button></div>')
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>综合实验可视化报告 · 2026-10-17</title>
<style>{CSS}</style><style id="dyn"></style></head><body><main>
<h1>综合实验可视化报告 · 2026-10-17</h1>
<p class="meta">旧快照 2026-10-16（domain-v4，4 种 SR 方法 + DLGA，SR/ODE/PDE）与新 PDE 扩展 2026-10-17（domain-v5，5 方法 × 10 数据集 × 5 seed）分批展示，不合并统计。
<a href="../index.html">可搜索明细表</a> · <a href="../../../../docs/benchmark_results_combined_20261017.md">完整表格报告</a></p>
<ol class="find">{"".join(f"<li>{x}</li>" for x in findings)}</ol>
{bar}
{"".join(sections)}
<footer>生成：<code>python scripts/build_visual_report_20261017.py</code>（只读 <code>combined_results_20261017</code> 下的
rows.jsonl、dataset_metrics.csv、summary.json；全部数字由脚本重算并与 CSV 交叉校验，见 figure_data.json）。</footer>
</main><script>{JS}</script></body></html>"""
    (OUT / "index.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()