"""Build offline, family-grouped tables from the frozen cloud snapshot (stdlib only)."""

from __future__ import annotations

import csv
import html
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/analysis/cloud_results_20261016"
OUTPUT = SOURCE / "tables"
MARKDOWN = ROOT / "docs/benchmark_results_tables_20261016.md"
METHODS = ("sisso", "physo", "symbolicgpt", "llmsr")
LABELS = dict(zip(METHODS, ("SISSO", "PhySO", "SymbolicGPT", "LLM-SR"))) | {"dlga": "DLGA"}
METRICS = {
    "nrmse": "NRMSE", "mse": "MSE", "r2": "R²",
    "complexity": "表达式复杂度", "runtime": "运行时间（秒）",
    "structural_recovery": "结构恢复（忽略数值系数）",
    "algebraic_exact_recovery": "严格符号恢复诊断",
    "pde_support_recovery": "PDE 项支撑恢复",
}
RECOVERY = {"structural_recovery", "algebraic_exact_recovery", "pde_support_recovery"}
SCOPE = {"dataset": "数据集", "system": "ODE 系统（合并 RHS）", "rhs": "ODE RHS 目标"}


def natural(value):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def family(dataset):
    if re.fullmatch(r"R[123]a?", dataset):
        return "Rational"
    if dataset in {"Nonic", "Sine"}:
        return "其他单方程"
    # The a variants remain adjacent to their numbered family members.
    return re.sub(r"-\d+a?$", "", dataset)


def metric_field(task, metric):
    if metric in {"mse", "nrmse"}:
        return ("equation_residual_" if task == "pde" else "test_") + metric
    return {"r2": "test_r2", "complexity": "expression_complexity"}.get(metric, metric)


def describe(rows, task, metric):
    field = metric_field(task, metric)
    values = [row[field] for row in rows if row["status"] == "ok" and finite(row.get(field))]
    result = {"finite": len(values), "median": None, "q1": None, "q3": None,
              "recovery_yes": None, "recovery_evaluable": None}
    if metric in RECOVERY:
        assert all(value in (0, 1) for value in values), (field, values)
        result.update(recovery_yes=sum(values), recovery_evaluable=len(values))
    elif values:
        # Inclusive quartiles interpolate within the observed range, even for n=1.
        q = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else values * 3
        result.update(median=statistics.median(values), q1=q[0], q3=q[2])
    return result


def number(value):
    return "—" if value is None else format(value, ".4g")


def cell(record, metric="nrmse"):
    stats = record["metrics"][metric]
    count = f"[{stats['finite']}/{record['planned']}]"
    if stats["finite"]:
        return f"{number(stats['median'])} {count}"
    if not record["terminal"]:
        return "—"
    failures = f"超时{record['timeout']}；错误{record['error']}"
    if record["ok"]:
        return f"无有限值 {count}（成功{record['ok']}；{failures}）"
    return f"失败 {count}（{failures}）"


def validate(rows, summary, dataset_summary):
    assert len(rows) == summary["overall"]["finished_targets"] == 2556
    assert Counter(row["status"] for row in rows) == summary["overall"]["status_counts"]
    assert Counter(row["status"] for row in rows) == {"ok": 2089, "timeout": 334, "error": 133}
    assert {row["seed"] for row in rows} == set(range(5))
    assert {row["metric_protocol_version"] for row in rows} == {"2.1"}
    assert {row["dataset_protocol_version"] for row in rows} == {"domain-v4"}
    assert {row["profile"] for row in rows} == {"full"}
    assert {row["experiment"] for row in rows} == set(summary["matrix_plans"])
    identities = [(r["task"], r["model"], r["dataset"], r["target"], r["seed"]) for r in rows]
    assert len(set(identities)) == len(identities), "重复的 method/dataset/target/seed"
    main_finite = sum(describe([row], row["task"], "nrmse")["finite"] for row in rows)
    assert main_finite == 2088
    by_dataset = defaultdict(list)
    by_model_task = defaultdict(list)
    for row in rows:
        by_dataset[row["model"], row["task"], row["dataset"]].append(row)
        by_model_task[row["model"], row["task"]].append(row)
    for entry in summary["model_task"]:
        subset = by_model_task[entry["model"], entry["task"]]
        assert len(subset) == entry["finished_targets"]
        assert Counter(r["status"] for r in subset) == entry["status_counts"]
    assert len(dataset_summary) == len(by_dataset)
    for entry in dataset_summary:
        subset = by_dataset[entry["model"], entry["task"], entry["dataset"]]
        assert len(subset) == int(entry["finished_targets"])
        assert Counter(r["status"] for r in subset) == json.loads(entry["status_counts"])
        for metric in ("nrmse", "mse", "r2", "complexity", "runtime"):
            field = metric_field(entry["task"], metric)
            stats = describe(subset, entry["task"], metric)
            assert stats["finite"] == int(entry[field + "_finite_ok"])
            old = entry[field + "_median_ok"]
            assert (not old and stats["median"] is None) or (old and math.isclose(float(old), stats["median"], rel_tol=1e-12, abs_tol=1e-15))
    return by_model_task


def build_records(rows, summary):
    by_target = defaultdict(list)
    targets = defaultdict(set)
    for row in rows:
        targets[row["task"]].add((row["dataset"], row["target"]))
        by_target[row["task"], row["model"], row["dataset"], row["target"]].append(row)
    assert tuple(len(targets[t]) for t in ("sr", "ode", "pde")) == (268, 17, 3)
    assert len({d for d, _ in targets["ode"]}) == 8
    for task in targets:
        complete = "dlga" if task == "pde" else "sisso"
        for dataset, target in targets[task]:
            assert {r["seed"] for r in by_target[task, complete, dataset, target]} == set(range(5))
    records = []
    for task in ("sr", "ode", "pde"):
        methods = ("dlga",) if task == "pde" else METHODS
        for dataset in sorted({d for d, _ in targets[task]}, key=natural):
            dataset_targets = sorted((t for d, t in targets[task] if d == dataset), key=natural)
            scopes = [("system" if task == "ode" else "dataset", "", dataset_targets)]
            if task == "ode":
                scopes += [("rhs", target, [target]) for target in dataset_targets]
            for scope, label, selected in scopes:
                for method in methods:
                    subset = [r for target in selected for r in by_target[task, method, dataset, target]]
                    statuses = Counter(r["status"] for r in subset)
                    records.append({
                        "task": task, "scope": scope, "family": family(dataset) if task == "sr" else task.upper(),
                        "dataset": dataset, "target": label, "method": method,
                        "planned": len(selected) * 5, "terminal": len(subset), "ok": statuses["ok"],
                        "timeout": statuses["timeout"], "error": statuses["error"],
                        "missing": len(selected) * 5 - len(subset),
                        "metrics": {metric: describe(subset, task, metric) for metric in METRICS},
                    })
    planned = sum(r["planned"] for r in records if r["scope"] != "rhs")
    assert planned == summary["planned_targets"] == 5715
    for item in summary["planned_cases_by_model_task"]:
        chosen = [r for r in records if r["scope"] != "rhs" and (r["method"], r["task"]) == (item["model"], item["task"])]
        assert len(chosen) * 5 == item["cases"]
    assert sum(r["terminal"] for r in records if r["scope"] != "rhs") == 2556
    return records


def groups(records):
    result = [("ODE · 8 个系统（合并 RHS）", [r for r in records if r["scope"] == "system"], METHODS),
              ("ODE · 17 个 RHS 目标", [r for r in records if r["scope"] == "rhs"], METHODS),
              ("PDE · 3 个数据集（方程残差）", [r for r in records if r["task"] == "pde"], ("dlga",))]
    families = sorted({r["family"] for r in records if r["task"] == "sr"}, key=natural)
    for name in families:
        result.append((f"SR · {'Livermore2 / ' if name.startswith('Livermore2-') else ''}{name}",
                       [r for r in records if r["task"] == "sr" and r["family"] == name], METHODS))
    return result


def table_rows(records, methods):
    indexed = {(r["dataset"], r["target"], r["method"]): r for r in records}
    keys = sorted({(r["dataset"], r["target"]) for r in records}, key=lambda key: (natural(key[0]), natural(key[1])))
    return [(dataset + (f" · {target}" if target else ""), [indexed[dataset, target, m] for m in methods]) for dataset, target in keys]


def coverage_table(records):
    lines = ["| 基线 | 任务 | 计划目标 | 已有终态 | 成功 | 超时 | 错误 | 未见终态 | 主误差有限 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method in (*METHODS, "dlga"):
        for task in ("sr", "ode", "pde"):
            subset = [r for r in records if r["scope"] != "rhs" and r["method"] == method and r["task"] == task]
            if not subset:
                lines.append(f"| {LABELS[method]} | {task.upper()} | —（未规划） | — | — | — | — | — | — |")
                continue
            values = [sum(r[key] for r in subset) for key in ("planned", "terminal", "ok", "timeout", "error", "missing")]
            values += [sum(r["metrics"]["nrmse"]["finite"] for r in subset)]
            lines.append(f"| {LABELS[method]} | {task.upper()} | " + " | ".join(map(str, values)) + " |")
    return lines


def ic_table(rows):
    lines = ["| 已有 IC 类型 | 可用目标 | 适用范围与说明 |", "|---|---:|---|"]
    actual = [r for r in rows if r["status"] == "ok" and finite(r.get("aic")) and finite(r.get("bic"))]
    proxy = [r for r in rows if r["status"] == "ok" and finite(r.get("aic_proxy_heldout_complexity")) and finite(r.get("bic_proxy_heldout_complexity"))]
    assert len(actual) == 422 and Counter(r["task"] for r in actual) == {"sr": 337, "ode": 85}
    assert {r["model"] for r in actual} == {"llmsr"} and len(proxy) == 2086
    lines += ["| 条件训练 AIC/BIC | 422 | LLM-SR：SR 337，ODE 85；conditional_training_plugin |",
              "| 留出误差／复杂度代理 AIC/BIC | 2086 | exploratory_heldout_complexity；与上一行重叠，不能相加 |"]
    return lines


def write_csv(records):
    fields = ["任务", "汇总层级", "数据集家族", "数据集", "RHS目标", "基线", "计划目标", "已有终态", "成功", "超时", "错误", "未见终态",
              "指标", "源字段", "有限成功数", "成功但指标缺失或非有限", "中位数", "Q1", "Q3", "recovery_yes", "recovery_evaluable"]
    with (OUTPUT / "dataset_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for r in records:
            for metric, label in METRICS.items():
                s = r["metrics"][metric]
                writer.writerow([r["task"].upper(), SCOPE[r["scope"]], r["family"], r["dataset"], r["target"], LABELS[r["method"]],
                                 *[r[key] for key in ("planned", "terminal", "ok", "timeout", "error", "missing")],
                                 label, metric_field(r["task"], metric), s["finite"], r["ok"] - s["finite"],
                                 *[s[key] for key in ("median", "q1", "q3", "recovery_yes", "recovery_evaluable")]])
    with (OUTPUT / "nrmse_matrix.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        methods = (*METHODS, "dlga")
        writer.writerow(["任务", "汇总层级", "数据集家族", "数据集", "RHS目标"] + [f"{LABELS[m]}_{name}" for m in methods for name in ("NRMSE中位数", "有限成功数", "计划目标", "已有终态", "成功", "超时", "错误", "未见终态")])
        index = defaultdict(dict)
        for r in records:
            index[r["task"], r["scope"], r["family"], r["dataset"], r["target"]][r["method"]] = r
        for key, values in index.items():
            output = [key[0].upper(), SCOPE[key[1]], *key[2:]]
            for method in methods:
                r = values.get(method)
                output += ([r["metrics"]["nrmse"]["median"], r["metrics"]["nrmse"]["finite"], *[r[k] for k in ("planned", "terminal", "ok", "timeout", "error", "missing")]] if r else [None] * 8)
            writer.writerow(output)


INTRO = """这是 **2026-10-16 14:10:52 UTC+08:00** 的冻结云端快照，只含两个 full 实验矩阵的正式 chunk 终态。不是服务器实时进度，也不含历史 smoke 或其他历史基线。协议为 metric 2.1 / domain-v4；不混入本地更新后的协议。

计划 **5715** 个目标，已有终态 **2556**：成功 2089、超时 334、错误 133；未见终态 3159。主误差有限的成功目标为 **2088**，其中一个成功目标无有限主误差。SR 共 268 个数据集，每个方法／数据集 5 个 seed（0–4）；ODE 共 8 个系统、17 个 RHS 目标，每个方法／RHS 5 个 seed；PDE 共 3 个数据集，每个 5 个 seed。

单元格为 **成功且指标有限的子集之中位数 [有限成功目标数 / 计划目标数]**，不把失败或缺失填成零。`—` 表示尚无终态；`失败 [0/N]` 附超时／错误数量；成功但无有限指标单独标明。不同方法的成功子集可能不同，因此不标“最优”，也不能据此直接作完整横向排名。完整终态／缺失数量和四分位数见 CSV；Q1/Q3 使用 inclusive 插值。

SR 使用留出预测 NRMSE；ODE 使用 **RHS 导数预测 NRMSE**，不是轨迹积分误差。ODE 系统行把各 RHS 的目标记录合并后取中位数，分母为 RHS 数 × 5，不是先算各 RHS 再平均；下方另列全部 RHS，避免掩盖单个方程表现。PDE 使用 **方程残差 NRMSE**，与前两者含义不同。NRMSE/MSE 越低越好，R² 越高越好。运行时间取原始 `runtime` 字段（秒），仅统计成功有限值，不代表总计算消耗；ODE 系统时间是目标记录的中位数，不是整系统总耗时。

长 CSV 另列结构恢复、严格符号恢复诊断和 PDE 项支撑恢复的 `recovery_yes / recovery_evaluable`。结构恢复忽略数值系数；严格符号恢复是现有 `algebraic_exact_recovery` 诊断；PDE 支撑恢复评价项集合。三者都不等于运行成功，且不可互换。未适用／未提供的字段无可评估值。恢复计数同样只在成功记录中统计；可评估分母与计划分母分别保留。

长 CSV 对同一方法／数据集组合按指标分行，ODE 另有系统与 RHS 两种视图。汇总时须先筛选单一指标与汇总层级，不能直接累加这些行的目标数。
"""


def write_markdown(records, rows):
    lines = ["# 基线 × 数据集结果表（2026-10-16 冻结快照）", "", INTRO,
             "[离线 HTML（可切换五个指标）](../results/analysis/cloud_results_20261016/tables/index.html) · "
             "[长表 CSV](../results/analysis/cloud_results_20261016/tables/dataset_metrics.csv) · "
             "[NRMSE 宽表 CSV](../results/analysis/cloud_results_20261016/tables/nrmse_matrix.csv)", "",
             "## 计划与已观测覆盖", "", *coverage_table(records), "",
             "## 信息准则可用性", "", *ic_table(rows), "",
             "沿用现有逐目标 IC，不重新计算，不跨数据集／目标平均 AIC/BIC。代理 IC 不是真实训练似然 IC；二者不混为同一口径。"
             "逐次值及状态见 [已有 results.csv](../results/analysis/cloud_results_20261016/results.csv)。本表不含 PIC。", "",
             "## 按家族排列的 NRMSE", "",
             "Livermore2 按 Vars2–Vars7 列为子组；R1/R1a/R2/R2a/R3/R3a 归 Rational；Constant 与 Const-Test 分开。编号按自然顺序排列。", ""]
    for title, subset, methods in groups(records):
        entries = table_rows(subset, methods)
        lines += [f"### {title}（{len(entries)} 行）", "", "| 数据集／目标 | " + " | ".join(LABELS[m] for m in methods) + " |",
                  "|---|" + "---:|" * len(methods)]
        lines += ["| " + name + " | " + " | ".join(cell(r) for r in values) + " |" for name, values in entries]
        lines.append("")
    lines += ["## 来源与复现", "", "由 `scripts/build_result_tables_20261016.py` 使用 Python 标准库读取冻结目录内的 `rows.jsonl`、`summary.json`、`dataset_summary.csv` 生成。"
              "脚本独立核对终态总数、状态、逐数据集已有汇总、五个 seed 与计划目标数。只读原始文件，不启动实验。", "",
              "```powershell", "python scripts/build_result_tables_20261016.py", "```", ""]
    MARKDOWN.write_text("\n".join(lines), encoding="utf-8")


def write_html(records, rows):
    escape = html.escape
    sections, links = [], []
    for i, (title, subset, methods) in enumerate(groups(records)):
        entries = table_rows(subset, methods)
        links.append(f'<a href="#group-{i}">{escape(title)}</a>')
        body = []
        for name, values in entries:
            cells = []
            for r in values:
                data = escape(json.dumps(r, ensure_ascii=False, allow_nan=False), quote=True)
                detail = f"终态 {r['terminal']}/{r['planned']}；成功 {r['ok']}；超时 {r['timeout']}；错误 {r['error']}；未见终态 {r['missing']}"
                cells.append(f'<td data-record="{data}" title="{escape(detail)}">{escape(cell(r))}</td>')
            body.append(f'<tr><th scope="row">{escape(name)}</th>{"".join(cells)}</tr>')
        sections.append(f'<details id="group-{i}" open><summary>{escape(title)}（{len(entries)} 行）</summary><div class="table-wrap"><table><thead><tr><th>数据集／目标</th>{"".join(f"<th>{LABELS[m]}</th>" for m in methods)}</tr></thead><tbody>{"".join(body)}</tbody></table></div></details>')
    overview = coverage_table(records)
    overview_html = '<table><thead><tr>' + ''.join(f'<th>{escape(v.strip())}</th>' for v in overview[0].strip('|').split('|')) + '</tr></thead><tbody>'
    overview_html += ''.join('<tr>' + ''.join(f'<td>{escape(v.strip())}</td>' for v in line.strip('|').split('|')) + '</tr>' for line in overview[2:]) + '</tbody></table>'
    intro_html = ''.join(f'<p>{escape(p)}</p>' for p in INTRO.replace('**', '').replace('`', '').strip().split('\n\n'))
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>基线 × 数据集结果表 · 2026-10-16</title>
<style>body{font:15px/1.65 system-ui,"Microsoft YaHei",sans-serif;color:#243347;background:#f8fafc;margin:0}main{max-width:1450px;margin:auto;padding:28px}h1{font-size:27px}h2{font-size:21px}p{max-width:1120px}a{color:#175f9f}nav{display:flex;gap:8px 18px;flex-wrap:wrap;margin:18px 0}nav a{font-size:14px}.toolbar{position:sticky;top:0;z-index:3;background:#eaf1f8;padding:12px;border:1px solid #cad8e6;border-radius:6px}select{padding:5px;font:inherit}.table-wrap{overflow:auto;margin:8px 0 24px}table{border-collapse:separate;border-spacing:0;min-width:760px;width:100%;background:white}th,td{padding:8px 12px;border-bottom:1px solid #dde4ec;text-align:right;white-space:nowrap}thead th{position:sticky;top:64px;background:#eaf0f6;z-index:2}th:first-child,td:first-child{text-align:left}tbody th{font-weight:500}tbody tr:nth-child(even){background:#f5f8fb}tbody tr:hover{background:#e8f2fc}details{scroll-margin-top:75px}summary{font-size:18px;font-weight:650;cursor:pointer;padding:12px 0}small{color:#526477}.note{background:white;border-left:4px solid #92b3d2;padding:5px 16px}footer{margin-top:32px;color:#526477}@media(max-width:700px){main{padding:14px}h1{font-size:22px}th,td{padding:7px}}</style><main><h1>基线 × 数据集结果表</h1>
<p>冻结快照：2026-10-16 14:10:52 UTC+08:00</p>__INTRO__
<p><a href="dataset_metrics.csv">下载完整长表 CSV</a> · <a href="nrmse_matrix.csv">下载 NRMSE 宽表 CSV</a> · <a href="../results.csv">原始逐目标结果与 IC</a></p>
<h2>计划与已观测覆盖</h2><div class="table-wrap">__OVERVIEW__</div>
<h2>信息准则可用性</h2><p>条件训练 AIC/BIC：422 个（LLM-SR SR 337、ODE 85）；留出误差／复杂度代理：2086 个。两类有重叠，不能相加，不跨数据集／目标平均，不重新计算。代理 IC 不是真实训练似然 IC。逐目标值见原始 CSV。本表不含 PIC。</p>
<h2>数据集家族导航</h2><nav>__LINKS__</nav>
<div class="toolbar"><label for="metric">表格指标：</label><select id="metric"><option value="nrmse">NRMSE</option><option value="mse">MSE</option><option value="r2">R²</option><option value="complexity">表达式复杂度</option><option value="runtime">运行时间（秒）</option></select> <label for="search">搜索数据集：</label><input id="search" type="search" placeholder="例如 Robertson / Nguyen"><br><small>中位数 [有限成功数 / 计划目标数]；悬停查看终态明细。PDE 的 R² 未提供。</small></div>
__SECTIONS__<footer>来源：冻结目录的 rows.jsonl、summary.json、dataset_summary.csv。标准库生成；本页无需网络或依赖。ODE 系统与 RHS 是同一批目标的两种视图，勿重复计数。</footer></main>
<script>const cells=[...document.querySelectorAll('td[data-record]')].map(el=>[el,JSON.parse(el.dataset.record)]);function fmt(v){return v===null?'—':Number(v).toPrecision(4).replace(/(\\.\\d*?[1-9])0+(e|$)/,'$1$2').replace(/\\.0+(e|$)/,'$1')}function update(){const key=document.getElementById('metric').value;for(const [el,r] of cells){const s=r.metrics[key],count=`[${s.finite}/${r.planned}]`;el.textContent=s.finite?`${fmt(s.median)} ${count}`:!r.terminal?'—':r.ok?`无有限值 ${count}（成功${r.ok}；超时${r.timeout}；错误${r.error}）`:`失败 ${count}（超时${r.timeout}；错误${r.error}）`;el.title=`终态 ${r.terminal}/${r.planned}；成功 ${r.ok}；超时 ${r.timeout}；错误 ${r.error}；未见终态 ${r.missing}；成功但指标缺失或非有限 ${r.ok-s.finite}；Q1 ${fmt(s.q1)}；Q3 ${fmt(s.q3)}`}}document.getElementById('metric').addEventListener('change',update);for(const a of document.querySelectorAll('nav a'))a.addEventListener('click',()=>{document.querySelector(a.hash).open=true});update();</script></html>'''
    for token, value in {"__INTRO__": intro_html, "__OVERVIEW__": overview_html, "__LINKS__": ''.join(links), "__SECTIONS__": ''.join(sections)}.items():
        page = page.replace(token, value)
    search_script = '''<script>document.getElementById('search').addEventListener('input',event=>{const query=event.target.value.trim().toLowerCase();for(const group of document.querySelectorAll('details')){let visible=0;for(const row of group.querySelectorAll('tbody tr')){row.hidden=!row.querySelector('th').textContent.toLowerCase().includes(query);if(!row.hidden)visible++}group.hidden=visible===0;if(query&&visible)group.open=true}});</script>'''
    page = page.replace("</html>", search_script + "</html>")
    (OUTPUT / "index.html").write_text(page, encoding="utf-8")


def main():
    rows = [json.loads(line) for line in (SOURCE / "rows.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = json.loads((SOURCE / "summary.json").read_text(encoding="utf-8"))
    with (SOURCE / "dataset_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        dataset_summary = list(csv.DictReader(handle))
    validate(rows, summary, dataset_summary)
    records = build_records(rows, summary)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(records)
    write_markdown(records, rows)
    write_html(records, rows)
    print(f"已核验：5715 个计划目标，2556 个终态，2088 个有限主误差；生成 {len(records)} 个方法／数据集视图。")
    print(MARKDOWN.relative_to(ROOT))
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
