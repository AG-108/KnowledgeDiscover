"""Combine the historical snapshot and verified PDE extension without pooling cohorts.

Standard-library only. Reads frozen results; never imports fitting code or starts jobs.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import build_result_tables_20261016 as old

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / 'results/analysis/pde_extension_20261017'
OUT = ROOT / 'results/analysis/combined_results_20261017'
DOC = ROOT / 'docs/benchmark_results_combined_20261017.md'
METHODS = ('pdefind', 'pysindy', 'weakform', 'dlga', 'deepmod')
DATASETS = ('burgers', 'burgers_sine', 'kdv', 'chafee-infante', 'allen_cahn',
            'convection_diffusion', 'fisher', 'fisher_linear', 'PDE_divide', 'PDE_compound')
LABELS = old.LABELS | {'pdefind': 'PDE-FIND', 'pysindy': 'PySINDy',
                     'weakform': 'WSINDy', 'deepmod': 'DeepMoD'}
COHORTS = {'historical_20261016': '旧快照', 'pde_extension_20261017': '新 PDE 扩展'}
METRICS = old.METRICS | {'coefficient_error': '系数相对 L2 误差',
    'coefficient_recovery': '系数恢复', 'fit_seconds': '拟合时间（秒）',
    'eligible_support_recovery': '认证参考子集项结构恢复'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def checked_path(root, relative):
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    return path


def verify_new():
    formal = NEW / 'formal'
    for name in ('benchmark_freeze.json', 'matrix_plan.json'):
        index = read(formal / name)
        assert digest({k: v for k, v in index.items() if k != 'sha256'}) == index['sha256'], name
        for relative, expected in index['input_sha256'].items():
            assert sha(checked_path(formal, relative)) == expected, relative
    archive = read(NEW / 'download_verification.json')
    assert sha(NEW / 'frozen_results.tar.gz') == archive['archive_sha256']
    assert archive['freeze_sha256'] == read(formal / 'benchmark_freeze.json')['sha256']
    assert archive['matrix_plan_sha256'] == read(formal / 'matrix_plan.json')['sha256']
    assert read(NEW / 'pipeline_status.json')['stage'] == 'complete'
    assert read(NEW / 'pipeline.exit') == 0
    manifest, rows = read(formal / 'benchmark_manifest.json'), read(formal / 'benchmark_summary.json')
    assert len(rows) == len(manifest['cases']) == 250
    assert {r['profile'] for r in rows} == {'full'}
    assert {r['metric_protocol_version'] for r in rows} == {'2.1'}
    assert {r['dataset_protocol_version'] for r in rows} == {'domain-v5'}
    assert {r['benchmark_role'] for r in rows} == {'diagnostic'}
    keys = [(r['model'], r['dataset'], r['seed']) for r in rows]
    assert len(set(keys)) == 250
    assert set(keys) == {(m, d, s) for m in METHODS for d in DATASETS for s in range(5)}
    assert Counter(r['status'] for r in rows) == {'ok': 250}
    assert all(old.finite(r.get('equation_residual_nrmse')) for r in rows)
    declared = set()
    cases = {c['name']: c for c in manifest['cases']}
    for case in manifest['cases']:
        specs = read(formal / 'cases' / case['name'] / 'targets.json')
        assert len(specs) == 1 and 'preparation_row' not in specs[0]
        declared.add((case['name'], specs[0]['target_id']))
    assert {(r['name'], r['target_id']) for r in rows} == declared
    for row in rows:
        terminal = formal / 'cases' / row['name'] / 'targets' / row['target_id'] / 'result.json'
        assert read(terminal) == row, str(terminal)
        assert row['experiment_identity_hash'] == manifest['experiment_identity_hash']
        assert row['execution_policy']['timeout'] == 300
        assert cases[row['name']]['dataset_params'] == {}
    return manifest, rows


def describe(rows, task, metric):
    if metric == 'eligible_support_recovery':
        rows = [r for r in rows if r.get('recovery_eligible') is True]
        metric = 'pde_support_recovery'
    if metric == 'coefficient_recovery':
        values = [r[metric] for r in rows if r['status'] == 'ok' and old.finite(r.get(metric))]
        assert all(v in (0, 1) for v in values)
        return dict(finite=len(values), median=None, q1=None, q3=None,
                    recovery_yes=sum(values), recovery_evaluable=len(values))
    return old.describe(rows, task, metric)


def make_record(cohort, task, method, dataset, rows, planned=5, scope='dataset', target=''):
    statuses = Counter(r['status'] for r in rows)
    return dict(cohort=cohort, task=task, scope=scope, family=old.family(dataset) if task == 'sr' else task.upper(),
        method=method, dataset=dataset, target=target, planned=planned, terminal=len(rows),
        ok=statuses['ok'], timeout=statuses['timeout'], error=statuses['error'],
        skipped=statuses['skipped'], missing=planned-len(rows),
        metrics={metric: describe(rows, task, metric) for metric in METRICS})


def table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |'] + [
        '| ' + ' | '.join(map(str, row)) + ' |' for row in rows]


def count_cell(record, metric):
    s = record['metrics'][metric]
    return f"{s['recovery_yes']}/{s['recovery_evaluable']}" if s['recovery_evaluable'] else '—（不可评）'


def coverage(records):
    buckets = defaultdict(list)
    for r in records:
        if r['scope'] != 'rhs':
            buckets[r['cohort'], r['method'], r['task']].append(r)
    result = []
    for (cohort, model, task), subset in buckets.items():
        result.append(dict(cohort=cohort, model=model, task=task,
            **{k: sum(r[k] for r in subset) for k in ('planned','terminal','ok','timeout','error','skipped','missing')},
            finite=sum(r['metrics']['nrmse']['finite'] for r in subset)))
    return result


def write_csv(records):
    fields = ['cohort','task','scope','family','dataset','target','method','planned','terminal','ok',
              'timeout','error','skipped','missing','metric','source_field','finite','missing_metric_ok',
              'median','q1','q3','recovery_yes','recovery_evaluable']
    with (OUT / 'dataset_metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in records:
            for metric, s in r['metrics'].items():
                writer.writerow({k:r[k] for k in fields if k in r} | dict(metric=metric,
                    source_field='pde_support_recovery (recovery_eligible=true)' if metric=='eligible_support_recovery'
                                 else old.metric_field(r['task'], metric),
                    missing_metric_ok=r['ok']-s['finite'], **s))
    all_methods = (*old.METHODS, *METHODS)
    headers = ['cohort','task','scope','dataset','target'] + [
        f'{model}_{key}' for model in all_methods for key in ('median','finite','planned','terminal')]
    grouped = defaultdict(dict)
    for r in records:
        grouped[r['cohort'],r['task'],r['scope'],r['dataset'],r['target']][r['method']] = r
    with (OUT / 'nrmse_matrix.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        for key, group in grouped.items():
            values = list(key)
            for model in all_methods:
                r = group.get(model)
                values += [r['metrics']['nrmse']['median'],r['metrics']['nrmse']['finite'],r['planned'],r['terminal']] if r else ['']*4
            writer.writerow(values)


def write_markdown(records, old_rows, new_rows, manifest, summary, old_records):
    cov = summary['coverage']
    indexed = {(r['cohort'],r['method'],r['dataset']):r for r in records if r['task']=='pde'}
    new = lambda model, dataset: indexed['pde_extension_20261017', model, dataset]
    finished = datetime.fromisoformat(read(NEW/'pipeline_status.json')['updated_at']).astimezone(timezone(timedelta(hours=8)))
    lines = ['# 基线 × 数据集综合实验结果（2026-10-17）', '',
        '**本报告汇总原 2026-10-16 冻结表及新完成的 PDE 扩展实验。** 新实验于 '+finished.strftime('%Y-%m-%d %H:%M:%S')+
        '（台北时间）完成，250/250 个目标均为 `ok` 且方程残差有限。旧表的 SR、ODE 及 DLGA 数据逐项保留。', '',
        '覆盖 **9 种方法、286 个数据集／系统注册名称**（SR 268、ODE 系统 8、PDE 10；ODE 另有 17 个 RHS）。'
        '共收录 **2806 条终态记录**；按来源实验的计划合计 5965 个目标，旧快照仍有 3159 个目标未见终态。'
        '这些是归档快照的覆盖数，不表示旧服务器队列的当前状态。', '',
        '新、旧 PDE 的 metric 版本都是 2.1，但数据协议、代码身份和资源配置不同；因此保留批次，'
        '**不把两批 DLGA 在相同数据集与 seed 上的记录合成 10 次重复，也不跨批次计算统一排名**。'
        '新旧共有 15 个 DLGA／数据集／seed 标签；5965 是跨实验的计划次数，并非互不重复的实验条件数。50 个新 smoke 目标不计入本报告。', '',
        '[可搜索的离线表格](../results/analysis/combined_results_20261017/index.html) · '
        '[完整指标 CSV](../results/analysis/combined_results_20261017/dataset_metrics.csv) · '
        '[NRMSE 宽表 CSV](../results/analysis/combined_results_20261017/nrmse_matrix.csv) · '
        '[汇总与校验信息](../results/analysis/combined_results_20261017/summary.json)', '',
        '## 结果要点', '',
        '- 新 PDE 矩阵全部完成。WSINDy 的登记项结构匹配为 **40/50**，PDE-FIND 为 **15/50**；'
        'PySINDy、DLGA、DeepMoD 均为 **0/50**。运行成功与结构恢复须分别解读。',
        '- 按现有 `recovery_eligible` 认证规则，只有 Burgers、KdV、Chafee–Infante 三个数据集可进入认证参考子集：'
        'WSINDy **15/15**，PDE-FIND **10/15**，其余三种方法均 **0/15**。其余数据集的匹配仅作为登记参考诊断。',
        '- 在新批次各数据集的五方法 NRMSE 中位数中，PySINDy 在 5 个数据集最低，WSINDy 在 4 个最低，'
        'PDE-FIND 在 1 个最低。这是逐数据集描述，不是跨数据集平均排名，也不代表显著性结论。',
        '- PySINDy 在多个数据集上残差较小，但导出方程仍有额外项，因此精确项集合恢复为零。'
        '例如 Burgers 的 seed 0 除 `u*u_x`、`u_xx` 外还含 `u*u_xx`、`u²*u_x` 和 `u²*u_xx`。',
        '- 残差与结构并不等价：WSINDy 在 `PDE_compound` 的项结构匹配为 5/5，但残差中位数约 59.69；'
        '该数据集缺少带系数参考式与已认证来源，仍需独立核验。', '',
        '## 实验范围与统计口径', '',
        *table(['来源','完成／快照时间（台北）','数据协议','指标协议','运行设置'],[
            ['旧快照','2026-10-16 14:10:52','domain-v4','2.1','原冻结 D2 资源槽，每槽 4 核／28 GB'],
            ['新 PDE 扩展',finished.strftime('%Y-%m-%d %H:%M:%S'),'domain-v5','2.1',
             '独立扩展矩阵；CPU 3 worker、GPU 1 worker；每 worker 3 线程']]), '',
        '新批次使用 seeds 0–4、无新增噪声、最后 20% 时间块留出；拟合预算为 300 秒，另有 5 秒退出宽限，'
        '加载和评分各有独立的 300 秒上限。DeepMoD 的 50 个目标均以 `time_limit` 软截止返回有效候选，'
        '状态是 `ok`，不计为硬超时失败。稀疏方法可提前完成，不强制用满 300 秒。', '',
        'NRMSE 单元格为 **成功且指标有限的中位数 [有限成功数/计划数]**；缺失与失败不填零。'
        'SR 使用留出预测误差，ODE 使用 RHS 导数预测误差，PDE 使用方程残差；三类数值不合并平均。'
        'PDE 残差以留出目标 `u_t` 的标准差归一化，各方法采用共同有限差分评分域。'
        'ODE 系统行合并各 RHS 记录后取中位数，分母为 RHS 数 × 5；RHS 明细是同一批数据的另一视图。', '',
        '旧表的有限成功子集并不一致，不能据残差中位数直接给旧方法完整排名。CSV 保留 Q1/Q3（inclusive 插值）、'
        '状态数、计划分母和恢复可评估分母。CSV 中多指标行以及 ODE 系统／RHS 行不可直接累加目标数。', '',
        '## 覆盖和执行状态', '',
        *table(['批次','方法','任务','计划','终态','成功','超时','错误','未见终态','主误差有限'],[
            [COHORTS[r['cohort']],LABELS[r['model']],r['task'].upper(),r['planned'],r['terminal'],r['ok'],
             r['timeout'],r['error'],r['missing'],r['finite']] for r in cov]), '',
        *table(['汇总范围','计划','终态','成功','超时','错误','未见终态','主误差有限'],[
            [label,*[value[k] for k in ('planned','terminal','ok','timeout','error','missing','finite')]]
            for label,value in [('旧快照',summary['historical']),('新 PDE 扩展',summary['extension']),
                                ('跨批次记录合计',summary['combined'])]]), '',
        '## PDE：新五方法结果与旧 DLGA 对照', '',
        '前五列来自同一个新矩阵，每格均有 5 个有限成功结果；最后一列来自旧快照，仅作历史对照。'
        '旧列的“未规划”表示旧矩阵中没有该数据集。', '',
        *table(['数据集',*[LABELS[m] for m in METHODS],'DLGA（旧快照）'],[
            [d,*[old.cell(new(m,d)) for m in METHODS],
             old.cell(indexed['historical_20261016','dlga',d]) if ('historical_20261016','dlga',d) in indexed else '未规划']
            for d in DATASETS]), '',
        '### 新 PDE 项结构匹配', '',
        '下表按现有导出表达式与登记参考的加法项集合精确匹配，忽略系数大小；未额外设置事后删项阈值。'
        '除三个认证参考数据集外，数值表示登记参考诊断，不升级为已认证物理方程恢复。', '',
        *table(['数据集',*[LABELS[m] for m in METHODS]],[[d,*[count_cell(new(m,d),'pde_support_recovery') for m in METHODS]] for d in DATASETS]), '',
        '### 新 PDE 系数误差', '',
        '系数误差为发现与参考系数向量在项并集上的相对 L2 误差，中位数越低越好。'
        '它可能被主项系数主导，较小数值不保证小系数高阶项存在；例如 PDE-FIND 的 KdV 项结构未恢复，系数误差仍约 0.002923。'
        '四个没有带系数参考式的数据集显示“不可评”。', '',
        *table(['数据集',*[LABELS[m] for m in METHODS]],[[d,*[
            old.number(new(m,d)['metrics']['coefficient_error']['median']) if new(m,d)['metrics']['coefficient_error']['finite'] else '不可评'
            for m in METHODS]] for d in DATASETS]), '',
        '### 参考依据和恢复资格', '',
        *table(['数据集','带系数参考式','reference_status','source_status','认证恢复子集'],[
            [d,'有' if next(r for r in new_rows if r['dataset']==d).get('coefficient_error') is not None else '无（仅项结构）',
             next(r for r in new_rows if r['dataset']==d)['dataset_metadata']['reference_status'],
             next(r for r in new_rows if r['dataset']==d)['dataset_metadata']['source_status'],
             '是' if next(r for r in new_rows if r['dataset']==d)['recovery_eligible'] else '否'] for d in DATASETS]), '',
        '这一区分比“10 个已知真实方程”更严格：六个数据集有带系数参考式，其中三个满足当前来源与参考认证规则；'
        '另四个只支持登记项结构比较。', '',
        '### 方法实现、恢复和耗时', '',
    ]
    method_rows=[]
    for model in METHODS:
        group=[r for r in new_rows if r['model']==model]
        eligible=[r for r in group if r.get('recovery_eligible')]
        coeff=[r for r in group if r.get('coefficient_recovery') is not None]
        method_rows.append([LABELS[model],f"{sum(r['pde_support_recovery']==1 for r in group)}/50",
            f"{sum(r['pde_support_recovery']==1 for r in eligible)}/{len(eligible)}",
            f"{sum(r['coefficient_recovery']==1 for r in coeff)}/{len(coeff)}",
            old.number(statistics.median(r['fit_seconds'] for r in group)),
            old.number(statistics.median(r['runtime'] for r in group))])
    lines += table(['方法','全部登记项匹配','认证参考项恢复','有系数参考的系数恢复','fit 中位数/s','runtime 中位数/s'],method_rows)+['',
        '系数恢复沿用逐系数 `rtol=0.05, atol=1e-8` 判定。新方法耗时是各自 50 个目标的描述性中位数；'
        '`fit_seconds` 是受预算约束的拟合时间，`runtime` 沿用原始行字段，不代表整条队列耗时。'
        '新旧资源协议不同，不据此推断跨批次加速比。', '',
        '实现范围：PDE-FIND 是仓库的“单次最小二乘＋阈值”适配器，并非上游 STRidge；'
        'PySINDy 使用 SR3/L0 与三次多项式库；WSINDy 使用弱形式稀疏回归；'
        'DLGA 沿用原主实验的 float64 小网络、最小二乘代理与遗传搜索配置；DeepMoD 使用三次多项式和三阶空间导数。'
        '这些候选库不能表达 `PDE_divide` 的 `u_x/x`；该数据集仍保留在计划分母中。', '',
        '## ODE 与 SR：保留原冻结表的全部结果', '',
        '以下数据均来自 2026-10-16 14:10:52 旧快照，未用新 PDE 数据改写。'
        'SR 按原家族顺序排列，包含全部 268 个注册数据集；ODE 同时保留 8 个系统与 17 个 RHS 的视图。', '']
    original_text=old.MARKDOWN.read_text(encoding='utf-8')
    for title, subset, methods in old.groups(old_records):
        entries=old.table_rows(subset,methods)
        # Exact per-row match against the requested source document, not only rounded recomputation.
        rendered=['| '+name+' | '+' | '.join(old.cell(r) for r in values)+' |' for name,values in entries]
        assert all(line in original_text for line in rendered), title
        if title.startswith('PDE'):
            continue
        lines += [f'### {title}（{len(entries)} 行）','',
                  '| 数据集／目标 | '+' | '.join(LABELS[m] for m in methods)+' |',
                  '|---|'+'---:|'*len(methods),*rendered,'']
    lines += ['## 信息准则与其他指标','',*old.ic_table(old_rows),'',
        '旧数据的条件训练 AIC/BIC 与留出误差／复杂度代理 IC 保留各自含义，可用目标数重叠，不能相加。'
        '新 PDE 250 行没有 AIC/BIC 或代理 IC，未通过残差自行补造。两批均不在此汇总 PIC。'
        '旧逐目标 IC 见 [原始 CSV](../results/analysis/cloud_results_20261016/results.csv)；'
        'MSE、R²、表达式复杂度、runtime 和恢复指标见本报告配套完整指标 CSV。PDE 未提供的 R² 保留为空。','',
        '## 来源与复现','',
        '- 旧表：[benchmark_results_tables_20261016.md](benchmark_results_tables_20261016.md)；'
        '底层来源为 [旧逐目标记录](../results/analysis/cloud_results_20261016/rows.jsonl) 和原始汇总。',
        '- 新结果：[冻结逐目标汇总](../results/analysis/pde_extension_20261017/formal/benchmark_summary.json)、'
        '[冻结矩阵](../results/analysis/pde_extension_20261017/formal/benchmark_manifest.json)、'
        '[冻结封印](../results/analysis/pde_extension_20261017/formal/benchmark_freeze.json)、'
        '[下载校验](../results/analysis/pde_extension_20261017/download_verification.json)。',
        '- 汇总脚本：[build_combined_results_20261017.py](../scripts/build_combined_results_20261017.py)。'
        '使用 Python 标准库重新核对旧快照的 2556 行及全部原表单元格、新批次的 250 行、'
        '五个种子、输入哈希和首次终态；所有恢复与误差统计均从原始行计算。',
        '- 配套 CSV、HTML 和原始结果位于 Git 忽略的 `results/`；单独分享本 Markdown 不会自动携带这些文件。','',
        '新批次身份：`'+manifest['experiment_identity_hash']+'`。',
        '新批次源码指纹：`'+manifest['identity']['code']['sha256']+'`。','',
        '```powershell','python scripts/build_combined_results_20261017.py','```','']
    DOC.write_text('\n'.join(lines),encoding='utf-8')


def write_html(records, summary):
    payload=json.dumps(dict(records=records,labels=LABELS,cohorts=COHORTS,metrics=METRICS),
                       ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>综合实验结果 · 2026-10-17</title><style>
body{font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif;background:#f5f7fa;color:#203047;margin:0}main{max-width:1500px;margin:auto;padding:24px}h1{font-size:26px}a{color:#17639c}.tools{display:flex;flex-wrap:wrap;gap:12px;background:white;padding:16px;position:sticky;top:0;border:1px solid #d6e0e8}select,input{font:inherit;padding:5px}table{border-collapse:collapse;width:100%;background:white}th,td{padding:8px 12px;border-bottom:1px solid #dce3ec;text-align:right;white-space:nowrap}th{background:#e8eef5}th:first-child,td:first-child{text-align:left}.wrap{overflow:auto}tr:hover{background:#edf4fb}.note{color:#526477}#counts{font-weight:600}</style>
<main><h1>综合实验结果 · 2026-10-17</h1>
<p>旧快照 2556 条终态 + 新 PDE 250 条终态，共 2806 条；新 PDE 全部成功且残差有限。批次保留，旧快照未见终态 3159 个。</p>
<p><a href="../../../docs/benchmark_results_combined_20261017.md">完整 Markdown 报告</a> · <a href="dataset_metrics.csv">完整指标 CSV</a> · <a href="nrmse_matrix.csv">NRMSE 宽表</a> · <a href="summary.json">校验信息</a></p>
<p class="note">同名 NRMSE：SR 为预测误差、ODE 为 RHS 导数误差、PDE 为方程残差；不跨任务或批次平均。恢复列显示“成功数/可评估数”。认证参考项恢复仅计 recovery_eligible=true。ODE 系统与 RHS 是同一批记录的两种视图，请勿重复计数。</p>
<div class="tools"><label>批次 <select id="cohort"><option value="">全部</option><option value="historical_20261016">旧快照</option><option value="pde_extension_20261017">新 PDE 扩展</option></select></label>
<label>任务 <select id="task"><option value="">全部</option><option>sr</option><option>ode</option><option>pde</option></select></label>
<label>ODE 视图 <select id="scope"><option value="system">系统</option><option value="rhs">RHS</option></select></label>
<label>指标 <select id="metric"></select></label><label>搜索 <input id="search" type="search" placeholder="数据集或方法"></label></div>
<p id="counts"></p><div class="wrap"><table><thead><tr><th>数据集／目标</th><th>批次</th><th>方法</th><th>值</th><th>Q1–Q3</th><th>有限／计划</th><th>成功</th><th>超时</th><th>错误</th><th>未见终态</th></tr></thead><tbody id="body"></tbody></table></div></main>
<script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent),el=id=>document.getElementById(id);for(const [key,label]of Object.entries(data.metrics)){const op=document.createElement('option');op.value=key;op.textContent=label;el('metric').append(op)}
const fmt=v=>v===null?'—':Number(v).toPrecision(5).replace(/(\\.\\d*?[1-9])0+(e|$)/,'$1$2').replace(/\\.0+(e|$)/,'$1');
function update(){const cohort=el('cohort').value,task=el('task').value,scope=el('scope').value,key=el('metric').value,q=el('search').value.toLowerCase();const rows=data.records.filter(r=>(!cohort||r.cohort===cohort)&&(!task||r.task===task)&&(r.task!=='ode'||r.scope===scope)&&(!q||[r.dataset,r.target,r.method,data.labels[r.method]].join(' ').toLowerCase().includes(q)));el('body').replaceChildren();const frag=document.createDocumentFragment();for(const r of rows){const s=r.metrics[key],tr=document.createElement('tr'),v=s.recovery_evaluable!==null?(s.recovery_evaluable?s.recovery_yes+'/'+s.recovery_evaluable:'不可评'):fmt(s.median);for(const value of [r.dataset+(r.target?' · '+r.target:''),data.cohorts[r.cohort],data.labels[r.method],v,s.q1===null?'—':fmt(s.q1)+'–'+fmt(s.q3),s.finite+'/'+r.planned,r.ok,r.timeout,r.error,r.missing]){const td=document.createElement('td');td.textContent=value;tr.append(td)}frag.append(tr)}el('body').append(frag);el('counts').textContent='显示 '+rows.length+' 个方法／数据集组合；当前筛选的计划 '+rows.reduce((n,r)=>n+r.planned,0)+'，终态 '+rows.reduce((n,r)=>n+r.terminal,0)}
for(const id of ['cohort','task','scope','metric','search'])el(id).addEventListener('input',update);update();</script></html>'''
    # From results/analysis/combined_results_20261017 to the repository root: three parents.
    (OUT/'index.html').write_text(page.replace('__DATA__',payload),encoding='utf-8')


def main():
    old_rows=[json.loads(line) for line in (old.SOURCE/'rows.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    old_summary=read(old.SOURCE/'summary.json')
    with (old.SOURCE/'dataset_summary.csv').open(encoding='utf-8-sig',newline='') as stream:
        old.validate(old_rows,old_summary,list(csv.DictReader(stream)))
    old_records=old.build_records(old_rows,old_summary)
    manifest,new_rows=verify_new()
    records=[]
    for original in old_records:
        group=[r for r in old_rows if (r['task'],r['model'],r['dataset'])==
               (original['task'],original['method'],original['dataset']) and
               (original['scope']!='rhs' or r['target']==original['target'])]
        records.append(make_record('historical_20261016',original['task'],original['method'],original['dataset'],
                       group,original['planned'],original['scope'],original['target']))
    for dataset in DATASETS:
        for model in METHODS:
            records.append(make_record('pde_extension_20261017','pde',model,dataset,
                           [r for r in new_rows if r['model']==model and r['dataset']==dataset]))
    cov=coverage(records)
    def total(selected):
        return {key:sum(r[key] for r in selected) for key in ('planned','terminal','ok','timeout','error','skipped','missing','finite')}
    summary=dict(historical=total([r for r in cov if r['cohort']=='historical_20261016']),
                 extension=total([r for r in cov if r['cohort']=='pde_extension_20261017']),
                 combined=total(cov),coverage=cov)
    assert summary['combined']==dict(planned=5965,terminal=2806,ok=2339,timeout=334,error=133,skipped=0,missing=3159,finite=2338)
    overlap={(r['model'],r['dataset'],r['target'],r['seed']) for r in old_rows}&{(r['model'],r['dataset'],r['target'],r['seed']) for r in new_rows}
    assert len(overlap)==15
    assert len({r['dataset'] for r in old_rows+new_rows})==286
    assert len({r['model'] for r in old_rows+new_rows})==9
    expected_support={'pdefind':15,'pysindy':0,'weakform':40,'dlga':0,'deepmod':0}
    assert {m:sum(r['pde_support_recovery']==1 for r in new_rows if r['model']==m) for m in METHODS}==expected_support
    wins=Counter()
    for dataset in DATASETS:
        subset=[r for r in records if r['cohort']=='pde_extension_20261017' and r['dataset']==dataset]
        wins[min(subset,key=lambda r:r['metrics']['nrmse']['median'])['method']]+=1
    assert wins=={'pysindy':5,'weakform':4,'pdefind':1}
    # Cross-check the server's independently produced 50-row report.
    for server in read(NEW/'main_results.json'):
        record=next(r for r in records if r['cohort']=='pde_extension_20261017' and
                    (r['method'],r['dataset'])==(server['model'],server['dataset']))
        assert math.isclose(record['metrics']['nrmse']['median'],server['residual_nrmse_median'],rel_tol=1e-12)
        assert record['metrics']['pde_support_recovery']['recovery_yes']==server['support_recovered']
    summary.update(distinct_dataset_names=286,distinct_methods=9,overlapping_target_labels_between_cohorts=15,
                   new_pde_support_diagnostic=expected_support,new_pde_lowest_median_nrmse_datasets=dict(wins),
                   new_experiment_identity_hash=manifest['experiment_identity_hash'],
                   new_download_verification=read(NEW/'download_verification.json'),
                   checks={'old_snapshot_rows':2556,'new_first_terminals':250,
                           'new_freeze_and_matrix_hashes':True,'old_markdown_cells_preserved':True},
                   generator_sha256={path.name:sha(path) for path in (Path(__file__),Path(old.__file__))},
                   source_sha256={str(path.relative_to(ROOT)).replace('\\','/'):sha(path) for path in (
                       old.MARKDOWN,old.SOURCE/'rows.jsonl',old.SOURCE/'summary.json',old.SOURCE/'dataset_summary.csv',
                       NEW/'formal/benchmark_manifest.json',NEW/'formal/benchmark_summary.json',NEW/'formal/benchmark_freeze.json')})
    OUT.mkdir(parents=True,exist_ok=True)
    write_markdown(records,old_rows,new_rows,manifest,summary,old_records)
    write_csv(records)
    write_html(records,summary)
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (OUT/'rows.jsonl').open('w',encoding='utf-8') as stream:
        for cohort,rows in [('historical_20261016',old_rows),('pde_extension_20261017',new_rows)]:
            for r in rows:
                stream.write(json.dumps(dict(cohort=cohort,**r),ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(summary['combined'],ensure_ascii=False))
    print(DOC.relative_to(ROOT))
    print(OUT.relative_to(ROOT))


if __name__=='__main__':
    main()
