"""Rebuild the 2026-10-16 snapshot tables using only exact chunk summaries (stdlib)."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import tarfile
from cloud_result_ic import supplement_cloud_row

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = '2026-10-16 14:10:52 UTC+08:00'
METRICS = ['test_mse', 'test_nrmse', 'test_r2', 'equation_residual_mse', 'equation_residual_nrmse', 'expression_complexity', 'runtime']

def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

def canonical(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',', ':'))

def clean(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v) for v in x]
    return x

def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: canonical(v) if isinstance(v, (dict, list)) else clean(v) for k, v in row.items() if k in fields})

def stats(rows):
    out = {'finished_targets': len(rows), 'status_counts': dict(Counter(r.get('status', 'missing') for r in rows))}
    for key in METRICS + ['aic', 'bic', 'aic_proxy_heldout_complexity', 'bic_proxy_heldout_complexity']:
        vals = [r[key] for r in rows if r.get('status') == 'ok' and finite(r.get(key))]
        out[key + '_finite_ok'] = len(vals)
        # IC values cannot be pooled across observations/datasets.
        if key in METRICS:
            out[key + '_median_ok'] = statistics.median(vals) if vals else None
    out['training_ic_status_counts'] = dict(Counter(r['training_ic_status'] for r in rows))
    out['proxy_ic_status_counts'] = dict(Counter(r['proxy_ic_status'] for r in rows))
    return out

def grouped(rows, keys):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k) for k in keys)].append(r)
    return [{**dict(zip(keys, key)), **stats(rs)} for key, rs in sorted(groups.items(), key=lambda v: str(v[0]))]

def paired(rows):
    groups = defaultdict(dict)
    excluded = 0
    for r in rows:
        fp = r['pair_asset_fingerprints']
        required = ['task','dataset','instance','target','seed','split','n_test','metric_protocol_version','dataset_protocol_version']
        if not fp or any(r.get(k) is None for k in required):
            excluded += 1
            continue
        key = tuple(r.get(k) for k in required) + (canonical(fp), r.get('evaluation_protocol_hash'), canonical(r['pair_observation_protocol']))
        if r['model'] in groups[key]:
            raise ValueError('Ambiguous paired observations: ' + str(key))
        groups[key][r['model']] = r
    counts = defaultdict(Counter)
    for key, models in groups.items():
        for a, b in itertools.combinations(sorted(models), 2):
            x, y = models[a], models[b]
            c = counts[(key[0], a, b)]
            c['common_finished_targets'] += 1
            c['method_a_failures'] += x.get('status') != 'ok'
            c['method_b_failures'] += y.get('status') != 'ok'
            c['both_failures'] += x.get('status') != 'ok' and y.get('status') != 'ok'
            if x.get('status') != 'ok' or y.get('status') != 'ok':
                continue
            metric = 'equation_residual_nrmse' if key[0] == 'pde' else 'test_nrmse'
            checks = [('nrmse', metric, True)]
            same_proxy = x.get('proxy_ic_n') == y.get('proxy_ic_n') and x.get('proxy_ic_domain') == y.get('proxy_ic_domain')
            checks += [('proxy_aic','aic_proxy_heldout_complexity',same_proxy), ('proxy_bic','bic_proxy_heldout_complexity',same_proxy)]
            for label, field, comparable in checks:
                u, v = x.get(field), y.get(field)
                if comparable and finite(u) and finite(v):
                    c[label + '_both_finite'] += 1
                    result = 'tie' if math.isclose(u, v, rel_tol=1e-12, abs_tol=1e-12) else ('a_win' if u < v else 'b_win')
                    c[label + '_' + result] += 1
    fields = ['common_finished_targets','method_a_failures','method_b_failures','both_failures'] + [f'{m}_{s}' for m in ['nrmse','proxy_aic','proxy_bic'] for s in ['both_finite','a_win','tie','b_win']]
    output = [{'task': t, 'method_a': a, 'method_b': b, **{k: c[k] for k in fields}, 'observation_identity': 'manifest fingerprints and protocol only; sample bytes not independently verified'} for (t,a,b), c in sorted(counts.items())]
    return output, excluded

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, default=ROOT / 'results/knowledge_discover_results_20261016_summary.tar.gz')
    p.add_argument('--output', type=Path, default=ROOT / 'results/analysis/cloud_results_20261016')
    args = p.parse_args()
    archive_hash = hashlib.file_digest(args.archive.open('rb'), 'sha256').hexdigest()
    cases, plans, rows, seen = {}, {}, [], {}
    with tarfile.open(args.archive) as tar:
        members = tar.getmembers()
        for m in members:
            match = re.fullmatch(r'experiments/([^/]+)/matrix/(?:chunks/[^/]+/benchmark_manifest.json|matrix_plan.json)', m.name)
            if not match:
                continue
            data = json.load(tar.extractfile(m))
            exp = match[1]
            if m.name.endswith('matrix_plan.json'):
                plans[exp] = {k: data[k] for k in ['n_cases','n_declared_targets']}
            else:
                for case in data['cases']:
                    ck = (exp, case['name'])
                    if ck in cases and canonical(cases[ck]) != canonical(case):
                        raise ValueError('Conflicting case manifest: ' + str(ck))
                    cases[ck] = case
        for m in members:
            match = re.fullmatch(r'experiments/([^/]+)/matrix/chunks/[^/]+/benchmark_summary.json', m.name)
            if not match:
                continue
            blob = tar.extractfile(m).read()
            for index, raw in enumerate(json.loads(blob)):
                exp = match[1]
                identity = (exp, raw['name'], raw.get('target_id'))
                if identity in seen:
                    raise ValueError('Duplicate target (not silently selected): ' + str((identity,seen[identity],m.name,index)))
                seen[identity] = (m.name,index)
                case = cases[(exp,raw['name'])]
                supplement = supplement_cloud_row(raw)
                if set(raw) & set(supplement):
                    raise ValueError('Supplement would overwrite source columns')
                rows.append({**raw, **supplement, 'source_archive': args.archive.name, 'source_archive_sha256': archive_hash,
                    'source_member': m.name, 'source_member_sha256': hashlib.sha256(blob).hexdigest(), 'source_row_index': index,
                    'source_row_sha256': hashlib.sha256(canonical(raw).encode()).hexdigest(), 'experiment': exp,
                    'pair_asset_fingerprints': {k:v for k,v in case.get('asset_fingerprints',{}).items() if k.startswith('dataset:')},
                    'pair_observation_protocol': {'dataset_params':case.get('dataset_params',{}), 'run_params':{k:case.get('run_params',{}).get(k) for k in ['test_size','max_train_samples','sample_ratio','dlga_samples']}, 'evaluation':case.get('evaluation',{})}})
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'rows.jsonl').open('w', encoding='utf-8') as f:
        for r in rows:
            # Python JSON preserves any source NaN/Infinity; summary/CSV use null/blank.
            f.write(json.dumps(r,ensure_ascii=False) + '\n')
    fields = ['experiment','name','task','model','dataset','instance','target','target_id','seed','split','status','n_train','n_test'] + METRICS + ['aic','bic','training_ic_status','training_ic_reason','information_criterion_n','information_criterion_k','training_ic_domain','aic_proxy_heldout_complexity','bic_proxy_heldout_complexity','proxy_ic_status','proxy_ic_n','proxy_ic_k','proxy_ic_domain','proxy_ic_reason','source_member','source_row_index']
    write_csv(args.output / 'results.csv', rows, fields)
    mt = grouped(rows,['model','task'])
    ds = grouped(rows,['model','task','dataset'])
    write_csv(args.output / 'model_task_summary.csv', mt)
    write_csv(args.output / 'dataset_summary.csv', ds)
    pairs, excluded = paired(rows)
    write_csv(args.output / 'paired_comparisons.csv', pairs)
    planned = Counter((c['model'],c['task']) for c in cases.values())
    summary = {'snapshot':SNAPSHOT,'archive':args.archive.name,'archive_sha256':archive_hash,
        'selection':'Exact experiments/<exp>/matrix/chunks/<chunk>/benchmark_summary.json only; no reports or by_baseline mirrors',
        'matrix_plans':plans,'planned_cases':sum(v['n_cases'] for v in plans.values()),'planned_targets':sum(v['n_declared_targets'] for v in plans.values()),
        'unique_chunk_manifest_cases':len(cases),'planned_cases_by_model_task':[{'model':m,'task':t,'cases':n} for (m,t),n in sorted(planned.items())],
        'overall':stats(rows),'model_task':mt,'paired_groups':len(pairs),'pairing_excluded_missing_identity_or_fingerprint':excluded,
        'notes':['Target counts describe this incomplete snapshot, not final method performance.','Medians pool heterogeneous datasets and are descriptive only. IC values are never averaged across datasets.','AIC/BIC are conditional training plugin scores only where available; exploratory heldout complexity proxy columns are not formal AIC/BIC.','Pair identity uses declared dataset fingerprints and observation protocols; sample bytes have not been independently verified.','rows.jsonl preserves full original columns and any original nonstandard NaN/Infinity JSON values.']}
    (args.output / 'summary.json').write_text(json.dumps(clean(summary),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(clean(summary),ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
