#!/usr/bin/env python3
"""Experiment 5: choice-kernel initialization mechanisms (real-data K0 variants + parameter recovery).

Phases (each cached per participant; rerunning resumes):
  real       eight K0 variants fitted on S1-S4, scored on S5 (all 213 participants)
  synthetic  realistic recovery: each participant's own Part-A fits as truth, generators
             {released, uniform_0.5}, several seeds, refit under {released, uniform_0.5, minmax}
  grid       controlled recovery: true alpha_CK x beta_CK grid on a reproducible subset of sequences
Only training/initialization tables are read; the final test is never accessed.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from reward_pairs import ck_init_mechanisms as m  # noqa: E402
from reward_pairs.data import load_rdata  # noqa: E402
from reward_pairs.history_diagnostics import make_fold, permitted_inputs, training_initialization  # noqa: E402

TOLERANCE = dict(released_vs_saved_deviance=1e-6, released_s5_nll=1e-6)


def load(data, folds, participants, with_pretest=False):
    objects, order = permitted_inputs(load_rdata(data))  # training + initialization tables only
    frame = objects['data_4_analysis_RP_training']
    s5 = {r['participant']: r for r in folds['rows'] if r['session'] == 5}
    out = []
    for p in participants:
        fold = make_fold(frame.loc[frame.ID.astype(str).eq(p)].copy(), p, 5)
        q0, v = training_initialization(objects, p)
        out.append((p, order.index(p)+1, fold, q0, v, s5[p]['models']['ck']) + ((m.pretest_proportions(objects, p),) if with_pretest else ()))
    return out


def run(fn, jobs, workers, cache, tag, label):
    cache.mkdir(parents=True, exist_ok=True)
    rows, todo = {}, []
    for job in jobs:
        f = cache/f'{job[0]}.json'
        if f.exists() and json.loads(f.read_text()).get('config_tag') == tag:
            rows[job[0]] = json.loads(f.read_text())['row']
        else:
            todo.append(job)
    start = perf_counter()
    if todo:
        with ProcessPoolExecutor(workers) as pool:
            for i, row in enumerate(pool.map(fn, todo), 1):
                (cache/f'{row["participant"]}.json').write_text(json.dumps(dict(config_tag=tag, row=row), allow_nan=False))
                rows[row['participant']] = row
                if i % 25 == 0 or i == len(todo):
                    print(f'  [{label}] {i}/{len(todo)}, {perf_counter()-start:.0f}s', flush=True)
        log = cache/'compute_log.json'
        entries = json.loads(log.read_text()) if log.exists() else []
        entries.append(dict(config_tag=tag, participants_computed=len(todo), workers=workers, seconds=perf_counter()-start))
        log.write_text(json.dumps(entries, indent=2))
    log = cache/'compute_log.json'
    compute = [e for e in json.loads(log.read_text()) if e['config_tag'] == tag] if log.exists() else []
    return [rows[j[0]] for j in jobs], compute


def subset(order, k):
    idx = np.unique(np.round(np.linspace(0, len(order)-1, k)).astype(int))
    return [order[i] for i in idx]


def write_csv(path, rows):
    out = []
    for r in rows:
        row = dict(participant=r['participant'], n_train=r['n_train'], n_validation=r['n_validation'])
        for v in m.VARIANTS:
            x = r['variants'][v]
            if x['status'] != 'ok':
                row[f'{v}_status'] = x['status']; continue
            row.update({f'{v}_{k}': val for k, val in x['parameters'].items()})
            row.update({f'{v}_train_nll': x['train_nll'], f'{v}_s5_loss': x['s5_loss'], f'{v}_s1_early_loss': x['s1_early_loss'],
                        f'{v}_k0_level': x['k0_descriptors']['level'], f'{v}_k0_spread': x['k0_descriptors']['spread'],
                        f'{v}_k0_pattern_corr': x['k0_descriptors']['pattern_corr'], f'{v}_reaching_best': x['reaching_best']})
            row.update({f'{v}_S{s}_loss': val for s, val in x['session_loss'].items()})
        out.append(row)
    keys = list(dict.fromkeys(k for r in out for k in r))
    with open(path, 'w', newline='') as h:
        w = csv.DictWriter(h, fieldnames=keys); w.writeheader()
        for r in out:
            w.writerow({k: '' if r.get(k) is None else (f'{r[k]:.10g}' if isinstance(r.get(k), float) else r[k]) for k in keys})


def write_synth_csv(path, rows, regime):
    out = []
    for r in rows:
        for d in r['datasets']:
            for f, x in d['fits'].items():
                out.append(dict(regime=regime, participant=r['participant'], generator=d['generator'], seed=d['seed'], fit_init=f,
                                **{f'true_{p}': t for p, t in zip(m.PARAMETERS, d['truth'])},
                                **{f'fit_{p}': t for p, t in zip(m.PARAMETERS, x['theta'])},
                                optimization_failure=x['optimization_failure'], nll_at_truth_minus_fit=x['nll_at_truth_minus_fit'],
                                delta_nll_true_alpha=x['delta_nll_true_alpha'], s5_loss=x['s5_loss'], s5_loss_truth=x['s5_loss_truth'],
                                s5_mean_abs_dp_vs_truth=x['s5_mean_abs_dp_vs_truth'], s1_early_loss=x['s1_early_loss'], train_loss=x['train_loss']))
    keys = list(out[0])
    with open(path, 'a' if regime == 'grid' else 'w', newline='') as h:
        w = csv.DictWriter(h, fieldnames=keys)
        if regime != 'grid':
            w.writeheader()
        for r in out:
            w.writerow({k: '' if r.get(k) is None else (f'{r[k]:.10g}' if isinstance(r.get(k), float) else r[k]) for k in keys})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=Path('data/raw/02_comp_mod_RP_task_data_in.RData'))
    ap.add_argument('--fold-results', type=Path, default=Path('docs/history_diagnostics_results.json'))
    ap.add_argument('--participant', nargs='+', help='Pilot subset (outputs go to the cache directory).')
    ap.add_argument('--starts', type=int, default=200)
    ap.add_argument('--seeds', type=int, default=3, help='Synthetic realizations per participant and generator.')
    ap.add_argument('--grid-participants', type=int, default=40)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--cache-dir', type=Path, default=Path('data/derived/experiment5_ck_init_mechanisms'))
    ap.add_argument('--output-dir', type=Path, default=Path('docs'))
    ap.add_argument('--prefix', default='experiment5_ck_init_mechanisms')
    args = ap.parse_args()
    t0 = perf_counter()
    folds = json.loads(args.fold_results.read_text())
    raw_hash = sha256(args.data.read_bytes()).hexdigest()
    if raw_hash != folds['provenance']['raw_sha256']:
        raise ValueError('Raw data differ from the data behind the saved fits.')
    protected = sorted(Path('src/reward_pairs').glob('authors_*.py'))
    hashes = {str(p): sha256(p.read_bytes()).hexdigest() for p in protected}
    order = list(dict.fromkeys(r['participant'] for r in folds['rows'] if r['session'] == 5))
    participants = args.participant or order
    inputs = load(args.data, folds, participants, with_pretest=True)
    pretest = {x[0]: x[6] for x in inputs}
    inputs = [x[:6] for x in inputs]
    by_p = {x[0]: x for x in inputs}

    real_tag = f'variants={m.VARIANTS};starts={args.starts}+saved;perm_seed={m.PERMUTATION_SEED}'
    real, real_log = run(m.fit_real, [x+(args.starts, True) for x in inputs], args.workers, args.cache_dir/'real', real_tag, 'real')
    worst = {k: max(r['checks'][k] for r in real) for k in TOLERANCE}
    if any(worst[k] > TOLERANCE[k] for k in TOLERANCE):
        raise RuntimeError(f'Released-K0 reproduction failed {worst}; nothing interpreted.')
    good, excluded = m.usable(real)

    truths = {r['participant']: {g: r['variants'][g]['theta'] for g in m.GENERATORS} for r in good}
    syn_tag = f'gens={m.GENERATORS};fits={m.FIT_INITS};seeds={args.seeds};starts={m.SYNTH_STARTS};seed={m.SYNTH_SEED}'
    syn_jobs = [(p, by_p[p][1], by_p[p][2], by_p[p][3], by_p[p][4], truths[p], args.seeds) for p in truths]
    synth, syn_log = run(m.synth_participant, syn_jobs, args.workers, args.cache_dir/'synthetic', syn_tag, 'synthetic')

    base = {g: {k: float(np.median([r['variants'][g]['parameters'][k] for r in good])) for k in ('alpha_rl', 'beta_rl')}
            for g in m.GENERATORS}
    grid_ps = subset([p for p in participants if p in truths], args.grid_participants)
    grid_tag = f'alpha={m.GRID_ALPHA};beta={m.GRID_BETA_CK};base={json.dumps(base, sort_keys=True)};seed={m.SYNTH_SEED+1}'
    grid_jobs = [(p, by_p[p][1], by_p[p][2], by_p[p][3], by_p[p][4], base) for p in grid_ps]
    grid, grid_log = run(m.grid_participant, grid_jobs, args.workers, args.cache_dir/'grid', grid_tag, 'grid')

    follow_tag = f'followup={m.FOLLOWUP};starts={args.starts}+saved'
    follow, follow_log = run(m.fit_followup, [x+(args.starts, True, pretest[x[0]]) for x in inputs], args.workers,
                             args.cache_dir/'followup', follow_tag, 'followup')

    if hashes != {str(p): sha256(p.read_bytes()).hexdigest() for p in protected}:
        raise RuntimeError('Validated authors implementation changed during the run.')
    result = dict(
        configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        design=dict(variants=m.DESCRIPTION, q0='released pretraining vector in every variant',
                    real=f'fit S1-S4 with {args.starts} R-seeded starts + saved released optimum (identical across variants); S5 causal replay',
                    synthetic=dict(generators=list(m.GENERATORS), fit_inits=list(m.FIT_INITS), seeds_per_participant=args.seeds,
                                   truth="each participant's own Part-A fit under the generating K0",
                                   sequences='real S1-S5 pairs and deterministic rewards; choices simulated, never human',
                                   starts=f'{m.SYNTH_STARTS} uniform random (seed {m.SYNTH_SEED}) + separate truth start (optimization-failure check)',
                                   identifiability='fixed-alpha fit at the true alpha_CK (truth + 8 random nuisance starts)'),
                    grid=dict(alpha=list(m.GRID_ALPHA), beta_ck=list(m.GRID_BETA_CK), nuisance_medians=base, participants=grid_ps),
                    final_test='never read'),
        provenance=dict(raw_sha256=raw_hash, protected_source_sha256=hashes, python=sys.version.split()[0],
                        numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform()),
        verification=dict(worst=worst, tolerance=TOLERANCE),
        compute=dict(real=real_log, synthetic=syn_log, grid=grid_log, followup=follow_log),
        followup=m.followup_summary(good, follow), signature=m.signature_comparison(good, synth),
        real=dict(participants=len(real), analysed=len(good), excluded=excluded) | m.real_summary(good),
        recovery=m.recovery_summary(synth), grid=m.grid_summary(grid), runtime_seconds=perf_counter()-t0)
    out = args.output_dir if not args.participant else args.cache_dir/'pilot_output'
    (out/'figures').mkdir(parents=True, exist_ok=True)
    (out/f'{args.prefix}.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    write_csv(out/f'{args.prefix}_participants.csv', good)
    write_synth_csv(out/f'{args.prefix}_recovery.csv', synth, 'realistic')
    write_synth_csv(out/f'{args.prefix}_recovery.csv', grid, 'grid')
    try:
        from reward_pairs import ck_init_mechanisms_report as report
        report.render(result, good, synth, grid, out, args.prefix)
    except ImportError:
        print('report module missing; JSON/CSV written')
    V = result['real']['variants']
    print('alpha_ck median:', {v: round(V[v]['parameters']['alpha_ck']['median'], 4) for v in m.VARIANTS})
    print('verification', worst, f'runtime {result["runtime_seconds"]:.0f}s')


if __name__ == '__main__':
    main()
