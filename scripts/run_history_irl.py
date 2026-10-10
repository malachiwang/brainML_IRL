"""Synthetic matched finite-lookahead IRL, toy verification and mismatch diagnostics."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter
import sys

import numpy as np
import pandas as pd

from reward_pairs.data import load_rdata, select_training_object
from reward_pairs.history_mdp import HistoryMDP
from reward_pairs.history_planning import planning_rollouts, soft_plan, check_planning
from reward_pairs.history_planning_evaluation import TOY_PAIRS, benchmark, fit_record, sensitivity, summarize
from reward_pairs.history_replication import structural_training, scaffold_trajectory, scaffold_summary
from reward_pairs.history_simulation import independent_rollouts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--data', type=Path, default=root/'data/raw/02_comp_mod_RP_task_data_in.RData')
    parser.add_argument('--participant', default='E11T9A')
    parser.add_argument('--beta', type=float, default=1.)
    parser.add_argument('--gamma', type=float, default=.95)
    parser.add_argument('--lookahead', type=int, default=3)
    parser.add_argument('--lookaheads', type=int, nargs='+', default=[0, 1, 3, 5])
    parser.add_argument('--alphas', type=float, nargs='+', default=[0., .5, 1.])
    parser.add_argument('--rollouts', type=int, nargs='+', default=[5, 20])
    parser.add_argument('--seed-batches', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--toy-rollouts', type=int, default=2000)
    parser.add_argument('--output', type=Path, default=root/'docs/history_irl_results.json')
    args = parser.parse_args()
    for h in [args.lookahead, *args.lookaheads]:
        check_planning(args.beta, args.gamma, h)
    if min(args.rollouts) < 1 or args.toy_rollouts < 1 or min(args.seed_batches) < 0 or not np.isfinite(args.alphas).all():
        parser.error('Require positive rollout counts, nonnegative seeds, and finite alpha values.')
    for values in (args.rollouts, args.seed_batches, args.alphas, args.lookaheads):
        if len(values) != len(set(values)):
            parser.error('Duplicate experiment settings are not allowed.')
    started = perf_counter()
    payload = dict(command=[sys.executable, *sys.argv], beta=args.beta, gamma=args.gamma,
                   lookahead=args.lookahead, lookaheads=args.lookaheads, alphas=args.alphas,
                   seed_batches=args.seed_batches, rollout_counts=args.rollouts,
                   rng='SeedSequence(batch).spawn(max_rollouts); independent fresh histories, nested counts; common draws across model/alpha conditions',
                   history_carries_across_sessions=True, planning_crosses_session_boundaries=True,
                   truncation_value=0, reference_stimulus=1, truth_aligned=[0,1,1,2,2,3,3,4],
                   primary_horizon_selection='h=3 fixed after runtime smoke (5-rollout fit 0.68s, 20-rollout fit 2.51s), not recovery quality',
                   environment={p: version(p) for p in ('numpy','pandas','scipy','rdata')})
    # Phase A runs before loading/using the real scaffold.
    tiny = HistoryMDP([(2,3),(2,1)])
    hand = soft_plan(tiny, tiny.initial, np.zeros(8), alpha=2, beta=1, gamma=.95, lookahead=1)
    payload['hand_check'] = dict(pairs=tiny.pairs.tolist(), alpha=2, beta=1, gamma=.95,
                                 utilities=[0]*8, q=hand.q.tolist(), probability_left=float(np.exp(hand.log_policy[0])))
    toy = HistoryMDP(TOY_PAIRS)
    toy_runs = []
    for alpha in args.alphas:
        for seed in args.seed_batches[:3]:
            agents = planning_rollouts(toy, args.toy_rollouts, alpha=alpha, beta=args.beta, gamma=args.gamma, lookahead=7, seed=seed)
            data = pd.concat([a.choices for a in agents], ignore_index=True)
            record = fit_record(toy, data, model='planning', alpha_true=alpha, beta=args.beta, gamma=args.gamma, lookahead=7)
            toy_runs.append(dict(alpha_true=alpha, seed_batch=seed, rollouts=args.toy_rollouts, **record))
            print('Exact 8-trial recovery:', alpha, seed, record, flush=True)
    payload['toy'] = dict(pairs=TOY_PAIRS, lookahead=7, runs=toy_runs)
    name, source = select_training_object(load_rdata(args.data))
    trajectory = scaffold_trajectory(structural_training(source), args.participant)
    mdp = HistoryMDP.from_trials(trajectory)
    payload.update(training_object=name, scaffold=scaffold_summary(trajectory), source_sha256=sha256(args.data.read_bytes()).hexdigest())
    print('Real displayed-pair scaffold:', name, payload['scaffold'], flush=True)
    reference = independent_rollouts(trajectory, 1, seed=0, alpha=1, beta=args.beta)[0]
    payload['tractability'] = benchmark(mdp, reference.choices, beta=args.beta, gamma=args.gamma,
                                        horizons=sorted(set([*args.lookaheads, 8])))
    print('Tractability:', payload['tractability'], flush=True)
    sensitivities, probes = [], []
    for alpha in args.alphas:
        agent = independent_rollouts(trajectory, 1, seed=0, alpha=alpha, beta=args.beta)[0]
        s, p = sensitivity(mdp, agent, alpha=alpha, beta=args.beta, gamma=args.gamma, horizons=args.lookaheads)
        sensitivities.extend(s); probes.extend(p)
    payload.update(sensitivity_history_source='myopic Bob, seed batch 0, first child stream; fixed histories across horizons',
                   sensitivity=sensitivities, probes=probes)
    runs = []
    # Primary + planning-generated/myopic-fit mismatch use identical observations.
    # Reverse mismatch is predeclared: alpha=1, max rollouts, all requested batches.
    for generator, alphas, sizes in [('planning', args.alphas, args.rollouts), ('myopic', [1.], [max(args.rollouts)])]:
        for alpha in alphas:
            for seed in args.seed_batches:
                agents = (planning_rollouts(mdp, max(sizes), alpha=alpha, beta=args.beta, gamma=args.gamma,
                                             lookahead=args.lookahead, seed=seed) if generator=='planning' else
                          independent_rollouts(trajectory, max(sizes), alpha=alpha, beta=args.beta, seed=seed))
                for n in sorted(sizes):
                    choices = pd.concat([a.choices for a in agents[:n]], ignore_index=True)
                    fits = {}
                    for train in (False, True):
                        for model in ('planning', 'myopic'):
                            key = ('train_' if train else 'all_') + model
                            fits[key] = fit_record(mdp, choices, model=model, alpha_true=alpha, beta=args.beta,
                                                   gamma=args.gamma, lookahead=args.lookahead, train=train)
                    run = dict(generator=generator, alpha_true=alpha, rollouts=n, seed_batch=seed, fits=fits)
                    runs.append(run)
                    print('Real-scaffold fit:', generator, alpha, n, seed,
                          {k: (v['status'], v.get('alpha'), v.get('mae'), v.get('test_log_loss')) for k,v in fits.items()}, flush=True)
    payload.update(runs=runs, summaries=summarize(runs), completed_utc=datetime.now(timezone.utc).isoformat(),
                   total_seconds_before_serialization=perf_counter()-started)
    primary_seconds = sum(v['seconds'] for r in runs if r['generator']=='planning'
                          for k,v in r['fits'].items() if k.endswith('_planning'))
    base = next(r['batch_policy_gradient_seconds'] for r in payload['tractability']['horizons'] if r['lookahead']==args.lookahead) if args.lookahead in [r['lookahead'] for r in payload['tractability']['horizons']] else None
    for row in payload['tractability']['horizons']:
        row['projected_primary_recovery_seconds'] = primary_seconds*row['batch_policy_gradient_seconds']/base if base else None
    files = ['history.py','history_simulation.py','history_recovery.py','history_mdp.py','history_planning.py','history_irl.py','history_planning_evaluation.py']
    payload['code_sha256'] = {f:sha256((root/'src/reward_pairs'/f).read_bytes()).hexdigest() for f in files}
    failures = sum(v['status']!='ok' for r in runs for v in r['fits'].values()) + sum(r['status']!='ok' for r in toy_runs)
    payload.update(fits_attempted=len(runs)*4+len(toy_runs), failed_fits=failures)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    for row in payload['summaries']:
        print(row)
    print('Fits:',payload['fits_attempted'],'failures:',failures,'seconds:',perf_counter()-started, 'saved:',args.output)
    print('Synthetic finite-lookahead policy-likelihood IRL; not full-horizon IRL or human reward inference.')
    if failures:
        parser.exit(1, 'Failed fits retained; no regularization or silent fallback.\n')


if __name__ == '__main__':
    main()
