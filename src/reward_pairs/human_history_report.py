"""Descriptive reporting for the predeclared human history comparison."""
from pathlib import Path

MODEL_LABELS = {'rl': 'RL', 'ck': 'RL+CK', 'cumulative': 'RL+cumulative'}


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'


def write_report(data, path):
    path=Path(path)
    rows=data['rows']; participants=data['participant_order']; cfg=data['configuration']
    unique={r['participant']:r for r in rows}
    lines=['# Human RL + cumulative-history predictive comparison','',
        'This is the first fit of our cumulative representation to real human choices. It is **not forward planning**, reward-function IRL, or an individual habit diagnostic. Authors RL and RL+CK are unchanged. The primary benchmark is released-code-intended with pi=0; literal mathematical-pi scoring is not used.','',
        '## Model, data, and fixed analysis design','',
        '`P(left) = sigmoid(beta_RL*(Q_L-Q_R) + beta_H*(H_L-H_R))`. Before each retained trial, `H=C/N-.5` for seen stimuli and zero otherwise. Counts start at zero, carry across all training sessions and into test, and update only after scoring the observed response. Both displayed Qs receive the authors’ same-alpha feedback update in training. Q freezes during test while counts continue to update. There is no parameter refitting during test.','',
        'Bounds: alpha_RL [0,1], beta_RL [0,10], beta_H [-10,10]. Three parameters versus two for RL and four for RL+CK. Q initialization is exactly shared with the benchmark variant: primary uses released pre_val; sensitivity uses the paper rating rule. No pretest choices initialize cumulative counts. The existing pretest component remains in Q0 because changing it would change the reward-learning benchmark.','',
        f'Participants: **{len(participants)}**. Retained training rows: **{sum(r["structure"]["training"]["rows"] for r in unique.values()):,}**; test rows: **{sum(r["structure"]["test"]["rows"] for r in unique.values()):,}**. Already chronological: training {sum(r["structure"]["training"]["original_order_chronological"] for r in unique.values())}/{len(participants)}, test {sum(r["structure"]["test"]["original_order_chronological"] for r in unique.values())}/{len(participants)}. No missing rows are invented, no new exclusions, no synthetic replication of human participants.','',
        f'New-model starts: {cfg["restarts"]} per participant/variant; NumPy SeedSequence([{cfg["seed"]}, one-based source participant index]). The existing 200-start authors fits are replayed and checked against their saved training/test deviances and initial states on the same data, not fitted anew. Raw and reference-file hashes are stored. Wall time: **{data["runtime_seconds"]:.2f} s**.','',
        'Optimization: bounded L-BFGS-B, analytic gradients, ftol=1e-12, gtol=1e-7, maxiter=1000, maxls=40, no regularization. RL replay and its exact filtered recurrence/derivative reuse validated authors code. The multistart wrapper is separate because the authors fitter has model-specific parameter packing; that implementation was not edited. Training AIC/AICc/BIC are secondary; held-out log loss is primary.','',
        f'Paired differences are **reference loss minus cumulative loss**, so positive favors cumulative. `delta_test_LL = NLL_reference - NLL_cumulative`. Numerical ties use absolute per-choice difference <=1e-8 nats; this is not a practical-equivalence margin. Differences beyond +/-0.01 nats/choice are also counted descriptively, without claiming a validated meaningful-effect threshold. Percentile 95% bootstrap intervals use **{cfg["bootstrap_draws"]:,} participant resamples**, seed {cfg["seed"]}; entire paired participant differences are resampled, never individual trials. Cohort mean differences weight participants equally; trial-weighted totals are reported separately.','',
        '## Actual human cumulative histories, characterized before fitting','']
    h=data['history_summary']
    lines += [table(['Pair / expected direction','Expected count','Fraction','Mean final H difference','Median','P5 / P95'],[
        [key,stats['expected_order_count'],f'{stats["expected_order_fraction"]:.4f}',f'{stats["difference"]["mean"]:.6f}',
         f'{stats["difference"]["median"]:.6f}',f'{stats["difference"]["quantiles"]["p05"]:.6f} / {stats["difference"]["quantiles"]["p95"]:.6f}']
        for key,stats in h['equal_pairs'].items()]),
        'Differences are H3−H2, H4−H5, H7−H6, respectively, measured **after final training**, ready for the first test trial. They are not substituted for earlier trial histories. Expected directions are diagnostic labels, not fitting constraints.','',
        table(['Stimulus','Mean final H','Median','P5','P95'],[[i+1,f'{s["mean"]:.6f}',f'{s["median"]:.6f}',f'{s["quantiles"]["p05"]:.6f}',f'{s["quantiles"]["p95"]:.6f}'] for i,s in enumerate(h['final_h_by_stimulus'])])]
    if cfg.get('stability_check'):
        lines[0]='# Restart-stability subset (not the primary cohort analysis)'
        lines.insert(2, 'This structural four-participant subset checks optimizer restarts only. '
                     'Predictive scores below are incidental; no scientific conclusion is based on them. '
                     'See [the full cohort report](human_history_results.md) for the primary analysis.\n')
    for variant in sorted(data['summary'],key=lambda v:v!='released_code_intended'):
        s=data['summary'][variant]
        lines += [f'## {variant}','',f'Complete paired participants: {s["complete_pairs"]}; failed participant fits: {s["failed_participants"]}. Failures are retained, not silently removed.','']
        if not s['complete_pairs']: continue
        lines += [table(['Model','Total test NLL','Trial-weighted log loss','Participant mean log loss','Participant median log loss','Training total NLL'],[
            [MODEL_LABELS[name],f'{v["total_test_nll"]:.6f}',f'{v["trial_weighted_test_log_loss"]:.8f}',f'{v["participant_test_log_loss"]["mean"]:.8f}',f'{v["participant_test_log_loss"]["median"]:.8f}',f'{v["training_total_nll"]:.4f}'] for name,v in s['models'].items()]),
            table(['Comparison','Better / worse / tied','Mean gain','Median gain','P5 / P95 across people','95% bootstrap CI for mean','Gain >.01 / loss >.01'],[
            [f'cumulative vs {MODEL_LABELS[name]}',f'{v["cumulative_better"]} / {v["cumulative_worse"]} / {v["ties"]}',
             f'{v["delta_test_log_loss"]["mean"]:.8f}',f'{v["delta_test_log_loss"]["median"]:.8f}',
             f'{v["delta_test_log_loss"]["quantiles"]["p05"]:.6f} / {v["delta_test_log_loss"]["quantiles"]["p95"]:.6f}',
             f'[{v["bootstrap"]["ci95"][0]:.8f}, {v["bootstrap"]["ci95"][1]:.8f}]',f'{v["gain_over_001_count"]} / {v["loss_over_001_count"]}'] for name,v in s['comparisons'].items()])]
        for name,v in s['comparisons'].items():
            lines += [f'Against {MODEL_LABELS[name]}: total held-out log-likelihood gain {v["total_delta_test_LL"]:.6f} nats; mean participant absolute prediction-probability difference {v["mean_abs_probability_difference"]["mean"]:.6f}.','']
        b=s['beta_h']; d=b['distribution']
        lines += ['### Descriptive parameters and fitting diagnostics','',
            f'beta_H: mean {d["mean"]:.6f}, median {d["median"]:.6f}, SD {d["sd"]:.6f}, P5 {d["quantiles"]["p05"]:.6f}, P25 {d["quantiles"]["p25"]:.6f}, P75 {d["quantiles"]["p75"]:.6f}, P95 {d["quantiles"]["p95"]:.6f}, min {d["minimum"]:.6f}, max {d["maximum"]:.6f}. Signs (tolerance 1e-8): {b["positive"]} positive, {b["negative"]} negative, {b["near_zero"]} near zero. These are not individual habit-strength measurements.','',
            f'Unsuccessful starts: {s["failed_starts"]}. Boundary counts within 1e-6: {s["boundary_counts"]}. Full starts/failure diagnostics, training criteria, parameters, final Q/H, and counts are in JSON.','',
            '### Equal-value test comparisons','',
            '“Frequent” means stimulus 3, 4, or 7 for pairs 2–3, 4–5, 6–7. Observed and predicted rates below are trial-weighted over the identical retained comparisons. Predictions are one-step-ahead, using prior observed choices. They are not free-running simulated preferences.','',
            table(['Pair','Model','Trials','Observed frequent rate','Predicted frequent rate','Log loss','Mean Qfrequent−Qrare'],[
            [pair,MODEL_LABELS[name],v['n'],f'{v["observed_frequent_rate"]:.6f}' if v['n'] else 'NA',f'{v["predicted_frequent_rate"]:.6f}' if v['n'] else 'NA',f'{v["log_loss"]:.6f}' if v['n'] else 'NA',f'{v["q_difference"].get("mean",0):.6f}'] for pair,p in s['equal_pairs'].items() for name,v in p.items()])]
    lines += ['## Limits and interpretation','',
        'The paired participant bootstrap describes uncertainty in this cohort-level mean predictive contrast, assuming independent participants. It is not a confidence interval for any person’s beta_H, not a causal test of habit, and not evidence of planning. Previous synthetic single-trajectory calibration was poor; moreover that estimator used fixed utilities, whereas this milestone uses learned Q. Its exact error rates do not transfer to this new parameterization.','',
        'Holding the RL rule and Q initialization constant isolates the reward-learning component, not every property of the history families: the specified CK initializes from pre_val, cumulative history starts at zero, and cumulative beta_H is signed whereas authors beta_CK is nonnegative. Parameter counts alone do not establish fairness. The optional nonnegative-weight and shared-beta analyses are not performed; joint shared-weight nuisance optimization is deferred rather than introduced into this first participant-specific comparison.','',
        'No planning was fitted. A cumulative predictive advantage would not identify forward-looking decisions; a disadvantage would not establish that humans never use cumulative history. Final scientific conclusions are reviewed below after inspecting the complete predeclared cohort results.','',
        '<!-- Reviewed interpretation -->','']
    marker='<!-- Reviewed interpretation -->'
    old=path.read_text() if path.exists() else ''
    if marker in old: lines.append(old.split(marker,1)[1].strip())
    path.write_text('\n'.join(lines)+'\n')
