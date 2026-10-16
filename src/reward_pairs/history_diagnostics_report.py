"""Reproducible training-only diagnostic tables; reviewed conclusions appended separately."""
from pathlib import Path
from .human_history_report import table, MODEL_LABELS


def f(value):
    return 'NA' if value is None else f'{value:.6f}'


def write_report(data,path):
    path=Path(path); summary=data['summary']; cfg=data['configuration']
    missing=[pair for pair in ('2-3','4-5','6-7') if not data['support']['unordered_pairs'].get(pair,0)]
    lines=['# Training-only phase-resolved history diagnostics','',
        'This analysis refits only the three existing models. It adds no mechanism, planning, regularization, reset, or new exclusion. '
        'The previously inspected final test is not used for fitting, scoring, diagnostics, or selection. '
        'Primary validation is S1–S4 → S5; secondary folds are S1 → S2, S1–S2 → S3, S1–S3 → S4. '
        'All held-out sessions still contain reward feedback: Q and histories continue updating after each scored response, without parameter refitting.','',
        '## Protocol and data support','',
        f'{data["support"]["participants"]} participants; {data["support"]["training_rows"]:,} retained training rows. '
        f'Session counts: {data["support"]["session_counts"]}. No missing trials are reconstructed. '
        f'Primary restart count {cfg["primary_restarts"]}; rolling count {cfg["rolling_restarts"]}. '
        f'Runtime {data["runtime_seconds"]:.2f} s, including trace compression but excluding final JSON/report serialization. Fits: {summary["fit_counts"]}.','',
        'Unchanged bounded L-BFGS-B and analytic gradients are called through existing fitters. '
        'Authors starts use the established participant-index RNG convention (R seed 1234+17×index by default); cumulative starts use SeedSequence([20261015,index]). '
        'No five-session fitted parameters or final-test results are imported. Q/CK initialization uses the existing released-code-intended pre-training inputs; cumulative counts start at zero. pi=0 throughout.','',
        'Gain A over B = loss(B) − loss(A): positive always favors A. Inferential means weight participants equally. '
        'Bootstrap: 10,000 paired participant resamples, seed 20261015; percentile 95% intervals. Numerical ties: 1e-8 nats/choice. '
        'Rolling-session and phase comparisons are diagnostic, not multiplicity-adjusted confirmatory tests.','',
        f'Observed unordered training pairs: {data["support"]["unordered_pairs"]}. Reward-gap counts: {data["support"]["reward_gap_counts"]}.','',
        f'**Support limitation:** equal-value pairs absent from this run: {missing}. Observed reward gaps: {list(data["support"]["reward_gap_counts"])}. '
        'For absent pairs, empirical preference, conditional loss, calibration, and observed overconfidence are unavailable in training. Comparisons with absent reward-gap categories are also unavailable. '
        'We do not fabricate observations or recover them from final test. The state probes below are explicitly unobserved counterfactual contrasts, not additional validation data.','',
        '## Rolling-origin held-out performance','',
        table(['Validation session','RL loss','RL+CK loss','Cumulative loss','Cumulative over RL','Cumulative over CK','CK over RL'],[
            [s]+[f(v['models'][m]['participant_log_loss']['mean']) for m in ('rl','ck','cumulative')]+
            [f(v['gains'][k]['distribution']['mean']) for k in ('cumulative_over_rl','cumulative_over_ck','ck_over_rl')]
            for s,v in summary['sessions'].items() if v['complete_participants']])]
    if cfg.get('stability_check'):
        lines[0]='# Structural restart-stability subset (not cohort inference)'
        lines.insert(2,'This four-participant subset checks restart minima only. See the full cohort report for scientific conclusions.\n')
    for session,v in summary['sessions'].items():
        lines += [f'### Session {session}','',f'Complete paired participants: {v["complete_participants"]}; failed: {v["failed_participants"]}.','']
        if not v['complete_participants']: continue
        lines += [table(['Model','NLL','Trials','Mean participant loss','Median','Brier (participant mean)'],[
            [MODEL_LABELS[m],f(r['nll']),r['n'],f(r['participant_log_loss']['mean']),f(r['participant_log_loss']['median']),f(r['participant_brier']['mean'])] for m,r in v['models'].items()]),
            table(['Gain','Mean','Median','P5 / P95','95% participant-bootstrap CI','Better / worse / tied'],[
                [k,f(r['distribution']['mean']),f(r['distribution']['median']),f(r['distribution']['quantiles']['p05'])+' / '+f(r['distribution']['quantiles']['p95']),
                 str(r['bootstrap']['ci95']),f'{r["better"]} / {r["worse"]} / {r["tied"]}'] for k,r in v['gains'].items()]),
            f'Lowest-loss model counts: {v["best_model_counts"]}; numerical best-model ties: {v["best_model_ties"]}.','',
            table(['Phase','RL loss','RL+CK loss','Cumulative loss','Cumulative over RL','Cumulative over CK'],[
                [p]+[f(r['models'][m]['participant_log_loss']['mean']) for m in ('rl','ck','cumulative')]+
                [f(r['gains'][k]['distribution']['mean']) for k in ('cumulative_over_rl','cumulative_over_ck')] for p,r in v['phases'].items()]),
            f'Within-person late-minus-early change in cumulative-over-CK gain: {v["late_minus_early_cumulative_over_ck_gain"]}.','']
    lines += ['## Memory dynamics','',
        'For a seen stimulus, cumulative R changes by `(chosen_indicator − R)/(N+1)`; unseen stimuli begin at R=.5 and first update by ±.5. '
        'An old cumulative choice has weight 1/N, not a fixed-size response to a new choice. CK changes by `alpha_CK*(chosen_indicator − H)`, only when that stimulus is displayed. '
        'Its old-state multiplier after k relevant presentations is `(1-alpha_CK)^k`. These are encounter counts, not total trials, days, or a calibrated psychological memory duration.','',
        'Interior CK half-life is log(.5)/log(1−alpha). Alpha=0 is infinite/no updating (also no new choice enters); alpha=1 erases old state at the first update, reported as one update. '
        'Interior fractional half-lives are kept as such. Finite-only means explicitly exclude infinite cases. CK pre_val can initially exceed 1; initial raw-state magnitudes therefore need caution.','',
        table(['Held-out session','Mean alpha_CK','Median alpha_CK','Mean finite half-life','Median','Q25 / Q75','Infinite / alpha=1'],[
            [s,f(v['ck_timescale']['alpha'].get('mean')),f(v['ck_timescale']['alpha'].get('median')),
             f(v['ck_timescale']['finite_half_life'].get('mean')),f(v['ck_timescale']['finite_half_life'].get('median')),
             f(v['ck_timescale']['finite_half_life'].get('quantiles',{}).get('p25'))+' / '+f(v['ck_timescale']['finite_half_life'].get('quantiles',{}).get('p75')),
             f'{v["ck_timescale"]["infinite_count"]} / {v["ck_timescale"]["alpha_one_count"]}'] for s,v in summary['sessions'].items()]),
        'Fixed S1–S4-fitted parameters replayed over all five sessions: S1–S4 here are **descriptive fitted-state replay**, not validation. '
        'This separates the count-denominator effect from changing fitted parameters across folds. Means below average participant mean changes over displayed-stimulus updates; undisplayed zeros are not included. '
        'All stimulus-by-session values and strict rolling-fold update summaries are saved in JSON.','',
        table(['Session','CK raw update','Cumulative raw update','CK weighted update','Cumulative weighted update'],[
            [s,f(v['ck']['mean_abs_raw_update'].get('mean')),f(v['cumulative']['mean_abs_raw_update'].get('mean')),
             f(v['ck']['mean_abs_weighted_update'].get('mean')),f(v['cumulative']['mean_abs_weighted_update'].get('mean'))]
             for s,v in summary['fixed_primary_parameter_memory'].items()])]
    lines += ['## Unobserved equal-value state contrasts','',
        'Each actual pre-choice state supplies hypothetical frequent-minus-rare Q and H contrasts for 3−2, 4−5, 7−6. '
        'No hypothetical choice or feedback is generated, and states are not updated for these probes. '
        'Their predicted preference cannot be compared with an observed training equal-pair rate, because no such rate exists.','',
        table(['Session','Pair','CK history logit','Cumulative history logit','RL hypothetical P(frequent)','CK hypothetical P','Cumulative hypothetical P'],[
            [s,pair,f(p['ck']['unobserved_probes']['weighted_history'].get('mean')),f(p['cumulative']['unobserved_probes']['weighted_history'].get('mean'))]+
            [f(p[m]['unobserved_probes']['mean_probability'].get('mean')) for m in ('rl','ck','cumulative')]
            for s,v in summary['sessions'].items() for pair,p in v['equal_pairs'].items()])]
    if '5' in summary['sessions']:
        s=summary['sessions']['5']
        lines += ['## Reward-gap support in primary S5','',
            table(['Reward category','Model','Trials','Participant mean log loss'],[
                [category,MODEL_LABELS[name],v['n'],f(v['participant_log_loss'].get('mean'))]
                for category,records in s['reward_classes'].items() for name,v in records.items()]),
            'NA denotes no observations, not zero loss. These training data cannot establish whether a deficit is concentrated in equal-reward choices or depends on larger reward gaps.','',
            '## Calibration across rolling sessions','',
            table(['Session','RL pooled ECE','CK pooled ECE','Cumulative pooled ECE'],[
                [session]+[f(v['calibration'][name]['pooled_ECE']) for name in ('rl','ck','cumulative')]
                for session,v in summary['sessions'].items()]),
            'ECE is the predeclared-bin weighted absolute confidence/accuracy difference, not an optimized criterion. Pooled ECE can conceal participant heterogeneity; per-participant ECE distributions remain in JSON.','']
        lines += ['## Primary S5 calibration and logit decomposition','',
            'Calibration uses favored-option confidence max(Pleft,1−Pleft), with left favored at exactly .5. '
            'Bins are [.50,.55), …, [.95,1.00]. Overconfidence is mean confidence minus observed favored-option accuracy. '
            'These tables pool trials descriptively; participant Brier scores and participant ECE are also stored. No calibration results change the fitted models.','']
        for name,b in s['calibration'].items():
            lines += [f'### {MODEL_LABELS[name]}','',f'Pooled ECE: {f(b["pooled_ECE"])}; mean participant ECE: {f(b["participant_ECE"].get("mean"))}.','',
                table(['Lower','Upper','n','Mean confidence','Empirical accuracy','Overconfidence'],[
                    [f(r['lower']),f(r['upper']),r['n'],f(r['mean_confidence']),f(r['accuracy']),f(r['overconfidence'])] for r in b['bins']])]
        lines += [table(['Model','Mean absolute reward logit','Mean absolute history logit','History dominates fraction','Mean ratio (abs reward >1e-6)','Excluded ratio trials'],[
            [MODEL_LABELS[m],f(v['abs_reward'].get('mean')),f(v['abs_history'].get('mean')),f(v['history_dominates_fraction'].get('mean')),
             f(v['participant_mean_history_reward_ratio'].get('mean')),v['ratio_excluded_trials']] for m,v in s['contributions'].items()]),
            'Raw DeltaH and full per-participant distributions are saved alongside weighted contributions. The ratio can be large near zero reward logits; the >1e-6 denominator rule and excluded count are explicit. '
            'History dominance in equal-value decisions is not empirically estimable here; unobserved probes cannot fill that gap.','',
            '## Primary S5 model disagreements','',
            table(['abs(Pck−Pcum) threshold','n','Fraction','CK assigns higher P to actual choice','Cumulative higher','CK LL gain / selected trial'],[
                [k,v['n'],f(v.get('fraction')),v.get('ck_assigns_higher'),v.get('cumulative_assigns_higher'),f(v.get('ck_gain_per_disagreement'))] for k,v in s['disagreement'].items()])]
        for threshold,v in s['disagreement'].items():
            lines += [f'Threshold {threshold}: positions {v.get("phases")}; pair counts {v.get("pair_counts")}; reward gaps {v.get("reward_gap_counts")}.','']
        lines += ['Exploratory association (no individual psychological interpretation): '
            f'alpha_CK versus CK gain over cumulative: {s["ck_timescale"]["alpha_ck_vs_ck_gain"]}; '
            f'finite half-life versus CK gain: {s["ck_timescale"]["half_life_vs_ck_gain"]}. '
            'Both quantities are fitted or scored on the same participant, so these are descriptive associations, not causal tests of recency.','']
    lines += ['## Reproducibility and boundaries','',
        'Full precision participant/fold summaries, parameters, start outcomes, confidence-bin counts, pair probes, and bootstrap intervals are in the companion JSON. '
        'Every validation trial’s pre-choice Q/H, raw and weighted logits, predicted probability, loss, and post-choice history change is recorded in the local ignored NPZ archive: '
        f'`{data["trace_archive"]["path"]}`. Its SHA-256 and key schema are in JSON. '
        'The RData container is deserialized by the loader, but the pipeline never accesses the final-test object. '
        'Phase guards reject test rows even if they carry session label 5, as well as all session-6 rows. '
        'Protected model source hashes are checked before and after fitting. No original model equations or optimizer settings were edited.','',
        'These comparisons do not causally isolate recency from CK’s additional free update-rate parameter, its initialization, or representation scale. '
        'They also cannot establish that a deficit is *specific* to the training-to-test transition, because this milestone deliberately does not score test. '
        'No new history model or planning model was fitted. No commit or push was made.','',
        '<!-- Reviewed findings -->','']
    marker='<!-- Reviewed findings -->'
    old=path.read_text() if path.exists() else ''
    if marker in old: lines.append(old.split(marker,1)[1].strip())
    path.write_text('\n'.join(lines)+'\n')
