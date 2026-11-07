"""Figures and Markdown for Experiment 1 (participant-level error diagnostics).

Matplotlib is imported lazily so the computational module and its tests do not
depend on it. Colours: validated categorical slots (CK blue, cumulative orange,
RL aqua); every series also carries a legend/label, never colour alone.
"""
from pathlib import Path

import numpy as np

from .participant_error_diagnostics import (
    DESCRIPTIVE_MARGIN, FREQUENT, HISTORY_MODELS, MODEL_LABELS, MODELS, PAIR_KEYS, RARE, TIE_TOLERANCE,
)

COLORS = {'ck': '#2a78d6', 'cumulative': '#eb6834', 'rl': '#1baf7a'}
INK, INK2, GRID, SURFACE = '#1f1f1e', '#5c5c58', '#e4e3de', '#fcfcfb'
MARKER = '<!-- Reviewed interpretation -->'


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
                         'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                         'text.color': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .7,
                         'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10,
                         'axes.titlesize': 11, 'axes.titleweight': 'bold', 'legend.frameon': False})
    return plt


def _save(fig, path):
    fig.savefig(path, dpi=180, bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)


def figure_delta(metrics, summary, path):
    plt = _plt()
    d = np.array([p['delta_vs_ck'] for p in metrics])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw=dict(width_ratios=[1.35, 1]))
    bins = np.arange(np.floor(d.min()/.01)*.01, d.max()+.01, .01)
    ax.hist(d, bins=bins, color=COLORS['cumulative'], edgecolor=SURFACE, linewidth=1.2)
    ax.axvline(0, color=INK, lw=1.4, label='0 (identical loss)')
    ax.axvline(d.mean(), color=INK2, lw=1.4, ls='--', label=f'mean {d.mean():+.4f}')
    ax.axvline(np.median(d), color=INK2, lw=1.4, ls=':', label=f'median {np.median(d):+.4f}')
    s = summary
    ax.text(.98, .70, f'cumulative worse: {s["worse"]}\ncumulative better: {s["better"]}\nties (|Δ|≤1e-8): {s["tied"]}\nN = {len(d)}',
            transform=ax.transAxes, ha='right', va='top', fontsize=9.5, color=INK)
    ax.set_xlabel('Δ = loss(RL+cumulative) − loss(RL+CK)   [nats per test choice]')
    ax.set_ylabel('Participants')
    ax.set_title('E1-A  Participant final-test log-loss differences')
    ax.legend(loc='upper right', fontsize=9)
    order = np.sort(d)[::-1]
    share = np.cumsum(order)/order.sum()
    x = np.arange(1, len(d)+1)
    bx.plot(x, share, color=COLORS['cumulative'], lw=2, label='observed (worst Δ first)')
    bx.plot([0, len(d)], [0, 1], color=INK2, lw=1, ls='--', label='equal contribution per person')
    bx.axhline(1, color=GRID, lw=1)
    bx.text(len(d)*.98, 1.04, f'exceeds 1: {s["better"]} people have Δ < 0', ha='right', fontsize=8.5, color=INK2)
    k = s['top_contributors']['top_10pct']
    bx.annotate(f'top 10% ({k["participants"]} people)\n= {100*k["share_of_total"]:.0f}% of summed Δ',
                xy=(k['participants'], share[k['participants']-1]), xytext=(60, .35), fontsize=9,
                arrowprops=dict(arrowstyle='-', color=INK2, lw=.8))
    bx.set_xlabel('Participants ranked by Δ (largest first)')
    bx.set_ylabel('Cumulative share of summed Δ')
    bx.set_title('Concentration of the average gap')
    bx.set_xlim(0, len(d)); bx.legend(loc='lower right', fontsize=9)
    fig.tight_layout(); _save(fig, path)


def figure_scatter(metrics, path):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5))
    y = np.array([p['loss']['cumulative'] for p in metrics])
    allv = np.r_[y, [p['loss'][r] for p in metrics for r in ('ck', 'rl')]]
    lim = (np.floor(allv.min()*20)/20, np.ceil(allv.max()*20)/20)
    for ax, ref in zip(axes, ('ck', 'rl')):
        x = np.array([p['loss'][ref] for p in metrics])
        ax.plot(lim, lim, color=INK, lw=1.2, label='y = x')
        ax.scatter(x, y, s=18, color=COLORS[ref], alpha=.75, edgecolor=SURFACE, linewidth=.5, label='participant')
        above, below = int((y-x > TIE_TOLERANCE).sum()), int((x-y > TIE_TOLERANCE).sum())
        ax.text(.04, .96, f'above line (cumulative worse): {above}\nbelow line (cumulative better): {below}',
                transform=ax.transAxes, va='top', fontsize=9)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
        ax.set_xlabel(f'{MODEL_LABELS[ref]} final-test log loss [nats/choice]')
        ax.set_ylabel('RL+cumulative final-test log loss [nats/choice]')
        ax.legend(loc='lower right', fontsize=9)
    axes[0].set_title('E1-B  Cumulative vs authors RL+CK')
    axes[1].set_title('Cumulative vs RL only')
    fig.tight_layout(); _save(fig, path)


def figure_pair_errors(metrics, result, path):
    plt = _plt()
    rng = np.random.default_rng(0)  # jitter only; no analytic role
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), sharey=True)
    for ax, key in zip(axes, PAIR_KEYS):
        for j, model in enumerate(HISTORY_MODELS):
            recs = [p['pairs'][key][model] for p in metrics if p['pairs'][key][model]['n']]
            e = np.array([r['error'] for r in recs]); n = np.array([r['n'] for r in recs])
            ax.boxplot(e, positions=[j], widths=.5, showfliers=False, medianprops=dict(color=INK, lw=1.6),
                       boxprops=dict(color=INK2), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
            ax.scatter(j+rng.uniform(-.2, .2, len(e)), e, s=6+1.2*n, color=COLORS[model], alpha=.5,
                       edgecolor='none', label=MODEL_LABELS[model] if key == PAIR_KEYS[0] else None)
            s = result['pairs'][key][model]
            ax.text(j, -.83, f'mean {s["error"]["mean"]:+.3f}\n{100*s["positive_fraction"]:.0f}% > 0',
                    ha='center', va='bottom', fontsize=8.5)
        ax.axhline(0, color=INK, lw=1.2)
        s = result['pairs'][key]['ck']
        ax.set_title(f'Shape {FREQUENT[key]} (frequent) vs {RARE[key]}\n{s["participants"]} people · {s["observations"]} choices · '
                     f'median {s["n_per_participant"]["median"]:.0f}/person', fontsize=9.5)
        ax.set_xticks([0, 1], ['RL+CK', 'RL+cumulative'])
        ax.set_ylim(-.85, 1.0)
    axes[0].set_ylabel('Predicted − observed frequent-shape rate\n(positive = overpredicts frequent shape)')
    fig.suptitle('E1-C  Per-participant signed errors on equal-reward pairs (point area ∝ observations)', fontweight='bold')
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.legend(loc='lower center', ncol=2, fontsize=9.5, markerscale=2)
    _save(fig, path)


def figure_observed_predicted(metrics, result, path):
    plt = _plt()
    fig, axes = plt.subplots(2, 3, figsize=(12, 8.2), sharex=True, sharey=True)
    for i, model in enumerate(HISTORY_MODELS):
        for j, key in enumerate(PAIR_KEYS):
            ax = axes[i, j]
            recs = [p['pairs'][key][model] for p in metrics if p['pairs'][key][model]['n']]
            o = np.array([r['observed_rate'] for r in recs]); pr = np.array([r['predicted_rate'] for r in recs])
            n = np.array([r['n'] for r in recs])
            ax.axvspan(0, .5, ymin=.5, ymax=1, color=GRID, alpha=.55, lw=0)
            ax.plot([0, 1], [0, 1], color=INK, lw=1.1)
            ax.axhline(.5, color=INK2, lw=.8, ls=':'); ax.axvline(.5, color=INK2, lw=.8, ls=':')
            ax.scatter(o+np.random.default_rng(j).uniform(-.008, .008, len(o)), pr, s=4+1.5*n,
                       color=COLORS[model], alpha=.6, edgecolor='none')
            s = result['pairs'][key][model]
            ax.text(.02, .98, f'shaded: chose rare more often\nbut model favours frequent: {s["rare_observed_but_frequent_predicted"]}',
                    transform=ax.transAxes, va='top', fontsize=8.3)
            ax.text(.98, .03, f'direction agrees: {s["direction_agreement"]}/{s["direction_decided"]}',
                    transform=ax.transAxes, ha='right', fontsize=8.3)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
            if i == 0: ax.set_title(f'Shape {FREQUENT[key]} vs {RARE[key]}')
            if j == 0: ax.set_ylabel(f'{MODEL_LABELS[model]}\npredicted frequent rate')
            if i == 1: ax.set_xlabel('Observed frequent-shape rate')
    fig.suptitle('E1-D  Participant observed vs predicted frequent-shape rates (identity line; point area ∝ observations)', fontweight='bold')
    fig.tight_layout(); _save(fig, path)


def figure_gap_link(metrics, link, path):
    plt = _plt()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.6))
    use = [p for p in metrics if p['pooled_error']['cumulative'] is not None]
    err = np.array([p['pooled_error']['cumulative'] for p in use]); d = np.array([p['delta_vs_ck'] for p in use])
    ax.scatter(err, d, s=16, color=COLORS['cumulative'], alpha=.7, edgecolor='none')
    ax.axhline(0, color=INK, lw=1); ax.axvline(0, color=INK, lw=1)
    ax.set_xlabel('Cumulative pooled equal-pair error\n(predicted − observed frequent rate, all three pairs)')
    ax.set_ylabel('Δ = loss(cumulative) − loss(CK) [nats/choice]')
    ax.set_title(f'E1-E  Overprediction vs log-loss gap (Spearman ρ = {link["spearman_error_vs_delta"]["rho"]:.2f})', fontsize=10.5)
    eq = np.array([p['equal_pair_nll_gap_vs_ck'] for p in use]); tot = np.array([p['nll_gap_vs_ck'] for p in use])
    lim = (min(eq.min(), tot.min())-2, max(eq.max(), tot.max())+2)
    bx.plot(lim, lim, color=INK, lw=1.1, label='all of gap from equal pairs')
    bx.axhline(0, color=INK2, lw=.8); bx.axvline(0, color=INK2, lw=.8)
    bx.scatter(eq, tot, s=16, color=COLORS['ck'], alpha=.7, edgecolor='none', label='participant')
    bx.set_xlabel('Equal-pair NLL gap, cumulative − CK [nats]'); bx.set_ylabel('Total test NLL gap, cumulative − CK [nats]')
    bx.set_title(f'Equal pairs: {100*link["equal_pair_share_of_test_choices"]:.0f}% of choices, '
                 f'{100*link["equal_pair_share_of_total_gap"]:.0f}% of summed gap', fontsize=10.5)
    bx.legend(loc='upper left', fontsize=9)
    fig.tight_layout(); _save(fig, path)


def figure_examples(replays, selected, path):
    plt = _plt()
    fig, axes = plt.subplots(len(selected), 2, figsize=(12.5, 2.7*len(selected)), squeeze=False,
                             gridspec_kw=dict(width_ratios=[1, 1.35]))
    for row, sel in zip(axes, selected):
        rep = replays[sel['participant']]
        t = np.arange(1, len(rep['left'])+1)
        gap = np.cumsum(rep['loss']['cumulative']-rep['loss']['ck'])
        row[0].plot(t, gap, color=INK, lw=1.6)
        row[0].axhline(0, color=INK2, lw=.8)
        eq = np.zeros(len(t), bool)
        for key in PAIR_KEYS:
            a, b = map(int, key.split('-'))
            eq |= (np.sort(rep['pairs'], 1) == [a, b]).all(1)
        row[0].scatter(t[eq], gap[eq], s=9, color=COLORS['cumulative'], zorder=3, label='equal-pair trial')
        row[0].set_ylabel('Running NLL gap\ncum − CK [nats]')
        row[0].set_title(f'{sel["participant"]} — {sel["role"].replace("_", " ")} (Δ = {sel["delta"]:+.3f})', fontsize=10, loc='left')

        ax = row[1]
        idx = np.nonzero(eq)[0]
        for model, ls in (('ck', '-'), ('cumulative', '-')):
            pf = []
            for i in idx:
                l, r = rep['pairs'][i]
                f = FREQUENT[f'{min(l, r)}-{max(l, r)}']
                pf.append(rep['p_left'][model][i] if l == f else 1-rep['p_left'][model][i])
            ax.plot(t[idx], pf, color=COLORS[model], lw=1.4, marker='o', ms=3, ls=ls, label=MODEL_LABELS[model])
        chose = []
        for i in idx:
            l, r = rep['pairs'][i]
            f = FREQUENT[f'{min(l, r)}-{max(l, r)}']
            chose.append((l == f) == bool(rep['left'][i]))
        chose = np.array(chose)
        ax.scatter(t[idx], np.where(chose, 1.06, -.06), marker='|', s=60, color=INK, label='observed: frequent (top) / rare (bottom)')
        ax.axhline(.5, color=INK2, lw=.8, ls=':')
        ax.set_ylim(-.12, 1.12); ax.set_ylabel('P(frequent shape)')
    handles = axes[0, 0].get_legend_handles_labels()
    model_handles = axes[0, 1].get_legend_handles_labels()
    axes[-1, 0].set_xlabel('Final-test trial (chronological)'); axes[-1, 1].set_xlabel('Final-test trial (equal-pair trials only)')
    fig.suptitle('E1-F  Selected participants: causal one-step predictions during the final test', fontweight='bold')
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.legend(handles[0]+model_handles[0], handles[1]+model_handles[1], loc='lower center', ncol=4, fontsize=9)
    _save(fig, path)


# --------------------------------------------------------------------- Markdown
def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'] +
                     ['| '+' | '.join(map(str, r))+' |' for r in rows])+'\n'


def f(v, d=4, sign=False):
    return 'NA' if v is None else (f'{v:+.{d}f}' if sign else f'{v:.{d}f}')


def write_report(path, *, primary, metrics, checks, selected, sensitivity, replay_info, figures, provenance):
    path = Path(path)
    s = primary['delta_vs_ck']; dd = s['distribution']; q = dd['quantiles']
    tw = primary['trial_weighted']
    link = primary['gap_link']
    lines = ['# Experiment 1 — Participant-level prediction-error diagnostics', '',
             '> **Archived — historical comparator.** This experiment diagnosed the retired RL+cumulative model; '
             'its next-step suggestions are superseded by [research_direction.md](research_direction.md).', '',
             'Diagnostic only: no model was refitted, no parameter/bound/update rule changed, no participant excluded, '
             'and final-test performance was not used to select or tune any model. Generated by '
             '`scripts/report_participant_errors.py` from saved results; numbers below are recomputed on every run. '
             'Sections D–E (after the marker) are hand-written interpretation and are preserved on regeneration.', '',
             '## A. Research question', '',
             'Why does our RL + cumulative choice-history model generalize worse to the final Reward Pairs test than the '
             "authors' RL + recency-weighted choice kernel (CK)? Specifically: are poor transfer and the pooled "
             'equal-reward overprediction widespread across participants, or concentrated in an identifiable subset?', '',
             '## B. Methods', '',
             f'**Data.** `{provenance["source"]}` (sha256 `{provenance["sha256"][:16]}…`), primary variant '
             f'**`{primary["variant"]}`**, scoring pi = 0. {primary["participants"]} unique participants, '
             f'{tw["ck"]["test_choices"]:,} final-test choices. `paper_intended` rows are filtered out of every primary '
             'number and figure and appear only in the appendix.', '',
             '**Models (as saved, unchanged).** RL only (authors model 3); RL+CK (authors model 19; JSON key `ck`); '
             'RL+cumulative (`H=C/N−0.5`, signed beta_H). Test predictions are causal one-step-ahead: Q frozen, history '
             'state updated after each observed test choice, parameters fixed from training.', '',
             '**Loss sign convention.** `Δ_i = loss_cumulative,i − loss_CK,i` (mean nats per test choice). Positive = '
             'cumulative worse. The saved file stores the opposite (`reference − cumulative`); every participant value '
             'was sign-converted and checked against direct subtraction of the stored losses (|diff| ≤ 1e-12) and '
             f'against the stored winner labels. Ties: |Δ| ≤ {TIE_TOLERANCE:g} (the repository’s numerical tolerance). '
             f'±{DESCRIPTIVE_MARGIN} nats/choice is counted descriptively only.', '',
             '**Equal-pair errors.** For pairs 2–3, 4–5, 6–7 the historically frequent member is 3, 4, 7. Per '
             'participant × pair × model: `observed = observed_frequent_count/n`, `predicted = predicted_frequent_sum/n`, '
             '`error = predicted − observed` (positive = overpredicts the frequent shape). The saved records already '
             'orient each trial to the frequent member whichever side it appeared on. Zero-observation records would '
             'yield undefined rates and are counted separately (there are none in this cohort).', '',
             '**Noise reference.** With ~12 observations per participant and pair, individual rates are noisy. Two '
             'calibrated-model references are reported: (i) `z = (pred_sum − obs_count)/sqrt(n·p̄(1−p̄))`, whose '
             'binomial variance upper-bounds the Poisson-binomial variance of a calibrated model (so |z| is conservative); '
             'a calibrated model would give |z| > 1.96 for roughly 5% of people, split evenly by sign; '
             '(ii) the expected |observed − 0.5| that binomial noise alone would produce if the model’s mean predicted '
             'rate were true. Neither is a formal per-person test; no multiplicity correction is implied.', '',
             '**Overprediction vs overconfidence.** *Directional* error is `predicted − observed` for the designated '
             'frequent shape. *Extremity* compares |predicted − 0.5| with |observed − 0.5| regardless of direction. '
             'A model can overpredict the frequent shape while being under-confident about the participant’s actual '
             'preference (e.g. predicting 0.65 frequent for someone who chose the rare shape 80% of the time).', '',
             '**Aggregation.** Participant-weighted (each person equal) and trial-weighted (each choice equal) '
             'summaries are labelled separately. Pooled pair rates are observation-weighted sums, reconstructing '
             'the published table. Correlations are Spearman, descriptive, not causal.', '',
             '## C. Results', '',
             '### C0. Reproduction of established benchmark statistics', '',
             f'{sum(c["passed"] for c in checks)}/{len(checks)} reproduction checks passed '
             '(participant count, better/worse/tied counts, mean/median Δ with sign flip, total NLL gap, '
             'trial- and participant-weighted log losses, pooled equal-pair n / observed / predicted / log loss for all '
             'three models; tolerance 1e-9, 1e-6 for summed nats). Failed checks: '
             f'{[c["check"] for c in checks if not c["passed"]] or "none"}.', '',
             table(['Model', 'Trial-weighted test log loss', 'Participant mean', 'Participant median'],
                   [[MODEL_LABELS[m], f(tw['ck']['trial_weighted_loss'][m]), f(primary['loss_distribution'][m]['mean']),
                     f(primary['loss_distribution'][m]['median'])] for m in MODELS]),
             '### C1. Analysis A — participant-level predictive performance', '',
             f'![E1-A]({figures["a"]})', '', f'![E1-B]({figures["b"]})', '',
             table(['Statistic (Δ = cum − CK, nats/choice)', 'Value'], [
                 ['Mean (participant-weighted)', f(dd['mean'], 5, True)],
                 ['Trial-weighted Δ (total NLL gap / choices)', f(tw['ck']['trial_weighted_delta'], 5, True)],
                 ['Median', f(dd['median'], 5, True)], ['SD', f(dd['sd'], 5)],
                 ['P5 / P25 / P75 / P95', f'{f(q["p05"],4,True)} / {f(q["p25"],4,True)} / {f(q["p75"],4,True)} / {f(q["p95"],4,True)}'],
                 ['Min / max', f'{f(dd["minimum"],4,True)} / {f(dd["maximum"],4,True)}'],
                 ['Cumulative worse / better / tied', f'{s["worse"]} / {s["better"]} / {s["tied"]}  (saved: 172 / 41 / 0)'],
                 [f'Worse by > {DESCRIPTIVE_MARGIN} / better by > {DESCRIPTIVE_MARGIN}', f'{s["worse_over_margin"]} / {s["better_over_margin"]}'],
                 ['Existing participant-bootstrap 95% CI of mean (sign-flipped, not resampled)',
                  f'[{f(-provenance["bootstrap_ci"][1],5,True)}, {f(-provenance["bootstrap_ci"][0],5,True)}]'],
                 ['10% trimmed mean', f(s['trimmed_mean_10pct'], 5, True)],
                 ['Top 5% / 10% / 20% of people: share of summed Δ',
                  ' / '.join(f'{100*s["top_contributors"][k]["share_of_total"]:.1f}%' for k in ('top_5pct', 'top_10pct', 'top_20pct'))],
                 ['Mean Δ after removing top 5% / 10% / 20%',
                  ' / '.join(f(s['top_contributors'][k]['mean_without'], 4, True) for k in ('top_5pct', 'top_10pct', 'top_20pct'))],
                 ['People accounting for half of the positive Δ mass', f'{s["participants_for_half_of_positive_sum"]} of {s["positive_participants"]} positive'],
             ]),
             f'Versus RL only, Δ_RL = cum − RL: mean {f(primary["delta_vs_rl"]["mean"],5,True)}, median '
             f'{f(primary["delta_vs_rl"]["distribution"]["median"],5,True)}, cumulative worse/better '
             f'{primary["delta_vs_rl"]["worse"]}/{primary["delta_vs_rl"]["better"]} (saved 120/93).', '',
             '### C2. Analysis B — equal-reward pairs at the individual level', '',
             f'![E1-C]({figures["c"]})', '', f'![E1-D]({figures["d"]})', '']
    rows = []
    for key in PAIR_KEYS:
        for model in MODELS:
            p = primary['pairs'][key][model]
            rows.append([f'{FREQUENT[key]} vs {RARE[key]}', MODEL_LABELS[model], f'{p["participants"]} / {p["observations"]}',
                         f'{p["n_per_participant"]["minimum"]:.0f}–{p["n_per_participant"]["maximum"]:.0f} (med {p["n_per_participant"]["median"]:.0f})',
                         f'{f(p["pooled"]["observed_rate"],3)} / {f(p["pooled"]["predicted_rate"],3)}',
                         f(p['error']['mean'], 3, True), f(p['error']['median'], 3, True),
                         f'{100*p["positive_fraction"]:.0f}% / {100*p["negative_fraction"]:.0f}%',
                         f'{f(p["abs_error"]["median"],3)} ({f(p["abs_error"]["quantiles"]["p25"],2)}–{f(p["abs_error"]["quantiles"]["p75"],2)})',
                         f'{p["z_over"]} / {p["z_under"]}'])
    lines += [table(['Pair (freq vs rare)', 'Model', 'People / obs', 'Obs per person', 'Pooled obs / pred', 'Mean err', 'Median err',
                     'err > 0 / < 0', 'Median |err| (IQR)', 'z > 1.96 / z < −1.96'], rows),
              'Calibrated-model reference for the last column: about 5 / 5 of 213 people.', '']
    rows = []
    for key in PAIR_KEYS:
        for model in MODELS:
            p = primary['pairs'][key][model]
            rows.append([f'{FREQUENT[key]} vs {RARE[key]}', MODEL_LABELS[model],
                         f'{p["observed_frequent_majority"]} / {p["observed_rare_majority"]}',
                         p['predicted_frequent_majority'], p['rare_observed_but_frequent_predicted'],
                         f'{p["direction_agreement"]}/{p["direction_decided"]} ({100*p["direction_agreement"]/p["direction_decided"]:.0f}%)',
                         f(p['mean_predicted_extremity'], 3), f(p['mean_observed_extremity'], 3), f(p['mean_calibrated_expected_extremity'], 3)])
    lines += ['**Direction versus extremity.**', '',
              table(['Pair', 'Model', 'People choosing frequent / rare more often', 'People predicted to favour frequent',
                     'Chose rare more often, model favoured frequent', 'Predicted direction matches observed',
                     'Mean |pred − .5|', 'Mean |obs − .5|', 'Mean |obs − .5| expected from noise if model true'], rows)]
    for model in HISTORY_MODELS:
        c = primary['consistency'][model]
        lines += [f'**Cross-pair consistency ({MODEL_LABELS[model]}).** People with all three pairs observed: {c["participants"]}. '
                  f'Number of pairs with positive error (0/1/2/3): {c["positive_pair_count"]}; expected if errors were '
                  f'independent across pairs with the same marginal rates: {[round(x,1) for x in c["expected_if_independent"]]}. '
                  'Pairwise Spearman correlations of signed errors: ' +
                  '; '.join(f'{k} ρ = {v["spearman"]:+.2f}' for k, v in c['spearman'].items()) + '.', '']
    g = link
    lines += ['### C3. Are overprediction and poor log loss the same people?', '', f'![E1-E]({figures["e"]})', '',
              table(['Descriptive association', 'Value'], [
                  ['Spearman ρ, cumulative pooled equal-pair error vs Δ', f'{g["spearman_error_vs_delta"]["rho"]:+.3f}'],
                  ['Spearman ρ, (cumulative − CK) pooled error vs Δ', f'{g["spearman_error_difference_vs_delta"]["rho"]:+.3f}'],
                  ['Spearman ρ, cumulative pooled error vs per-choice gap on NON-equal-pair trials', f'{g["spearman_error_vs_non_equal_pair_gap"]["rho"]:+.3f}'],
                  ['Median Δ, top-quartile overprediction vs rest', f'{f(g["top_quartile_error"]["median_delta"],4,True)} vs {f(g["top_quartile_error"]["rest_median_delta"],4,True)}'],
                  ['Overlap of top-20% Δ and top-20% overprediction', f'{g["top20pct_overlap"]["overlap"]} of {g["top20pct_overlap"]["k"]} (chance ≈ {g["top20pct_overlap"]["expected_by_chance"]:.1f})'],
                  ['Equal-pair share of test choices', f'{100*g["equal_pair_share_of_test_choices"]:.1f}%'],
                  ['Equal-pair share of summed NLL gap (cum − CK)', f'{100*g["equal_pair_share_of_total_gap"]:.1f}%  ({g["equal_pair_gap_nats"]:.1f} of {g["total_gap_nats"]:.1f} nats)'],
                  ['People with positive equal-pair gap / positive gap on other trials', f'{g["participants_with_positive_equal_pair_gap"]} / {g["participants_with_positive_other_gap"]} of {g["participants"]}'],
              ]),
              'Note: Δ includes the equal-pair trials, so the first association is partly mechanical. The non-equal-pair '
              'association is the less circular check.', '',
              '### C4. Analysis C — selected individual fits', '',
              'Deterministic rule (defined before looking at the individuals): for each target quantile q ∈ {0.95, 0.50, '
              '0.05} of Δ (numpy linear quantile), take the participant with minimal |Δ − quantile|, ties by '
              'lexicographically smallest ID, skipping anyone already chosen; then add the participant with the largest '
              'observation-weighted pooled cumulative equal-pair error (ties by ID). Parameters are descriptive context '
              'only and are not interpreted psychologically.', '']
    by_id = {p['participant']: p for p in metrics}
    rows = []
    for sel in selected:
        p = by_id[sel['participant']]
        rates = '; '.join(f'{FREQUENT[k]}v{RARE[k]}: {f(p["pairs"][k]["ck"]["observed_rate"],2)} / {f(p["pairs"][k]["ck"]["predicted_rate"],2)} / '
                          f'{f(p["pairs"][k]["cumulative"]["predicted_rate"],2)} (n={p["pairs"][k]["ck"]["n"]})' for k in PAIR_KEYS)
        cp, kp = p['parameters']['cumulative'], p['parameters']['ck']
        rows.append([sel['role'].replace('_', ' '), sel['participant'], p['n_test'], f(p['loss']['ck']), f(p['loss']['cumulative']),
                     f(p['delta_vs_ck'], 4, True), rates,
                     f'α_RL {cp["alpha_rl"]:.3f}, β_RL {cp["beta_rl"]:.2f}, β_H {cp["beta_h"]:.2f}',
                     f'α_CK {kp["alpha_ck"]:.3f}, β_CK {kp["beta_ck"]:.2f}, α_RL {kp["alpha_rl"]:.3f}, β_RL {kp["beta_rl"]:.2f}'])
    lines += [table(['Role', 'ID', 'Test n', 'CK loss', 'Cum loss', 'Δ', 'Frequent rate: observed / CK / cum', 'Cumulative params', 'CK params'], rows)]
    if replay_info.get('available'):
        lines += [f'![E1-F]({figures["f"]})', '',
                  f'Trial-level replay used the validated `replay_reference` (RL+CK) and `cumulative_history_human.evaluate` '
                  f'with saved parameters on `{provenance["source"]}`. Maximum absolute test-deviance discrepancy versus '
                  f'stored values: CK {replay_info["max_ck_error"]:.2e}, cumulative {replay_info["max_cum_error"]:.2e} '
                  '(tolerance 1e-7); replayed equal-pair predicted sums matched the saved records (tolerance 1e-9).', '',
                  table(['ID', 'Gap in 1st half of test [nats]', 'Gap in 2nd half [nats]', 'Gap on equal-pair trials [nats]', 'Total gap [nats]'],
                        [[pid, f(v['first_half'], 2, True), f(v['second_half'], 2, True), f(v['equal_pairs'], 2, True), f(v['total'], 2, True)]
                         for pid, v in replay_info['split'].items()])]
    else:
        lines += [f'Trial-level replay not generated: {replay_info.get("reason")}. No trial-level observations were fabricated.', '']
    sp = sensitivity
    if sp:
        d2 = sp['delta_vs_ck']
        lines += ['### Appendix — `paper_intended` sensitivity (kept separate; not in any primary figure)', '',
                  f'Δ = cum − CK: mean {f(d2["mean"],5,True)}, median {f(d2["distribution"]["median"],5,True)}, worse/better/tied '
                  f'{d2["worse"]}/{d2["better"]}/{d2["tied"]}, top-10% share {100*d2["top_contributors"]["top_10pct"]["share_of_total"]:.1f}%. '
                  f'Equal-pair share of summed gap: {100*sp["gap_link"]["equal_pair_share_of_total_gap"]:.1f}%.', '',
                  table(['Pair', 'Model', 'Mean err', 'err > 0', 'Direction agrees'],
                        [[f'{FREQUENT[k]} vs {RARE[k]}', MODEL_LABELS[m], f(sp['pairs'][k][m]['error']['mean'], 3, True),
                          f'{100*sp["pairs"][k][m]["positive_fraction"]:.0f}%',
                          f'{sp["pairs"][k][m]["direction_agreement"]}/{sp["pairs"][k][m]["direction_decided"]}']
                         for k in PAIR_KEYS for m in HISTORY_MODELS])]
    lines += ['### Limitations of the underlying comparison (not resolved here)', '',
              '- CK history is initialized from released `pre_val`; cumulative counts start at zero (no pretest counts).',
              '- beta_CK is bounded to [0, 10]; cumulative beta_H is signed in [−10, 10]. Parameter counts differ (4 vs 3).',
              '- Individual pair rates rest on ~12 test choices; per-person errors are noisy and not individual diagnoses.',
              '- Both history models are evaluated one-step-ahead, so CK can adapt to a person’s own earlier test choices '
              'at its fitted alpha_CK; this is legitimate causal prediction but it is tracking, not forecasting from training alone.',
              '- This final test has now been inspected repeatedly; it must not be used to select or tune a revised model.', '',
              MARKER, '']
    old = path.read_text() if path.exists() else ''
    if MARKER in old:
        lines.append(old.split(MARKER, 1)[1].strip())
    path.write_text('\n'.join(lines)+'\n')
