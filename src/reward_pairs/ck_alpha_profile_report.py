"""Figures and Markdown for Experiment 3 (alpha_CK profile likelihood)."""
from pathlib import Path

import numpy as np

from .ck_alpha_profile import GRID, NEAR_OPTIMAL_NATS, NUISANCE, PRIMARY_TOL, grid_points, near_optimal, usable
from .participant_error_report import GRID as GRIDCOLOR, INK, INK2, _plt, _save, table

BLUE, ORANGE, AQUA, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#9a9a94'
MARKER = '<!-- Reviewed interpretation -->'
X = np.arange(len(GRID))
LABELS = [f'{a:g}' for a in GRID]
NAMES = {'authors_beta_ck': 'β_CK', 'authors_alpha_rl': 'α_RL', 'authors_beta_rl': 'β_RL'}
Y_CAP = 12.


def xpos(alpha):
    return float(np.interp(alpha, GRID, X))


def _xaxis(ax, label=True):
    ax.set_xticks(X, LABELS, rotation=60, fontsize=7.5)
    if label:
        ax.set_xlabel('fixed α_CK (grid positions; not to scale)')
    ax.grid(axis='y', color=GRIDCOLOR, lw=.6)


def _band(ax, values, color, label):
    v = np.asarray(values, float)
    q25, med, q75 = np.quantile(v, [.25, .5, .75], axis=0)
    ax.fill_between(X, q25, q75, color=color, alpha=.22, lw=0)
    ax.plot(X, med, color=color, lw=2, marker='o', ms=4, label=label)


def _by(rows, fn):
    return np.array([[fn(p) for p in grid_points(r)] for r in rows])


def figure_profiles(rows, reps, path):
    plt = _plt()
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2), sharex=True)
    lookup = {r['participant']: r for r in rows}
    for ax, p in zip(axes.flat[:6], reps):
        r = lookup[p]; pts = r['points']
        x = [xpos(q['alpha_ck']) for q in pts]; y = [q['delta_nll'] for q in pts]
        ax.plot(x, np.minimum(y, Y_CAP), color=BLUE, lw=2, marker='o', ms=3.5)
        for tol, ls in zip(NEAR_OPTIMAL_NATS, (':', '--', '-.')):
            ax.axhline(tol, color=INK2, lw=.8, ls=ls)
        ax.axvline(xpos(r['saved_theta'][0]), color=ORANGE, lw=1.4)
        ax.set_ylim(-.3, Y_CAP+.3)
        ax.set_title(f'{p}: saved α_CK={r["saved_theta"][0]:.3g}\n{r["n_train"]} S1–S4 choices', fontsize=9)
        _xaxis(ax, label=False)
    _band(axes.flat[6], _by(rows, lambda q: min(q['delta_nll'], 60)), BLUE, 'median, IQR band')
    axes.flat[6].set_ylim(-.3, Y_CAP+.3); axes.flat[6].set_title(f'All {len(rows)} participants', fontsize=9)
    for tol, ls in zip(NEAR_OPTIMAL_NATS, (':', '--', '-.')):
        axes.flat[6].axhline(tol, color=INK2, lw=.8, ls=ls, label=f'{tol} nat')
    axes.flat[6].legend(fontsize=7.5, loc='upper center', frameon=False, ncol=2)
    _xaxis(axes.flat[6], label=False)
    ax = axes.flat[7]
    for tol, color in zip(NEAR_OPTIMAL_NATS, (AQUA, BLUE, ORANGE)):
        frac = (_by(rows, lambda q: q['delta_nll']) <= tol).mean(axis=0)
        ax.plot(X, frac, color=color, lw=2, marker='o', ms=3.5, label=f'ΔNLL ≤ {tol} nat')
    ax.set_ylim(0, 1); ax.set_ylabel('fraction of participants'); ax.set_title('Share of cohort near the optimum', fontsize=9)
    ax.legend(fontsize=7.5, frameon=False); _xaxis(ax, label=False)
    for ax in axes[:, 0]:
        ax.set_ylabel('Δ training NLL [nats]')
    for ax in axes[1]:
        ax.set_xlabel('fixed α_CK (grid positions)')
    fig.suptitle(f'E3-A  Profile likelihood of α_CK (nuisance parameters re-optimized at each α; orange line = saved fit; '
                 f'y capped at {Y_CAP:g})', fontweight='bold', fontsize=11)
    fig.tight_layout(); _save(fig, path)


def figure_identifiability(rows, path):
    plt = _plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw=dict(width_ratios=[2.3, 1]))
    order = sorted(rows, key=lambda r: (r['saved_theta'][0], r['participant']))
    for i, r in enumerate(order):
        for tol, color, lw in ((1.92, '#c9d9ef', 2.4), (PRIMARY_TOL, BLUE, 1.4)):
            s = near_optimal(r, tol)
            if s['count']:
                a1.plot([i, i], [xpos(s['lower']), xpos(s['upper'])], color=color, lw=lw, solid_capstyle='butt')
        a1.plot(i, xpos(r['saved_theta'][0]), 'o', color=ORANGE, ms=2.2)
    a1.set_yticks(X, LABELS, fontsize=7.5); a1.set_ylabel('α_CK (grid positions)')
    a1.set_xlabel('participants, sorted by saved α_CK'); a1.set_xlim(-2, len(order)+1)
    a1.plot([], [], color=BLUE, lw=2, label=f'grid α with ΔNLL ≤ {PRIMARY_TOL:g} nat')
    a1.plot([], [], color='#c9d9ef', lw=3, label='ΔNLL ≤ 1.92 nat')
    a1.plot([], [], 'o', color=ORANGE, ms=4, label='saved unconstrained α_CK')
    a1.legend(fontsize=8, loc='upper left', frameon=False); a1.grid(axis='y', color=GRIDCOLOR, lw=.6)
    a1.set_title('Near-optimal α_CK range per participant (descriptive tolerance, not a CI)', fontsize=10)
    counts = {tol: [near_optimal(r, tol)['count'] for r in rows] for tol in NEAR_OPTIMAL_NATS}
    bins = np.arange(len(GRID)+2)-.5
    for (tol, c), color in zip(counts.items(), (AQUA, BLUE, ORANGE)):
        a2.hist(c, bins=bins, histtype='step', lw=2, color=color, label=f'≤ {tol} nat')
    a2.set_xlabel(f'grid values inside the near-optimal set (of {len(GRID)})'); a2.set_ylabel('participants')
    a2.legend(fontsize=8, frameon=False); a2.set_title('How many grid α are training-compatible', fontsize=10)
    fig.suptitle('E3-B  Identifiability of α_CK from S1–S4 training choices', fontweight='bold', fontsize=11)
    fig.tight_layout(); _save(fig, path)


def figure_compensation(rows, reps, path):
    plt = _plt()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    lookup = {r['participant']: r for r in rows}
    for ax, name in zip(axes[0], NUISANCE):
        for p in reps:
            ax.plot(X, [q['nuisance'][name] for q in grid_points(lookup[p])], color=GRAY, lw=.9, alpha=.8)
        _band(ax, _by(rows, lambda q: q['nuisance'][name]), BLUE, 'cohort median, IQR')
        ax.set_ylabel(NAMES[name]); ax.set_title(f'Re-optimized {NAMES[name]} (grey: 6 representatives)', fontsize=9.5)
        _xaxis(ax, label=False)
    axes[0, 0].legend(fontsize=8, frameon=False)
    ax = axes[1, 0]
    _band(ax, _by(rows, lambda q: q['train_components']['median_abs_kernel']), BLUE, '|β_CK ΔK| kernel')
    _band(ax, _by(rows, lambda q: q['train_components']['median_abs_reward']), ORANGE, '|β_RL ΔQ| reward')
    ax.set_ylabel('per-participant median |logit term|'); ax.set_title('Weighted contributions, S1–S4 training trials', fontsize=9.5)
    ax.legend(fontsize=8, frameon=False); _xaxis(ax)
    ax = axes[1, 1]
    _band(ax, _by(rows, lambda q: q['s5_components']['median_abs_kernel']), BLUE, '|β_CK ΔK| kernel')
    _band(ax, _by(rows, lambda q: q['s5_components']['median_abs_reward']), ORANGE, '|β_RL ΔQ| reward')
    ax.set_title('Weighted contributions, session-5 trials', fontsize=9.5); ax.legend(fontsize=8, frameon=False); _xaxis(ax)
    ax = axes[1, 2]
    _band(ax, _by(rows, lambda q: q['s5_mean_abs_probability_change']), BLUE, 'profile (nuisance re-fit)')
    fixed = np.array([[f['s5_mean_abs_probability_change'] for f in r['fixed_nuisance'] if f['alpha_ck'] in GRID] for r in rows])
    _band(ax, fixed, ORANGE, 'secondary: nuisance held at saved fit')
    ax.set_ylabel('mean |P(left) − P_saved(left)|'); ax.set_title('Session-5 prediction change vs saved fit', fontsize=9.5)
    ax.legend(fontsize=8, frameon=False); _xaxis(ax)
    fig.suptitle('E3-C  Parameter compensation along the α_CK profile', fontweight='bold', fontsize=11)
    fig.tight_layout(); _save(fig, path)


def figure_train_vs_s5(result, rows, reps, path):
    plt = _plt()
    S = result['summary']['by_alpha']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax = axes[0]
    tr = [s['delta_train_loss_per_choice']['mean'] for s in S]
    m = np.array([s['delta_s5_loss']['mean'] for s in S]); ci = np.array([s['delta_s5_loss']['ci95'] for s in S])
    ax.plot(X, tr, color=BLUE, lw=2, marker='o', ms=4, label='training (S1–S4): Δ NLL per choice vs best')
    ax.plot(X, m, color=ORANGE, lw=2, marker='s', ms=4, label='session 5: Δ log loss vs saved fit')
    ax.fill_between(X, ci[:, 0], ci[:, 1], color=ORANGE, alpha=.2, lw=0)
    ax.axhline(0, color=INK, lw=1); ax.set_ylabel('nats per choice (participant-weighted mean)')
    ax.set_title('Cohort: training vs session-5 cost of fixing α_CK', fontsize=9.5); ax.legend(fontsize=8, frameon=False); _xaxis(ax)
    ax = axes[1]
    lookup = {r['participant']: r for r in rows}
    shades = plt.get_cmap('Blues')(np.linspace(.45, .95, len(reps)))
    for p, c in zip(reps, shades):
        pts = grid_points(lookup[p])
        ax.plot([q['delta_nll'] for q in pts], [q['delta_s5_loss'] for q in pts], color=c, lw=1.2, marker='o', ms=3, label=p)
    ax.axvline(PRIMARY_TOL, color=INK2, ls='--', lw=.8); ax.axhline(0, color=INK, lw=1)
    ax.set_xscale('symlog', linthresh=1); ax.set_xlim(left=-.05); ax.set_xlabel('Δ training NLL [nats] (symlog)')
    ax.set_ylabel('Δ S5 log loss vs saved fit [nats/choice]')
    ax.set_title('Representatives: each point is one grid α', fontsize=9.5); ax.legend(fontsize=7, frameon=False, ncol=2)
    ax = axes[2]
    ranges = np.array([near_optimal(r, PRIMARY_TOL)['s5_loss_range'] for r in rows])
    ax.hist(ranges[np.isfinite(ranges)], bins=40, color=BLUE, alpha=.85)
    for v, ls in ((.005, ':'), (.01, '--')):
        ax.axvline(v, color=INK2, ls=ls, lw=1, label=f'{v} nats/choice')
    ax.set_xlabel(f'S5 loss range across grid α with ΔNLL ≤ {PRIMARY_TOL:g} nat'); ax.set_ylabel('participants')
    ax.set_title('Spread of S5 loss among training-compatible α', fontsize=9.5); ax.legend(fontsize=8, frameon=False)
    fig.suptitle('E3-D  Training profile versus session-5 (previously inspected validation) loss', fontweight='bold', fontsize=11)
    fig.tight_layout(); _save(fig, path)


def figure_s5_sensitivity(result, rows, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(13, 5))
    d = _by(rows, lambda q: q['delta_s5_loss'])
    rng = np.random.default_rng(0)
    for j in X:
        ax.scatter(j+rng.uniform(-.2, .2, len(d)), d[:, j], s=5, color=BLUE, alpha=.3, edgecolor='none')
    ax.boxplot(d, positions=X, widths=.55, showfliers=False, medianprops=dict(color=INK, lw=1.5),
               boxprops=dict(color=INK2), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
    S = result['summary']['by_alpha']
    m = np.array([s['delta_s5_loss']['mean'] for s in S]); ci = np.array([s['delta_s5_loss']['ci95'] for s in S])
    ax.errorbar(X+.32, m, yerr=[m-ci[:, 0], ci[:, 1]-m], fmt='D', ms=4, color=ORANGE, capsize=2.5, lw=1.1, label='mean, 95% bootstrap CI')
    ax.axhline(0, color=INK, lw=1)
    lo, hi = np.quantile(d, [.01, .99])
    ax.set_ylim(min(lo, -.01)*1.1, hi*1.1)
    ax.set_ylabel('S5 loss(fixed α) − S5 loss(saved fit)\n[nats/choice; positive = worse]')
    ax.legend(fontsize=8, frameon=False, loc='upper center'); _xaxis(ax)
    ax.set_title('E3-E  Participant-level session-5 sensitivity to α_CK (profile fits; y clipped to 1st–99th percentile)',
                 fontweight='bold', fontsize=11)
    fig.tight_layout(); _save(fig, path)


def render(result, rows, out, prefix):
    out = Path(out); fig = out/'figures'
    good, _ = usable(rows)
    reps = result['representatives']
    names = dict(a='e3_a_profile_likelihood.png', b='e3_b_identifiability.png', c='e3_c_compensation.png',
                 d='e3_d_train_vs_s5.png', e='e3_e_s5_sensitivity.png')
    figure_profiles(good, reps, fig/names['a'])
    figure_identifiability(good, fig/names['b'])
    figure_compensation(good, reps, fig/names['c'])
    figure_train_vs_s5(result, good, reps, fig/names['d'])
    figure_s5_sensitivity(result, good, fig/names['e'])
    write_report(result, good, out/f'{prefix}_profile.md', names)


def _f(x, d=4):
    return 'n/a' if x is None else f'{x:.{d}f}'


def write_report(result, rows, path, figures):
    path = Path(path); S = result['summary']; O = S['optimization']; V = result['verification']
    I = S['identifiability']; B = S['by_alpha']; C = S['compensation']; L = S['training_validation']
    n = S['analysed']
    ntr = np.array([r['n_train'] for r in rows]); nva = np.array([r['n_validation'] for r in rows])
    d = result['design']
    lines = [
        '# Experiment 3: identifiability and sensitivity of the choice-kernel learning rate α_CK',
        '',
        'Authors\' RL+CK (model 19, `released_code_intended`, π = 0, bias fixed at 2.5). Fit on training sessions 1–4, '
        'scored on training session 5. Session 5 was already inspected in earlier experiments, so it is '
        '**previously inspected held-out validation, not an untouched prospective test**. The final test is never read.',
        '',
        '## 1. Question and method',
        '',
        'Choice rule: P(left) = σ(β_RL (Q_L − Q_R) + β_CK (K_L − K_R)), with K_i ← K_i + α_CK (I_i − K_i) for the two '
        'displayed stimuli and Q_i ← Q_i + α_RL (r_i − Q_i) from feedback.',
        '',
        '*Profile likelihood.* For each fixed α_CK on a predetermined grid, β_CK, α_RL and β_RL are re-optimized to '
        'minimize the unchanged authors\' training deviance; ΔNLL(α) = [D(α) − D_min]/2 nats, where D_min is the '
        'smaller of the saved unconstrained optimum and every constrained optimum. Each participant\'s saved α_CK is '
        'added as an anchor point.',
        '',
        f'Grid: {", ".join(LABELS)}. The grid is used for every cohort summary, and the anchor is used only for checks.',
        '',
        '*Secondary diagnostic (not a profile).* α_CK varied with β_CK, α_RL, β_RL held at their saved values. '
        'Comparing it with the profile shows how much the nuisance parameters compensate.',
        '',
        '## 2. Data and fitting protocol',
        '',
        f'- {S["participants"]} participants; {n} have complete successful profiles at every grid point '
        f'(excluded: {S["excluded_incomplete_profiles"] or "none"}).',
        f'- Training choices per participant (S1–S4): median {int(np.median(ntr))} (range {ntr.min()}–{ntr.max()}); '
        f'session-5 choices: median {int(np.median(nva))} (range {nva.min()}–{nva.max()}).',
        f'- Initialization: released-code K0 = Q0 = pre_val (up to ≈4.1), unchanged. Note that at α_CK = 0, '
        'K never moves from K0 = Q0, so the kernel term becomes a **fixed pre-training preference** β_CK(K0_L − K0_R), '
        'not "no kernel". At α_CK = 1, K holds only the last choice outcome for each stimulus.',
        f'- Optimizer: SciPy L-BFGS-B with the authors\' bounds {d["bounds"]} (β_CK, α_RL, β_RL) and the options '
        f'used by `authors_fitting` ({d["options"]}).',
        f'- Starts at each α: the saved nuisance values, {d["starts"]["random"]} random starts '
        f'({d["starts"]["random_seed"]}; same ranges as the authors\' driver), and one warm start from each neighbouring '
        'grid solution (an ascending sweep, then a descending sweep). The best successful start is kept.',
        '- Session 5: `history_diagnostics.replay_fold`. Q and K are carried from the end of S4, both keep updating, '
        'parameters stay fixed, and each prediction precedes its own update.',
        f'- Bootstrap: participant resampling, {result["configuration"]["bootstrap_draws"]} draws, seed {result["configuration"]["seed"]}.',
        f'- Software: Python {result["provenance"]["python"]}, NumPy {result["provenance"]["numpy"]}, SciPy {result["provenance"]["scipy"]}. '
        'Compute: ' + '; '.join(f'{e["participants_computed"]} participants in {e["seconds"]:.0f} s with {e["workers"]} workers'
                                 for e in result['cache']['compute_log']) + f' (this rendering: {result["runtime_seconds"]:.0f} s).',
        '',
        '## 3. Reproduction and optimization checks',
        '',
        table(['check', 'worst over participants', 'tolerance'], [
            ['saved S1–S4 fit deviance vs replay', f'{V["worst"]["saved_fit_deviance"]:.1e}', f'{V["tolerance"]["saved_fit_deviance"]:.0e}'],
            ['fast objective vs scalar replay', f'{V["worst"]["objective_vs_replay"]:.1e}', f'{V["tolerance"]["objective_vs_replay"]:.0e}'],
            ['saved session-5 NLL vs replay', f'{V["worst"]["saved_s5_nll"]:.1e}', f'{V["tolerance"]["saved_s5_nll"]:.0e}'],
            ['constrained fit at saved α minus saved deviance (max)', f'{V["worst"]["anchor_minus_saved_deviance_max"]:.1e}', f'≤ {V["tolerance"]["anchor"]:.0e}'],
        ]),
        '',
        f'- {O["points"]} profile points ({O["starts"]} optimizer starts): **{O["failed_points"]} failed points**, '
        f'{O["failed_starts"]} individual failed starts (all superseded by successful starts at the same α and recorded in the cache).',
        f'- Constrained optima better than the saved unconstrained optimum by more than {d["violation_tol"]:g} deviance: '
        f'**{len(O["violations"])}**' + ('' if not O['violations'] else
            ' — ' + '; '.join(f'{v["participant"]} at α={v["alpha_ck"]:g} ({v["improvement_nll"]:.3g} nats)' for v in O['violations'][:10])) + '.',
        f'- Winning start type: {O["winning_start"]}. Median share of random starts that reach the best value: '
        f'{O["random_fraction_reaching_best"]["median"]:.2f} (5th percentile {O["random_fraction_reaching_best"]["quantiles"]["p05"]:.2f}); '
        f'points where no random start reached the best value: {O["points_where_no_random_start_reached_best"]}.',
        f'- Largest projected gradient at an accepted optimum: {O["max_projected_gradient"]:.1e} (deviance units).',
        '- Pilot (9 participants spanning the saved α distribution, `experiment3_alpha_pilot.json`): 100 random starts '
        'changed no point by more than 1e-6 deviance compared with 20 starts.',
        '',
        '## 4. Identifiability',
        '',
        f'Saved unconstrained α_CK: median {S["saved_alpha"]["median"]:.3f}, IQR {S["saved_alpha"]["quantiles"]["p25"]:.3f}–'
        f'{S["saved_alpha"]["quantiles"]["p75"]:.3f}.',
        '',
        'The near-optimal set is the set of grid α with ΔNLL ≤ tolerance. The tolerances are descriptive. 1.92 nats is the '
        'value a naive χ²₁ 95% rule would give, but it is not used as a confidence interval: α_CK and α_RL lie on or near '
        'their bounds for many participants, the profile is often multimodal or flat, and the nuisance parameters trade '
        'off strongly, all of which break the asymptotic χ² approximation.',
        '',
        table(['tolerance (nats)', 'median grid points inside', 'flat (all 15)', 'includes α=0', 'includes α=1',
               'non-contiguous', 'upper/lower ≥ 3×', 'upper/lower ≥ 10×'],
              [[f'{t}', f'{I[t]["grid_points_inside"]["median"]:.0f}', I[t]['flat_all_grid'], I[t]['includes_zero'],
                I[t]['includes_one'], I[t]['non_contiguous'], I[t]['upper_over_lower_ratio']['at_least_3x'],
                I[t]['upper_over_lower_ratio']['at_least_10x']] for t in map(str, NEAR_OPTIMAL_NATS)]),
        '',
        f'Each count is out of {n} participants. Upper/lower ratios with lower = 0 count as ≥ 10×.',
        '',
        f'With the {PRIMARY_TOL:g}-nat tolerance, the lower end of the near-optimal set has median {I[str(PRIMARY_TOL)]["lower"]["median"]:g} '
        f'(IQR {I[str(PRIMARY_TOL)]["lower"]["quantiles"]["p25"]:g}–{I[str(PRIMARY_TOL)]["lower"]["quantiles"]["p75"]:g}) and the upper end has median '
        f'{I[str(PRIMARY_TOL)]["upper"]["median"]:g} (IQR {I[str(PRIMARY_TOL)]["upper"]["quantiles"]["p25"]:g}–{I[str(PRIMARY_TOL)]["upper"]["quantiles"]["p75"]:g}). '
        f'The upper/lower α ratio has median {I[str(PRIMARY_TOL)]["upper_over_lower_ratio"]["finite"]["median"]:.2f}. '
        f'In half-lives (presentations of a stimulus), the set spans a median {I[str(PRIMARY_TOL)]["half_life_at_upper_alpha"]["median"]:.1f} to '
        f'{I[str(PRIMARY_TOL)]["half_life_at_lower_alpha"]["median"]:.1f}, a median ratio of {I[str(PRIMARY_TOL)]["half_life_upper_over_lower"]["median"]:.2f}× '
        f'(95th percentile {I[str(PRIMARY_TOL)]["half_life_upper_over_lower"]["quantiles"]["p95"]:.1f}×).',
        f'Multimodality: {I["multimodal"]["participants_with_multiple_local_minima"]} participants have more than one local minimum on the grid; '
        f'the second minimum lies within 1 nat of the best for {I["multimodal"]["second_minimum_within_1_nat"]} '
        f'and within 2 nats for {I["multimodal"]["second_minimum_within_2_nats"]}.',
        '',
        f'![E3-A]({"figures/"+figures["a"]})',
        '',
        f'![E3-B]({"figures/"+figures["b"]})',
        '',
        '### Cohort profile by grid α',
        '',
        table(['α_CK', 'half-life (presentations)', 'median ΔNLL', 'IQR ΔNLL', 'within 1 nat', 'mean Δ train loss/choice',
               'mean Δ S5 loss [95% CI]', 'S5 worse / better'],
              [[f'{s["alpha_ck"]:g}', 'never updates' if s['half_life']['infinite'] else
                ('old state erased at 1st update' if s['half_life']['case'] == 'immediate_one_update' else f'{s["half_life"]["updates"]:.2g}'),
                f'{s["delta_nll"]["median"]:.2f}', f'{s["delta_nll"]["quantiles"]["p25"]:.2f}–{s["delta_nll"]["quantiles"]["p75"]:.2f}',
                s['within_1_nat'], f'{s["delta_train_loss_per_choice"]["mean"]:.4f}',
                f'{s["delta_s5_loss"]["mean"]:+.4f} [{s["delta_s5_loss"]["ci95"][0]:+.4f}, {s["delta_s5_loss"]["ci95"][1]:+.4f}]',
                f'{s["delta_s5_loss"]["hurts"]} / {s["delta_s5_loss"]["helps"]}'] for s in B]),
        '',
        '## 5. Parameter compensation',
        '',
        'Within-participant Spearman correlation between the fixed α_CK and each re-optimized quantity across the 15 grid points '
        '(counts with ρ > 0.5 or < −0.5):',
        '',
        table(['quantity', 'median ρ', 'ρ > 0.5', 'ρ < −0.5'],
              [[k.replace('authors_', ''), f'{v["spearman"]["median"]:+.2f}', v['positive'], v['negative']] for k, v in C.items()]),
        '',
        table(['α_CK', 'median β_CK', 'median α_RL', 'median β_RL', 'α_RL at a bound', 'median |kernel logit| train',
               'median |reward logit| train', 'S5 mean |ΔP| profile', 'S5 mean |ΔP| fixed nuisance', 'ΔNLL fixed nuisance (median)'],
              [[f'{s["alpha_ck"]:g}', f'{s["nuisance"]["authors_beta_ck"]["median"]:.2f}', f'{s["nuisance"]["authors_alpha_rl"]["median"]:.2f}',
                f'{s["nuisance"]["authors_beta_rl"]["median"]:.2f}', s['nuisance_at_bound']['authors_alpha_rl'],
                f'{s["components"]["train_components.median_abs_kernel"]["median"]:.2f}',
                f'{s["components"]["train_components.median_abs_reward"]["median"]:.2f}',
                f'{s["s5_mean_abs_probability_change"]["median"]:.3f}', f'{s["fixed_nuisance"]["s5_mean_abs_probability_change"]["median"]:.3f}',
                f'{s["fixed_nuisance"]["delta_nll"]["median"]:.1f}'] for s in B]),
        '',
        f'![E3-C]({"figures/"+figures["c"]})',
        '',
        '## 6. Session-5 validation sensitivity',
        '',
        f'Saved-fit session-5 loss (participant mean): {S["baseline_s5_loss"]:.5f} nats/choice. Effects below are S5 loss at '
        'the profile fit for a fixed α minus S5 loss at the saved fit, so positive means worse. No α is selected on session 5.',
        '',
        f'- Within the {PRIMARY_TOL:g}-nat training-compatible set, the S5 loss range has median '
        f'{I[str(PRIMARY_TOL)]["s5_loss_range_within_set"]["median"]:.4f}, 75th percentile {I[str(PRIMARY_TOL)]["s5_loss_range_within_set"]["quantiles"]["p75"]:.4f} '
        f'and 95th percentile {I[str(PRIMARY_TOL)]["s5_loss_range_within_set"]["quantiles"]["p95"]:.4f} nats/choice. The range exceeds 0.005 for '
        f'{I[str(PRIMARY_TOL)]["s5_range_over_0005"]} participants and 0.01 for {I[str(PRIMARY_TOL)]["s5_range_over_001"]}.',
        f'- Worst S5 change (vs the saved fit) among each participant\'s training-compatible α: mean {L["worst_s5_delta_within_1nat"]["mean"]:+.4f} '
        f'[{L["worst_s5_delta_within_1nat"]["ci95"][0]:+.4f}, {L["worst_s5_delta_within_1nat"]["ci95"][1]:+.4f}]; '
        f'more than 0.005 for {L["worst_over_0005"]} participants and more than 0.01 for {L["worst_over_001"]}.',
        f'- Within-participant Spearman ρ between training ΔNLL and S5 Δ across the grid: median {L["spearman_train_vs_s5"]["median"]:+.2f} '
        f'(IQR {L["spearman_train_vs_s5"]["quantiles"]["p25"]:+.2f} to {L["spearman_train_vs_s5"]["quantiles"]["p75"]:+.2f}).',
        '- Trial-weighted cohort S5 Δ by α: ' + ', '.join(f'{s["alpha_ck"]:g}: {s["delta_s5_loss_trial_weighted"]:+.4f}' for s in B) + '.',
        '',
        '',
        'Training cost by session: S1–S4 log loss at the profile fit minus log loss at the saved fit, per choice '
        '(participant mean [95% CI]), shown next to the session-5 change. This is descriptive, because S1–S4 are the fitted data.',
        '',
        table(['α_CK', 'S1', 'S2', 'S3', 'S4', 'S5 (held out)'],
              [[f'{s["alpha_ck"]:g}'] + [f'{s["delta_train_session_loss"][k]["mean"]:+.4f} [{s["delta_train_session_loss"][k]["ci95"][0]:+.4f}, '
                                         f'{s["delta_train_session_loss"][k]["ci95"][1]:+.4f}]' for k in ('1', '2', '3', '4')]
               + [f'{s["delta_s5_loss"]["mean"]:+.4f}'] for s in B]),
        '',
        f'![E3-D]({"figures/"+figures["d"]})',
        '',
        f'![E3-E]({"figures/"+figures["e"]})',
        '',
        MARKER, '']
    old = path.read_text() if path.exists() else ''
    if MARKER in old:
        lines.append(old.split(MARKER, 1)[1].strip())
    path.write_text('\n'.join(lines)+'\n')
