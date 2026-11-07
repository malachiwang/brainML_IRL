# Research direction (updated 2026-11-07)

## Active direction: characterize, test, and extend the authors' RL + choice-kernel model

Following guidance from Professor Anqi Wu, the **authors' validated RL+CK model** (authors model 19,
`released_code_intended` initialization, pi = 0) is the primary baseline. The objective is to
characterize, test, and extend that model, not to rescue our earlier representation.

Working rules:

- The validated reference implementation (`src/reward_pairs/authors_*.py`) stays unchanged.
  Every modification is a separate variant module, so the unchanged baseline remains available
  for comparison.
- Order of work: (1) understand α, β, the weighted reward and kernel contributions, individual
  participants, and test-time adaptation; (2) make controlled changes to the choice kernel
  (initialization, update rule, timescale); (3) only then consider changes to the learned-value
  representation.
- Isolate effects with systematic sensitivity analyses and ablations.
- Model decisions use proper held-out evaluation on training-only folds (primary:
  sessions 1–4 → session 5). The final test has been inspected many times. It may be used for
  descriptive, exploratory comparisons against the unchanged baseline, but never to select or
  tune a modification.

Active experiments:

| Experiment | Status | Report |
| --- | --- | --- |
| E2: evaluation-phase kernel ablations (carried-in state vs updating vs kernel weight) | complete | [experiment2_ck_ablation.md](experiment2_ck_ablation.md) |
| E3: α_CK profile likelihood, compensation, S5 sensitivity | complete | [experiment3_alpha_profile.md](experiment3_alpha_profile.md) |
| E4: kernel initialization K0 ∈ {released, 0.5, 0} with Q0 unchanged; refits, session losses, α profiles | complete | [experiment4_ck_initialization.md](experiment4_ck_initialization.md) |
| E5: scale-vs-preference decomposition of K0 (released K0 rescaled to [0, 1]) + parameter recovery from K0 = 0.5 fits | proposed, not started | — |

Active inputs and code: `authors_baseline_results.json` (five-session fits), the RL/RL+CK
session-5 fold fits inside `history_diagnostics_results.json`, `ck_ablation.py`,
`ck_ablation_report.py`, `scripts/run_ck_ablation.py`, `ck_alpha_profile.py`, `ck_alpha_profile_report.py`, and
`scripts/run_ck_alpha_profile.py` (per-participant cache in `data/derived/experiment3_alpha_profile/`), `ck_initialization.py`,
`ck_initialization_report.py`, and `scripts/run_ck_initialization.py` (cache in `data/derived/experiment4_ck_initialization/`).

Current findings that shape the next step: the released K0 (= Q0, up to ≈ 4.1) inflates the fitted α_CK
(median 0.129 → 0.049 with K0 = 0.5). This explains the E3 session-1 demand for a fast kernel, improves training
likelihood by 4.4 nats on average, and leaves session-5 prediction unchanged. α_RL absorbs the pretraining-preference information
removed from K0. E5 separates the scale of K0 from its stimulus-specific pattern before any baseline change.

## Archived: RL + cumulative choice history (historical comparator only)

The cumulative representation `H = C/N − 0.5` (and the synthetic "history" planning/IRL line built
on it) is **retired as a research direction**. Its implementation, results, and tests are kept
unchanged for reproducibility, and its saved comparisons remain valid evidence. No new
improvements to it will be proposed.

Reason: on the final test, RL+cumulative was worse than RL+CK for 172/213 participants. Experiment 1
traced 75% of that gap to the novel equal-reward pairs. There, the cumulative model predicts a nearly
uniform preference for the frequent shape and cannot follow individual test-phase preferences.

Archived documents (each now carries an "Archived" banner under its title):

| Document | Content |
| --- | --- |
| `history_model.md`, `history_baseline.md`, `history_replication.md` | synthetic cumulative-history Fake Human |
| `history_calibration.md`, `history_calibration_results.md` | history uncertainty / planning distinguishability |
| `history_irl.md`, `history_irl_results.md` | history-aware finite-lookahead IRL |
| `human_history_results.md`, `human_history_restart_stability.md` | human RL+cumulative vs RL / RL+CK |
| `experiment1_participant_errors.md` | participant-level cumulative vs CK errors (E1) |
| `history_diagnostics_results.md`, `history_diagnostics_stability.md` | **mixed**: cumulative parts archived; the RL and RL+CK fold fits remain active inputs |

Archived code (unchanged; still covered by the test suite): `history.py`, `history_*.py`,
`cumulative_history_human.py`, `human_history_*.py`, `participant_error_*.py`, and the matching
scripts.

Not classified here: `README.md` and `method_overview.md` have uncommitted edits, and the untracked
`transfer_stress_test*` files are uncommitted work in progress. All of these were left untouched
on purpose. Decide their status separately.

Maintenance note: `human_history_results.md` and `history_diagnostics_results.md` are regenerated by
archived report writers that rewrite the top of the file. Rerunning those pipelines will drop the
banner, so re-add it if that happens. The Experiment 1 generator emits its banner itself.
