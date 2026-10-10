# History-sensitive Fake Human: multi-scaffold replication

This milestone asks whether the previous one-scaffold synthetic result survives
changes in the displayed-pair sequence. It does not fit human choices, introduce
a new behavioral model, or implement history-aware IRL. Results and full-precision
fitted parameters are in [history_replication_results.json](history_replication_results.json).

## Why replicate?

The [previous experiment](history_baseline.md) used E11T9A and five seed batches.
Its single-agent estimates were unreliable, whereas 20-agent recovery was much
better. Sequence ordering, missing retained trials, and opponent frequencies
could make that scaffold unusually easy or difficult. More seeds alone would
not test dependence on the experimental sequence. Here we vary both sequence
and synthetic random batch while leaving the statistical method unchanged.

## Predeclared design and reproduction

```sh
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_history_replication.py \
  --beta 1 --n-scaffolds 12 \
  --seed-batches 0 1 2 3 4 5 6 7 8 9 \
  --alphas 0 0.5 1 --rollouts 1 5 20 --secondary-negative \
  --output docs/history_replication_results.json
```

The primary grid has 12 scaffolds × 3 alpha values × 3 rollout counts × 10
batches = 1,080 configurations. A separately labeled, predeclared directional
control uses alpha=-1 on all 12 scaffolds with 20 rollouts and the same 10 batch
labels: another 120 configurations. Each configuration attempts four fits:
static/history on all sessions for recovery, and static/history on sessions
1–4 for held-out session-5 prediction. Total: 1,200 configurations / 4,800 fits.

Before the full grid, the complete test suite passed and a smoke test ran
E11T9A, alpha=1, batch 0, at all three rollout counts. Its 12 fits succeeded;
experiment time was 0.70 s (1.33 s including loading/serialization), projecting
roughly 4–6 minutes for the full design. Nothing was selected based on these
smoke-test estimates. No computation shortcut or estimator change was needed.

### Selection without human actions

Project the source to `ID, session, trial, stim1, stim2` before normalization.
Use an always-missing action placeholder solely for the existing trajectory API;
it is not an imputed human response. Preserve every retained row. No human
actions, accuracy, rewards, flags, test data, or fitted parameters enter selection.

Reserve E11T9A. Sort the remaining 212 participants by `(total retained rows,
lexicographic participant ID)`. Select 11 zero-based ranks using
`numpy.rint(linspace(0,211,11))`: **0, 21, 42, 63, 84, 106, 127, 148, 169,
190, 211**, with nearest-integer/ties-to-even rounding. Unite with E11T9A and
return all 12 sorted by row count and ID. Reserving the reference first makes
the total exactly 12 without duplicate or post-result replacement decisions.
The CLI's `--n-scaffolds 1` smoke-test mode selects only E11T9A.

An internal gap event is two adjacent retained trials in the same session whose
labels differ by more than one. Internal missing labels sum those differences
minus one. Leading/trailing absent labels are not counted as internal gaps.
No synthetic updates or transitions are invented for any missing trial.

### RNG and independence

For each participant ID, interpret the first eight bytes of SHA-256 of its UTF-8
ID as a little-endian integer. Derive a uint64 root with
`SeedSequence([batch, participant_key]).generate_state(1, dtype=uint64)[0]`.
The unchanged `independent_rollouts` spawns 20 child streams from that root.
Every synthetic agent starts with fresh counts and sees the same scaffold.
Counts 1, 5, and 20 use nested prefixes; they are not concatenated histories.

Scaffold/batch streams are separate, deterministic, and independent of the
order in which configurations are executed. Within a scaffold/batch, alpha
conditions share random draws, although resulting choices and histories differ.
Consequently alpha/count conditions are paired, not independent extra samples.
There are 2,400 unique rollouts per alpha condition, not 3,120. Each condition
summary has 120 scaffold × batch estimates. The different root assignment means
E11T9A's old seed-specific numerical estimates are not expected to repeat.

## What stayed unchanged

Before trial t, `H_i=C_i/N_i-0.5` if previously presented, and zero if unseen.
The simulator uses `sigmoid(beta*(U_L-U_R+alpha*(H_L-H_R)))`, fixed known beta=1,
and utilities `[1,2,2,3,3,4,4,5]`. It reads history, samples the action, and only
then updates presentation/selection counts. History carries across sessions
and gaps, using only retained synthetic trials. Neither objective rewards nor
alpha are learned by the simulated agent.

The estimator reconstructs history from earlier observed synthetic actions and
maximizes the same sequential conditional likelihood as before. It fits seven
free utility parameters plus signed alpha, fixing w1=0. It receives no true
parameters, reward columns, equality constraints, or supplied final histories.
Beta remains fixed. The optimizer, initial values, tolerance, separation/rank
checks, and observed-information diagnostic are unchanged. No priors, penalties,
sign constraints, retry scheme, or exclusions were introduced.

Static and history-aware models receive identical choices. Full-session fits
are recovery diagnostics. Separate sessions-1–4 fits score session 5 one step
ahead, carrying training history and updating it only after each observed
synthetic response. This is not free-running prediction or refitting on test data.

All summaries are descriptive, equally weighting scaffold × batch fits rather
than pooling all choices into one parameter estimate. SD is sample SD (`ddof=1`),
and quantiles use linear interpolation. Singleton SD is null. Missing/failed
fits remain in raw results and attempted denominators; individual metrics report
their successful and missing counts. Held-out win rates use paired successful
predictions, with the unavailable-pair count explicit. No confidence intervals
or population-generalization claims are attached to these purposive scaffolds.

For zero truth we predeclared `|alpha_hat| >= 0.5` and `>= 1` as descriptive
large-spurious-effect thresholds. These are **not significance tests or calibrated
false-positive rates**. Every raw estimate remains available, not just threshold
counts. Positive held-out gain means static log loss minus history log loss.

## Results

The sections below report the completed fixed grid, including all difficult
scaffolds and any fit failures; there is no selection based on recovery quality.

### Actual source and selected scaffolds

The loader used `data_4_analysis_RP_training`, not the absent expected
`data_RP_training_clean`: 164,087 retained rows across 213 participants. The
source SHA-256 is
`037dde9d968dd8d0705891a31315930d70f475fd33a99c4743712eed89e9c5e1`.
Raw/reference files remain Git-ignored. The existing R data.table conversion
warning was visible; no schema guessing or new preprocessing was introduced.

| Scaffold | Rows | Sessions 1–5 | Internal gap events | Internal missing labels |
| --- | ---: | --- | ---: | ---: |
| r01b4b | 667 | 128, 138, 127, 141, 133 | 97 | 131 |
| m03e0b | 745 | 149, 142, 148, 152, 154 | 46 | 52 |
| s02l2z | 758 | 144, 150, 150, 156, 158 | 40 | 40 |
| m09t1z | 766 | 149, 155, 155, 152, 155 | 28 | 31 |
| h01m4k | 771 | 152, 156, 160, 147, 156 | 25 | 28 |
| e05w0s | 775 | 153, 153, 157, 156, 156 | 23 | 24 |
| s05t1r | 779 | 150, 156, 160, 155, 158 | 16 | 19 |
| E11T9A | 783 | 153, 157, 156, 159, 158 | 16 | 16 |
| k08c1o | 783 | 150, 158, 160, 156, 159 | 16 | 16 |
| m08m7h | 785 | 153, 157, 158, 159, 158 | 15 | 15 |
| l06s0m | 788 | 153, 159, 159, 159, 158 | 11 | 11 |
| s04t2z | 797 | 159, 160, 159, 159, 160 | 3 | 3 |

All selected scaffolds have five sessions and 16 ordered pair states. Their
eight unordered-pair counts are:

| Scaffold | 1–2 | 1–3 | 2–4 | 3–5 | 4–6 | 5–7 | 6–8 | 7–8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| r01b4b | 34 | 128 | 126 | 35 | 32 | 123 | 144 | 45 |
| m03e0b | 42 | 133 | 143 | 45 | 45 | 145 | 142 | 50 |
| s02l2z | 47 | 141 | 139 | 43 | 50 | 140 | 150 | 48 |
| m09t1z | 45 | 146 | 147 | 47 | 45 | 141 | 146 | 49 |
| h01m4k | 48 | 145 | 149 | 46 | 45 | 144 | 146 | 48 |
| e05w0s | 47 | 145 | 142 | 49 | 49 | 144 | 150 | 49 |
| s05t1r | 49 | 142 | 143 | 49 | 48 | 149 | 149 | 50 |
| E11T9A | 47 | 147 | 148 | 49 | 50 | 143 | 149 | 50 |
| k08c1o | 50 | 145 | 145 | 47 | 48 | 148 | 150 | 50 |
| m08m7h | 48 | 143 | 149 | 48 | 49 | 149 | 149 | 50 |
| l06s0m | 49 | 148 | 148 | 49 | 49 | 147 | 149 | 49 |
| s04t2z | 50 | 149 | 149 | 49 | 50 | 150 | 150 | 50 |

### Completion, tests, and numerical status

Completed 2026-10-10. All **1,200 configurations / 4,800 fits succeeded**:
4,320 primary fits and 480 secondary fits. There were no separation, rank,
optimizer, or nonfinite-result failures. Experiment time, including aggregation,
was **312.685 s**; total CLI wall time including loading and JSON serialization
was **313.69 s**. The smoke-test fits reproduced exactly in the full run.

The suite has **99 passing tests**, including all 88 existing tests and 11 new
replication tests. New tests cover deterministic selection/range/reference,
lexicographic ties, human-action-invariant projection, gap counting, complete
unique grids, separate RNG roots and reproducible prefixes, fresh histories,
descriptive quantiles/sample SD, failed/partial-fit retention, paired win-rate
denominators, absent-scaffold summaries, measured probe direction, reproducible
end-to-end fitting, and session-5/current-action history isolation. Existing
tests that change future actions and verify unchanged earlier probabilities
and sessions-1–4 fits remain intact. No existing test was weakened.

Environment: Python 3.12.4, NumPy 2.5.3, pandas 3.0.6, SciPy 1.18.1,
rdata 1.1.0. No dependencies were added. JSON records package versions, source
and unchanged model-file SHA-256 hashes, exact command, timestamp, derived RNG
root for every configuration, every fitted vector/NLL/status, and per-agent
probe histories/probabilities. It contains 28,800 probe rows: three pairs ×
2,400 unique agents × four alpha conditions, without recounting nested subsets.
An independent post-run audit verified all 1,200 requested configuration keys,
recomputed every stored aggregate and probe summary exactly from the saved rows,
checked scaffold session/pair totals, and verified the unchanged model hashes.

### Alpha and utility recovery

All-session fits; each row summarizes 120 estimates. Bias is the mean signed
error `alpha_hat-alpha_true`. Utility MAE uses the aligned truth
`[0,1,1,2,2,3,3,4]`, including the fixed-zero reference. SD is across estimates,
not a standard error. Both static and history models estimate utilities freely.

| True alpha | Agents | Alpha mean | Alpha SD | Bias | Alpha MAE | History utility MAE | Static utility MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | −1.033864 | 1.045099 | −1.033864 | 1.121463 | 0.405809 | 0.233093 |
| 0 | 5 | −0.103410 | 0.280611 | −0.103410 | 0.228493 | 0.120039 | 0.104103 |
| 0 | 20 | −0.015989 | 0.131373 | −0.015989 | 0.103587 | 0.059125 | 0.052337 |
| 0.5 | 1 | −0.701667 | 1.065675 | −1.201667 | 1.238744 | 0.474448 | 0.296148 |
| 0.5 | 5 | 0.379085 | 0.269764 | −0.120915 | 0.225989 | 0.126556 | 0.161144 |
| 0.5 | 20 | 0.485880 | 0.124496 | −0.014120 | 0.098245 | 0.058524 | 0.140202 |
| 1 | 1 | −0.257651 | 1.006994 | −1.257651 | 1.290880 | 0.558975 | 0.442221 |
| 1 | 5 | 0.912647 | 0.220692 | −0.087353 | 0.183136 | 0.127027 | 0.296737 |
| 1 | 20 | 0.989271 | 0.102234 | −0.010729 | 0.081749 | 0.056850 | 0.280401 |

**Question 1:** For every alpha, all 12 scaffolds improved mean alpha absolute
error both from 1→5 and from 5→20 agents. This is a scaffold-level average over
10 batches, not a claim that every random batch improves monotonically.

**Questions 3–4:** History utility MAE also improved at both increments on
all 12 scaffolds for every alpha. Static utility MAE improved from 1→20 on all
scaffolds, but from 5→20 on only 11/12 at alpha=0.5 and 8/12 at alpha=1.
Its persistent utility error under positive alpha is consistent with absorbing
history bonuses into static utilities. At zero truth the static model has
lower grand-mean utility MAE at every sample size.

Single-agent recovery is worse than a mean alone might suggest: 86/120 estimates
were negative when true alpha=0.5, and 66/120 when true alpha=1. The original
five-batch result did not establish single-agent reliability. History fits also
made at least one strict-order mistake in 12/120, 22/120, and 37/120 single-agent
cases for alpha 0, 0.5, and 1 (minimum agreement 0.84). At 5 and 20 agents,
both models recovered all 25 unequal-utility orderings in every primary fit.
Exact ties were not imposed: at 20 agents, mean history-model true-tie gaps
were 0.068668, 0.070104, and 0.072572; static gaps were 0.064079, 0.142718,
and 0.336138.

### Question 2: zero-effect distribution, not just its mean

These are full-session recovery estimates at true alpha=0, 120 per row.

| Agents | Mean | Median | SD | P5 | P25 | P75 | P95 | Minimum | Maximum |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | −1.033864 | −0.824087 | 1.045099 | −3.067971 | −1.654918 | −0.290003 | 0.319313 | −4.306664 | 0.830088 |
| 5 | −0.103410 | −0.095695 | 0.280611 | −0.578484 | −0.242186 | 0.065112 | 0.336270 | −0.957069 | 0.549476 |
| 20 | −0.015989 | 0.004962 | 0.131373 | −0.252449 | −0.089961 | 0.071333 | 0.178270 | −0.452904 | 0.275399 |

| Agents | `abs(alpha_hat) >= 0.5` | `abs(alpha_hat) >= 1` |
| ---: | ---: | ---: |
| 1 | 80/120 (66.7%) | 56/120 (46.7%) |
| 5 | 12/120 (10.0%) | 0/120 |
| 20 | 0/120 | 0/120 |

At one agent, 102/120 estimates were negative. This is a broad distribution
with marked downward small-sample bias, not merely a single unlucky E11T9A
seed. Dispersion and bias shrink with independent agents. Zero events beyond
0.5 in 120 full-session fits does not prove that the event is impossible:
one separate sessions-1–4 fit at 20 agents had alpha=−0.5222. Neither this
experiment nor the threshold counts calibrate a hypothesis test.

### Questions 5–6: held-out session-5 prediction

Training uses sessions 1–4 only. Losses are nats per choice; total NLLs below
are averages of each configuration's session-5 total, whose row count varies
by scaffold. All 120 paired predictions succeeded in each row.

| Alpha | Agents | Static NLL | History NLL | Static mean loss | History mean loss | Mean gain | History wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 90.632767 | 90.833230 | 0.583980831 | 0.585287868 | −0.001307037 | 43/120 (35.8%) |
| 0 | 5 | 452.871495 | 452.873789 | 0.583551753 | 0.583554922 | −0.000003170 | 68/120 (56.7%) |
| 0 | 20 | 1806.295886 | 1806.331239 | 0.581800222 | 0.581811754 | −0.000011532 | 55/120 (45.8%) |
| 0.5 | 1 | 86.313434 | 86.438260 | 0.555961741 | 0.556799955 | −0.000838215 | 49/120 (40.8%) |
| 0.5 | 5 | 432.465960 | 432.400418 | 0.557162508 | 0.557073819 | +0.000088689 | 60/120 (50.0%) |
| 0.5 | 20 | 1726.269434 | 1725.843158 | 0.555951965 | 0.555813671 | +0.000138294 | 86/120 (71.7%) |
| 1 | 1 | 80.653508 | 80.734041 | 0.519246189 | 0.519782698 | −0.000536509 | 56/120 (46.7%) |
| 1 | 5 | 404.101651 | 403.604007 | 0.520525250 | 0.519866011 | +0.000659239 | 80/120 (66.7%) |
| 1 | 20 | 1611.520735 | 1609.100784 | 0.518924275 | 0.518144100 | +0.000780176 | 100/120 (83.3%) |

The gain distribution matters:

| Alpha | Agents | SD of gain | P5 | Median | P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0.003211982 | −0.007524786 | −0.000578583 | 0.003282938 |
| 0 | 5 | 0.000262672 | −0.000467384 | 0.000005321 | 0.000372283 |
| 0 | 20 | 0.000085043 | −0.000145340 | −0.000001152 | 0.000077796 |
| 0.5 | 1 | 0.002918389 | −0.006052303 | −0.000247476 | 0.002886147 |
| 0.5 | 5 | 0.000654363 | −0.000749025 | −0.000001568 | 0.001202839 |
| 0.5 | 20 | 0.000288282 | −0.000365848 | 0.000151670 | 0.000545267 |
| 1 | 1 | 0.004007058 | −0.006779315 | −0.000178031 | 0.004704540 |
| 1 | 5 | 0.001674738 | −0.002115569 | 0.000520953 | 0.003647942 |
| 1 | 20 | 0.000832769 | −0.000512425 | 0.000778868 | 0.002234076 |

At 20 agents, **all 12 scaffold-level mean gains are positive** for each
positive alpha. Thus the grand mean is not driven solely by E11T9A. Nevertheless,
34/120 alpha=0.5 batches and 20/120 alpha=1 batches favor static prediction.
Scaffold mean gains range from 0.0000328 to 0.0002648 at alpha=0.5 and from
0.0003923 to 0.0011597 at alpha=1. E11T9A has the largest mean gain in both
conditions, but is not the only favorable scaffold.

At zero truth, the 20-agent grand-mean difference is tiny (history loses by
0.0000115 nats/choice); only 4/12 scaffold means favor history. Random wins occur,
but there is no systematic average predictive advantage. At one agent the
richer model instead loses on average for **all three true alphas**. A better
training likelihood or occasional held-out win is not evidence of accurate
alpha recovery. No formal predictive-equivalence test was performed.

### Scaffold sensitivity and descriptive rankings

First, 20-agent mean recovered alpha across the ten batches on each scaffold:

| Scaffold | Truth 0 | Truth 0.5 | Truth 1 |
| --- | ---: | ---: | ---: |
| r01b4b | −0.038709 | 0.440930 | 0.944175 |
| m03e0b | −0.006744 | 0.479911 | 0.983056 |
| s02l2z | −0.044072 | 0.427915 | 0.972950 |
| m09t1z | −0.048094 | 0.478978 | 0.964931 |
| h01m4k | −0.054820 | 0.458005 | 0.961314 |
| e05w0s | 0.000855 | 0.495540 | 0.984222 |
| s05t1r | 0.008639 | 0.485916 | 1.017770 |
| E11T9A | 0.029027 | 0.551981 | 1.023623 |
| k08c1o | −0.029861 | 0.467594 | 0.980471 |
| m08m7h | −0.000168 | 0.507301 | 0.992609 |
| l06s0m | −0.003440 | 0.512696 | 1.023616 |
| s04t2z | −0.004478 | 0.523787 | 1.022518 |

Rank below was calculated **after** completing the grid, using mean alpha
absolute error across all 90 primary configurations per scaffold (equal weight
for each alpha/count/batch). Utility MAEs, gain, and information condition number
in this table use the same mixed primary configuration set. The separate
20-agent error column averages just its 30 primary fits. These rankings describe
the chosen finite batches, not intrinsic or statistically established difficulty.

| Rank | Scaffold | Alpha MAE, all counts | Alpha MAE, 20 only | History U MAE | Static U MAE | Held-out gain | Mean condition |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | m08m7h | 0.369986 | 0.072674 | 0.211387 | 0.238754 | 0.00011815 | 66.75 |
| 2 | r01b4b | 0.404753 | 0.087989 | 0.187183 | 0.209436 | −0.00036878 | 69.96 |
| 3 | h01m4k | 0.408662 | 0.104564 | 0.183732 | 0.183147 | 0.00009436 | 68.45 |
| 4 | m09t1z | 0.421445 | 0.102861 | 0.236050 | 0.271522 | −0.00053904 | 74.55 |
| 5 | s05t1r | 0.426603 | 0.093283 | 0.184449 | 0.200680 | −0.00005059 | 66.90 |
| 6 | s02l2z | 0.481913 | 0.119751 | 0.194186 | 0.205922 | −0.00011809 | 70.48 |
| 7 | k08c1o | 0.514548 | 0.066973 | 0.229318 | 0.250965 | −0.00011521 | 74.91 |
| 8 | E11T9A | 0.527214 | 0.092042 | 0.220564 | 0.196697 | 0.00062048 | 70.63 |
| 9 | s04t2z | 0.535565 | 0.084654 | 0.249867 | 0.243022 | −0.00025739 | 75.88 |
| 10 | m03e0b | 0.637483 | 0.129649 | 0.235262 | 0.218118 | −0.00064705 | 75.68 |
| 11 | e05w0s | 0.644992 | 0.112637 | 0.270711 | 0.250654 | −0.00013045 | 78.80 |
| 12 | l06s0m | 0.723218 | 0.067249 | 0.247093 | 0.206264 | 0.00002020 | 77.49 |

Every scaffold's mean alpha error at counts 1 / 5 / 20, by true condition:

| Scaffold | Alpha 0: 1 / 5 / 20 | Alpha 0.5: 1 / 5 / 20 | Alpha 1: 1 / 5 / 20 |
| --- | --- | --- | --- |
| r01b4b | .8407 / .1918 / .0880 | 1.0137 / .1594 / .0960 | 1.0130 / .1603 / .0800 |
| m03e0b | 1.6630 / .2443 / .1572 | 1.5346 / .2382 / .1307 | 1.4483 / .2200 / .1011 |
| s02l2z | .7761 / .3161 / .1488 | 1.0539 / .3634 / .1220 | 1.2044 / .2641 / .0885 |
| m09t1z | .8185 / .1916 / .1167 | 1.0387 / .1654 / .1141 | 1.1392 / .1310 / .0777 |
| h01m4k | .9870 / .1994 / .1124 | .9483 / .1989 / .1307 | .8967 / .1339 / .0707 |
| e05w0s | 1.4398 / .2486 / .1211 | 1.6014 / .2720 / .1039 | 1.6899 / .2153 / .1129 |
| s05t1r | 1.0183 / .1514 / .0955 | 1.0317 / .2018 / .0892 | .9905 / .1658 / .0952 |
| E11T9A | 1.0663 / .2696 / .0957 | 1.3442 / .2570 / .0909 | 1.2914 / .2403 / .0895 |
| k08c1o | 1.2022 / .2640 / .0631 | 1.2213 / .2530 / .0591 | 1.3277 / .1618 / .0787 |
| m08m7h | 1.0235 / .2094 / .0805 | .7870 / .1620 / .0693 | .8046 / .1252 / .0683 |
| l06s0m | 1.6537 / .2240 / .0727 | 1.9319 / .2444 / .0758 | 2.0647 / .1886 / .0533 |
| s04t2z | .9684 / .2318 / .0914 | 1.3583 / .1963 / .0973 | 1.6201 / .1913 / .0652 |

The initially difficult l06s0m is nearly the best at 20 agents. It has 788 rows,
only 11 internal gaps, and pair counts near the nominal 50/150 pattern. A
post-result structural check found all eight stimuli present by its sixth
retained trial and seven unordered pair types in its first 20 trials: there is
no obvious missing-stimulus explanation. Its single-agent mean condition number
is 123.70 versus 89.87 for m08m7h; corresponding minimum-eigenvalue means are
0.779 versus 1.056. e05w0s also has relatively weak single-agent information
(mean condition 124.03, smallest-eigenvalue mean 0.725). In contrast, the shortest
scaffold r01b4b, despite 97 gaps and 131 internally missing labels, is not the
hardest. Length and missingness alone do not explain the ranking.

A plausible explanation is that sequence order and stochastic early histories
change how much variation separates alpha from base utility. This is an
interpretation, not an isolated causal finding: scaffold-specific RNG batches
also contribute variation, only ten batches were used per scaffold, and the
rankings change with sample size. No difficult scaffold was removed or retuned.

### Question 7: frozen equal-value probes

History differences below are **H(left)−H(right)** after retained training,
averaged over 2,400 independent agents per alpha. Each agent's counts are frozen;
these are analytic probabilities, not extra simulated choices or human tests.
Rates can be recovered from stored centered histories by adding 0.5.

| True alpha | Pair | Mean final history difference | Mean P(left) | Mean P(right) |
| --- | --- | ---: | ---: | ---: |
| 0 | (2,3) | −0.236089 | 0.500000 | 0.500000 |
| 0 | (4,5) | 0.236351 | 0.500000 | 0.500000 |
| 0 | (6,7) | −0.234521 | 0.500000 | 0.500000 |
| 0.5 | (2,3) | −0.288786 | 0.463971 | 0.536029 |
| 0.5 | (4,5) | 0.289548 | 0.536124 | 0.463876 |
| 0.5 | (6,7) | −0.286668 | 0.464234 | 0.535766 |
| 1 | (2,3) | −0.353671 | 0.412572 | 0.587428 |
| 1 | (4,5) | 0.359142 | 0.588733 | 0.411267 |
| 1 | (6,7) | −0.350407 | 0.413363 | 0.586637 |

On **every scaffold**, all 200 agents per alpha developed higher final history
for 3 than 2, 4 than 5, and 7 than 6. Thus the measured history-direction rate
was 2,400/2,400 for each pair and condition. At positive alpha the predicted
preference followed that direction in 100% of probes. At alpha=0 the same
history asymmetry caused **no preference**: every probability was exactly 0.5.
The expected direction was only an evaluation comparator. It was not imposed
on histories, choices, or fitted parameters. Existing and new reversed-direction
tests verify that another observed history can favor the other member.

### Secondary directional control: alpha = −1

All 120 configurations / 480 fits succeeded at 20 agents. Recovered alpha was
**−1.015439 ± 0.162736** (mean ± SD), bias −0.015439, and MAE 0.126492. Every
estimate was negative; range **−1.463838 to −0.629273**. History utility MAE
was 0.058882, compared with static MAE 0.177618.

Mean held-out loss was 0.610527666 static versus 0.610339311 history, a gain
of 0.000188355 nats/choice. Mean total NLL was 1895.329824 versus 1894.742593.
History won 78/120 cases (65.0%); gain P5/median/P95 were
−0.000408135 / 0.000211195 / 0.000814710.

History still favored the frequent members, with mean H(left)−H(right) of
−0.165543, 0.168170, and −0.164898 for the three pairs. But negative alpha
reversed the resulting preference: mean P(3 over 2)=0.458724,
P(4 over 5)=0.458071, and P(7 over 6)=0.458884. Every rollout preferred the
lower-history member. Neither generation nor estimation assumes positive alpha.

### Conditioning and identifiability

Across all 2,400 history fits (all-session and training-session, including the
secondary control), information condition numbers ranged **34.61–331.74**.
The smallest observed-information eigenvalue was **0.201222**. None exceeded
the existing severe-conditioning threshold of `1e8`; all passed rank and
finite-MLE separation checks. This is local identifiability, not reliable
single-agent recovery or proof of absence of utility/history tradeoffs.

For primary all-session fits, mean condition numbers were 105.73 / 57.23 /
54.66 at 1 / 5 / 20 agents; mean smallest eigenvalues were 0.880 / 6.823 /
28.163. Independent rollouts increase information as well as stabilizing the
design. A moderate condition number alone would have missed the very large
single-agent errors and downward bias. No inference penalty was added to
conceal them.

## Interpretation, limitations, and next step

The central result replicates beyond E11T9A: recovery improves on every selected
scaffold with more independent agents, 20-agent alpha estimates are close to
truth, and positive-alpha predictive gains are positive on average on every
scaffold. Equal-value probe directions replicate completely in this finite
sample. The negative-alpha control confirms that the method can recover aversion.

The limitations also replicate. One-agent estimation is seriously unreliable,
often inferring a large negative effect at zero or positive truth. Twenty-agent
prediction gains are small, and static prediction wins in a substantial minority
of batches. Accurate prediction need not identify underlying base utility;
static estimates retain positive-alpha utility distortion. The old one-scaffold,
five-batch result understated the variability visible in this larger experiment.

These are matched-model simulations with fixed known beta and shared parameters
across artificial agents. Pooling 20 identical-parameter Fake Humans does **not**
establish that an individual human's alpha is estimable or that heterogeneous
human parameters can be pooled. The 12 length-spanning scaffolds are not a random
sample of humans. Ten batches per scaffold do not establish stable scaffold
rankings, and paired alpha/count conditions cannot be treated as independent
replications in a naive significance test. Retained-trial histories omit unknown
missing trials; the experiment never reconstructs the complete original history.
Frozen probes are generated by the toy mechanism, not psychological validation.
No human choice, human habit coefficient, test-phase behavior, new IRL model,
behavioral feature, or preprocessing exclusion was used.

**Recommended next step:** predeclare a synthetic **single-agent uncertainty
calibration** study on these same scaffolds, using the unchanged likelihood:
assess profile-likelihood interval coverage for alpha at 0/0.5/1 and held-out
detection power while refitting the nuisance utilities. Keep the zero control
and separate recovery from prediction. Establish when an individual trajectory
supports a history claim before human model comparison or formal history-aware
IRL. That new uncertainty analysis is recommended, not implemented here.
