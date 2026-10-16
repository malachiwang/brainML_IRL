# Restart-stability subset (not the primary cohort analysis)

This structural four-participant subset checks optimizer restarts only. Predictive scores below are incidental; no scientific conclusion is based on them. See [the full cohort report](human_history_results.md) for the primary analysis.

This is the first fit of our cumulative representation to real human choices. It is **not forward planning**, reward-function IRL, or an individual habit diagnostic. Authors RL and RL+CK are unchanged. The primary benchmark is released-code-intended with pi=0; literal mathematical-pi scoring is not used.

## Model, data, and fixed analysis design

`P(left) = sigmoid(beta_RL*(Q_L-Q_R) + beta_H*(H_L-H_R))`. Before each retained trial, `H=C/N-.5` for seen stimuli and zero otherwise. Counts start at zero, carry across all training sessions and into test, and update only after scoring the observed response. Both displayed Qs receive the authors’ same-alpha feedback update in training. Q freezes during test while counts continue to update. There is no parameter refitting during test.

Bounds: alpha_RL [0,1], beta_RL [0,10], beta_H [-10,10]. Three parameters versus two for RL and four for RL+CK. Q initialization is exactly shared with the benchmark variant: primary uses released pre_val; sensitivity uses the paper rating rule. No pretest choices initialize cumulative counts. The existing pretest component remains in Q0 because changing it would change the reward-learning benchmark.

Participants: **4**. Retained training rows: **3,022**; test rows: **527**. Already chronological: training 4/4, test 4/4. No missing rows are invented, no new exclusions, no synthetic replication of human participants.

New-model starts: 200 per participant/variant; NumPy SeedSequence([20261015, one-based source participant index]). The existing 200-start authors fits are replayed and checked against their saved training/test deviances and initial states on the same data, not fitted anew. Raw and reference-file hashes are stored. Wall time: **3.80 s**.

Optimization: bounded L-BFGS-B, analytic gradients, ftol=1e-12, gtol=1e-7, maxiter=1000, maxls=40, no regularization. RL replay and its exact filtered recurrence/derivative reuse validated authors code. The multistart wrapper is separate because the authors fitter has model-specific parameter packing; that implementation was not edited. Training AIC/AICc/BIC are secondary; held-out log loss is primary.

Paired differences are **reference loss minus cumulative loss**, so positive favors cumulative. `delta_test_LL = NLL_reference - NLL_cumulative`. Numerical ties use absolute per-choice difference <=1e-8 nats; this is not a practical-equivalence margin. Differences beyond +/-0.01 nats/choice are also counted descriptively, without claiming a validated meaningful-effect threshold. Percentile 95% bootstrap intervals use **10,000 participant resamples**, seed 20261015; entire paired participant differences are resampled, never individual trials. Cohort mean differences weight participants equally; trial-weighted totals are reported separately.

## Actual human cumulative histories, characterized before fitting

| Pair / expected direction | Expected count | Fraction | Mean final H difference | Median | P5 / P95 |
| --- | --- | --- | --- | --- | --- |
| 2-3 | 4 | 1.0000 | 0.427409 | 0.430097 | 0.283541 / 0.567514 |
| 4-5 | 4 | 1.0000 | 0.478110 | 0.533680 | 0.266724 / 0.611698 |
| 6-7 | 4 | 1.0000 | 0.521527 | 0.485604 | 0.451607 / 0.641738 |

Differences are H3−H2, H4−H5, H7−H6, respectively, measured **after final training**, ready for the first test trial. They are not substituted for earlier trial histories. Expected directions are diagnostic labels, not fitting constraints.

| Stimulus | Mean final H | Median | P5 | P95 |
| --- | --- | --- | --- | --- |
| 1 | -0.274523 | -0.261863 | -0.405165 | -0.161606 |
| 2 | -0.196854 | -0.252348 | -0.308156 | -0.007861 |
| 3 | 0.230555 | 0.252083 | 0.091642 | 0.339329 |
| 4 | 0.232458 | 0.304493 | 0.021128 | 0.342940 |
| 5 | -0.245652 | -0.273179 | -0.312122 | -0.140642 |
| 6 | -0.354623 | -0.371859 | -0.410974 | -0.274141 |
| 7 | 0.166904 | 0.160881 | 0.099741 | 0.242500 |
| 8 | 0.427018 | 0.427312 | 0.389752 | 0.463872 |

## released_code_intended

Complete paired participants: 4; failed participant fits: []. Failures are retained, not silently removed.

| Model | Total test NLL | Trial-weighted log loss | Participant mean log loss | Participant median log loss | Training total NLL |
| --- | --- | --- | --- | --- | --- |
| RL | 267.470479 | 0.50753412 | 0.50685790 | 0.48005957 | 1464.0452 |
| RL+CK | 232.531134 | 0.44123555 | 0.43904793 | 0.43040317 | 1289.2651 |
| RL+cumulative | 255.818868 | 0.48542480 | 0.48461131 | 0.47297754 | 1289.3784 |

| Comparison | Better / worse / tied | Mean gain | Median gain | P5 / P95 across people | 95% bootstrap CI for mean | Gain >.01 / loss >.01 |
| --- | --- | --- | --- | --- | --- | --- |
| cumulative vs RL | 2 / 2 / 0 | 0.02224659 | 0.02000598 | -0.071300 / 0.118930 | [-0.04917138, 0.09590519] | 2 / 1 |
| cumulative vs RL+CK | 1 / 3 / 0 | -0.04556338 | -0.04842414 | -0.119443 / 0.032321 | [-0.10318745, 0.01206070] | 1 / 3 |

Against RL: total held-out log-likelihood gain 11.651611 nats; mean participant absolute prediction-probability difference 0.132169.

Against RL+CK: total held-out log-likelihood gain -23.287734 nats; mean participant absolute prediction-probability difference 0.083605.

### Descriptive parameters and fitting diagnostics

beta_H: mean 2.179520, median 2.186641, SD 0.219459, P5 1.975683, P25 2.000351, P75 2.365810, P95 2.373388, min 1.969515, max 2.375283. Signs (tolerance 1e-8): 4 positive, 0 negative, 0 near zero. These are not individual habit-strength measurements.

Unsuccessful starts: 0. Boundary counts within 1e-6: {'alpha_rl': {'lower': 0, 'upper': 0}, 'beta_rl': {'lower': 0, 'upper': 0}, 'beta_h': {'lower': 0, 'upper': 0}}. Full starts/failure diagnostics, training criteria, parameters, final Q/H, and counts are in JSON.

### Equal-value test comparisons

“Frequent” means stimulus 3, 4, or 7 for pairs 2–3, 4–5, 6–7. Observed and predicted rates below are trial-weighted over the identical retained comparisons. Predictions are one-step-ahead, using prior observed choices. They are not free-running simulated preferences.

| Pair | Model | Trials | Observed frequent rate | Predicted frequent rate | Log loss | Mean Qfrequent−Qrare |
| --- | --- | --- | --- | --- | --- | --- |
| 2-3 | RL | 43 | 0.697674 | 0.444551 | 0.716792 | -0.033576 |
| 2-3 | RL+CK | 43 | 0.697674 | 0.613128 | 0.570202 | 0.000000 |
| 2-3 | RL+cumulative | 43 | 0.697674 | 0.689774 | 0.605630 | -0.028773 |
| 4-5 | RL | 46 | 0.195652 | 0.558580 | 0.831556 | 0.040356 |
| 4-5 | RL+CK | 46 | 0.195652 | 0.555410 | 0.801220 | 0.000000 |
| 4-5 | RL+cumulative | 46 | 0.195652 | 0.711318 | 1.092643 | 0.005181 |
| 6-7 | RL | 47 | 0.744681 | 0.478718 | 0.725052 | -0.013411 |
| 6-7 | RL+CK | 47 | 0.744681 | 0.639248 | 0.536108 | 0.000000 |
| 6-7 | RL+cumulative | 47 | 0.744681 | 0.738400 | 0.514165 | -0.017717 |

## paper_intended

Complete paired participants: 4; failed participant fits: []. Failures are retained, not silently removed.

| Model | Total test NLL | Trial-weighted log loss | Participant mean log loss | Participant median log loss | Training total NLL |
| --- | --- | --- | --- | --- | --- |
| RL | 248.487083 | 0.47151249 | 0.47145197 | 0.47017674 | 1482.7959 |
| RL+CK | 228.464504 | 0.43351898 | 0.43209140 | 0.44536139 | 1278.3580 |
| RL+cumulative | 256.057851 | 0.48587828 | 0.48511968 | 0.47226001 | 1292.6292 |

| Comparison | Better / worse / tied | Mean gain | Median gain | P5 / P95 across people | 95% bootstrap CI for mean | Gain >.01 / loss >.01 |
| --- | --- | --- | --- | --- | --- | --- |
| cumulative vs RL | 2 / 2 / 0 | -0.01366772 | 0.00046719 | -0.071890 / 0.024766 | [-0.05997997, 0.01850963] | 1 / 1 |
| cumulative vs RL+CK | 0 / 4 / 0 | -0.05302828 | -0.04951811 | -0.095817 / -0.015154 | [-0.08435850, -0.02169807] | 0 / 4 |

Against RL: total held-out log-likelihood gain -7.570768 nats; mean participant absolute prediction-probability difference 0.126197.

Against RL+CK: total held-out log-likelihood gain -27.593347 nats; mean participant absolute prediction-probability difference 0.069039.

### Descriptive parameters and fitting diagnostics

beta_H: mean 2.148709, median 2.201634, SD 0.299026, P5 1.818472, P25 1.967586, P75 2.382757, P95 2.404850, min 1.781193, max 2.410374. Signs (tolerance 1e-8): 4 positive, 0 negative, 0 near zero. These are not individual habit-strength measurements.

Unsuccessful starts: 1. Boundary counts within 1e-6: {'alpha_rl': {'lower': 0, 'upper': 2}, 'beta_rl': {'lower': 0, 'upper': 0}, 'beta_h': {'lower': 0, 'upper': 0}}. Full starts/failure diagnostics, training criteria, parameters, final Q/H, and counts are in JSON.

### Equal-value test comparisons

“Frequent” means stimulus 3, 4, or 7 for pairs 2–3, 4–5, 6–7. Observed and predicted rates below are trial-weighted over the identical retained comparisons. Predictions are one-step-ahead, using prior observed choices. They are not free-running simulated preferences.

| Pair | Model | Trials | Observed frequent rate | Predicted frequent rate | Log loss | Mean Qfrequent−Qrare |
| --- | --- | --- | --- | --- | --- | --- |
| 2-3 | RL | 43 | 0.697674 | 0.509159 | 0.683744 | 0.030760 |
| 2-3 | RL+CK | 43 | 0.697674 | 0.652981 | 0.581426 | 0.003216 |
| 2-3 | RL+cumulative | 43 | 0.697674 | 0.697874 | 0.601255 | 0.000016 |
| 4-5 | RL | 46 | 0.195652 | 0.490339 | 0.678552 | -0.031341 |
| 4-5 | RL+CK | 46 | 0.195652 | 0.533022 | 0.720685 | -0.003464 |
| 4-5 | RL+cumulative | 46 | 0.195652 | 0.705666 | 1.091017 | -0.000019 |
| 6-7 | RL | 47 | 0.744681 | 0.504239 | 0.684814 | 0.011641 |
| 6-7 | RL+CK | 47 | 0.744681 | 0.668647 | 0.523724 | 0.000971 |
| 6-7 | RL+cumulative | 47 | 0.744681 | 0.742235 | 0.513881 | 0.000003 |

## Limits and interpretation

The paired participant bootstrap describes uncertainty in this cohort-level mean predictive contrast, assuming independent participants. It is not a confidence interval for any person’s beta_H, not a causal test of habit, and not evidence of planning. Previous synthetic single-trajectory calibration was poor; moreover that estimator used fixed utilities, whereas this milestone uses learned Q. Its exact error rates do not transfer to this new parameterization.

Holding the RL rule and Q initialization constant isolates the reward-learning component, not every property of the history families: the specified CK initializes from pre_val, cumulative history starts at zero, and cumulative beta_H is signed whereas authors beta_CK is nonnegative. Parameter counts alone do not establish fairness. The optional nonnegative-weight and shared-beta analyses are not performed; joint shared-weight nuisance optimization is deferred rather than introduced into this first participant-specific comparison.

No planning was fitted. A cumulative predictive advantage would not identify forward-looking decisions; a disadvantage would not establish that humans never use cumulative history. Final scientific conclusions are reviewed below after inspecting the complete predeclared cohort results.

<!-- Reviewed interpretation -->


