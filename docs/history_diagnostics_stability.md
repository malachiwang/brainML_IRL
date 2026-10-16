# Structural restart-stability subset (not cohort inference)

This four-participant subset checks restart minima only. See the full cohort report for scientific conclusions.

This analysis refits only the three existing models. It adds no mechanism, planning, regularization, reset, or new exclusion. The previously inspected final test is not used for fitting, scoring, diagnostics, or selection. Primary validation is S1–S4 → S5; secondary folds are S1 → S2, S1–S2 → S3, S1–S3 → S4. All held-out sessions still contain reward feedback: Q and histories continue updating after each scored response, without parameter refitting.

## Protocol and data support

4 participants; 3,022 retained training rows. Session counts: {'1': 593, '2': 608, '3': 599, '4': 615, '5': 607}. No missing trials are reconstructed. Primary restart count 200; rolling count 50. Runtime 6.84 s, including trace compression but excluding final JSON/report serialization. Fits: {'attempted': 12, 'failed': 0, 'successful_starts': 2398, 'unsuccessful_starts': 2}.

Unchanged bounded L-BFGS-B and analytic gradients are called through existing fitters. Authors starts use the established participant-index RNG convention (R seed 1234+17×index by default); cumulative starts use SeedSequence([20261015,index]). No five-session fitted parameters or final-test results are imported. Q/CK initialization uses the existing released-code-intended pre-training inputs; cumulative counts start at zero. pi=0 throughout.

Gain A over B = loss(B) − loss(A): positive always favors A. Inferential means weight participants equally. Bootstrap: 10,000 paired participant resamples, seed 20261015; percentile 95% intervals. Numerical ties: 1e-8 nats/choice. Rolling-session and phase comparisons are diagnostic, not multiplicity-adjusted confirmatory tests.

Observed unordered training pairs: {'1-2': 178, '1-3': 569, '2-4': 565, '3-5': 182, '4-6': 181, '5-7': 560, '6-8': 593, '7-8': 194}. Reward-gap counts: {'1.0': 3022}.

**Support limitation:** equal-value pairs absent from this run: ['2-3', '4-5', '6-7']. Observed reward gaps: ['1.0']. For absent pairs, empirical preference, conditional loss, calibration, and observed overconfidence are unavailable in training. Comparisons with absent reward-gap categories are also unavailable. We do not fabricate observations or recover them from final test. The state probes below are explicitly unobserved counterfactual contrasts, not additional validation data.

## Rolling-origin held-out performance

| Validation session | RL loss | RL+CK loss | Cumulative loss | Cumulative over RL | Cumulative over CK | CK over RL |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 0.430209 | 0.386324 | 0.379648 | 0.050562 | 0.006676 | 0.043886 |

### Session 5

Complete paired participants: 4; failed: [].

| Model | NLL | Trials | Mean participant loss | Median | Brier (participant mean) |
| --- | --- | --- | --- | --- | --- |
| RL | 261.393679 | 607 | 0.430209 | 0.419495 | 0.128454 |
| RL+CK | 234.665743 | 607 | 0.386324 | 0.384715 | 0.117581 |
| RL+cumulative | 230.836521 | 607 | 0.379648 | 0.370222 | 0.115754 |

| Gain | Mean | Median | P5 / P95 | 95% participant-bootstrap CI | Better / worse / tied |
| --- | --- | --- | --- | --- | --- |
| cumulative_over_rl | 0.050562 | 0.049273 | 0.036358 / 0.066569 | [0.038031689364580734, 0.06309164385038299] | 4 / 0 / 0 |
| cumulative_over_ck | 0.006676 | 0.007113 | -0.001935 / 0.014675 | [-0.001141709652456635, 0.014493427454533275] | 2 / 2 / 0 |
| ck_over_rl | 0.043886 | 0.039769 | 0.027167 / 0.066368 | [0.0302249151683449, 0.06166326570914417] | 4 / 0 / 0 |

Lowest-loss model counts: {'rl': 0, 'ck': 2, 'cumulative': 2}; numerical best-model ties: 0.

| Phase | RL loss | RL+CK loss | Cumulative loss | Cumulative over RL | Cumulative over CK |
| --- | --- | --- | --- | --- | --- |
| early | 0.494235 | 0.433570 | 0.431229 | 0.063006 | 0.002341 |
| middle | 0.363647 | 0.347574 | 0.332612 | 0.031035 | 0.014962 |
| late | 0.431596 | 0.376654 | 0.373962 | 0.057634 | 0.002692 |

Within-person late-minus-early change in cumulative-over-CK gain: {'n': 4, 'mean': 0.0003514890228271117, 'median': 0.0009356857017225884, 'sd': 0.012157023975531256, 'minimum': -0.014961776873747556, 'maximum': 0.014496361561610827, 'quantiles': {'p05': -0.012879465969635675, 'p25': -0.004550222353188152, 'p75': 0.005837397077737852, 'p95': 0.012764568664836228}}.

## Memory dynamics

For a seen stimulus, cumulative R changes by `(chosen_indicator − R)/(N+1)`; unseen stimuli begin at R=.5 and first update by ±.5. An old cumulative choice has weight 1/N, not a fixed-size response to a new choice. CK changes by `alpha_CK*(chosen_indicator − H)`, only when that stimulus is displayed. Its old-state multiplier after k relevant presentations is `(1-alpha_CK)^k`. These are encounter counts, not total trials, days, or a calibrated psychological memory duration.

Interior CK half-life is log(.5)/log(1−alpha). Alpha=0 is infinite/no updating (also no new choice enters); alpha=1 erases old state at the first update, reported as one update. Interior fractional half-lives are kept as such. Finite-only means explicitly exclude infinite cases. CK pre_val can initially exceed 1; initial raw-state magnitudes therefore need caution.

| Held-out session | Mean alpha_CK | Median alpha_CK | Mean finite half-life | Median | Q25 / Q75 | Infinite / alpha=1 |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 0.096012 | 0.110270 | 32.417332 | 6.163870 | 4.712783 / 33.868419 | 0 / 0 |

Fixed S1–S4-fitted parameters replayed over all five sessions: S1–S4 here are **descriptive fitted-state replay**, not validation. This separates the count-denominator effect from changing fitted parameters across folds. Means below average participant mean changes over displayed-stimulus updates; undisplayed zeros are not included. All stimulus-by-session values and strict rolling-fold update summaries are saved in JSON.

| Session | CK raw update | Cumulative raw update | CK weighted update | Cumulative weighted update |
| --- | --- | --- | --- | --- |
| 1 | 0.041086 | 0.051000 | 0.084395 | 0.110456 |
| 2 | 0.029179 | 0.006312 | 0.059803 | 0.013592 |
| 3 | 0.028988 | 0.003650 | 0.058851 | 0.007806 |
| 4 | 0.029021 | 0.002479 | 0.058357 | 0.005258 |
| 5 | 0.028825 | 0.001903 | 0.058703 | 0.004062 |

## Unobserved equal-value state contrasts

Each actual pre-choice state supplies hypothetical frequent-minus-rare Q and H contrasts for 3−2, 4−5, 7−6. No hypothetical choice or feedback is generated, and states are not updated for these probes. Their predicted preference cannot be compared with an observed training equal-pair rate, because no such rate exists.

| Session | Pair | CK history logit | Cumulative history logit | RL hypothetical P(frequent) | CK hypothetical P | Cumulative hypothetical P |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 2-3 | 0.967045 | 0.927522 | 0.459578 | 0.715193 | 0.711149 |
| 5 | 4-5 | 1.160103 | 1.060128 | 0.546068 | 0.756956 | 0.733702 |
| 5 | 6-7 | 1.087293 | 1.139518 | 0.483102 | 0.745514 | 0.751674 |

## Reward-gap support in primary S5

| Reward category | Model | Trials | Participant mean log loss |
| --- | --- | --- | --- |
| equal | RL | 0 | NA |
| equal | RL+CK | 0 | NA |
| equal | RL+cumulative | 0 | NA |
| small | RL | 607 | 0.430209 |
| small | RL+CK | 607 | 0.386324 |
| small | RL+cumulative | 607 | 0.379648 |
| larger | RL | 0 | NA |
| larger | RL+CK | 0 | NA |
| larger | RL+cumulative | 0 | NA |

NA denotes no observations, not zero loss. These training data cannot establish whether a deficit is concentrated in equal-reward choices or depends on larger reward gaps.

## Calibration across rolling sessions

| Session | RL pooled ECE | CK pooled ECE | Cumulative pooled ECE |
| --- | --- | --- | --- |
| 5 | 0.086561 | 0.046208 | 0.031043 |

ECE is the predeclared-bin weighted absolute confidence/accuracy difference, not an optimized criterion. Pooled ECE can conceal participant heterogeneity; per-participant ECE distributions remain in JSON.

## Primary S5 calibration and logit decomposition

Calibration uses favored-option confidence max(Pleft,1−Pleft), with left favored at exactly .5. Bins are [.50,.55), …, [.95,1.00]. Overconfidence is mean confidence minus observed favored-option accuracy. These tables pool trials descriptively; participant Brier scores and participant ECE are also stored. No calibration results change the fitted models.

### RL

Pooled ECE: 0.086561; mean participant ECE: 0.086037.

| Lower | Upper | n | Mean confidence | Empirical accuracy | Overconfidence |
| --- | --- | --- | --- | --- | --- |
| 0.500000 | 0.550000 | 0 | NA | NA | NA |
| 0.550000 | 0.600000 | 0 | NA | NA | NA |
| 0.600000 | 0.650000 | 0 | NA | NA | NA |
| 0.650000 | 0.700000 | 0 | NA | NA | NA |
| 0.700000 | 0.750000 | 0 | NA | NA | NA |
| 0.750000 | 0.800000 | 314 | 0.779355 | 0.878981 | -0.099626 |
| 0.800000 | 0.850000 | 136 | 0.815922 | 0.845588 | -0.029666 |
| 0.850000 | 0.900000 | 41 | 0.878583 | 0.536585 | 0.341998 |
| 0.900000 | 0.950000 | 61 | 0.941843 | 0.918033 | 0.023811 |
| 0.950000 | 1.000000 | 55 | 0.959108 | 0.927273 | 0.031835 |

### RL+CK

Pooled ECE: 0.046208; mean participant ECE: 0.071802.

| Lower | Upper | n | Mean confidence | Empirical accuracy | Overconfidence |
| --- | --- | --- | --- | --- | --- |
| 0.500000 | 0.550000 | 9 | 0.530097 | 0.444444 | 0.085653 |
| 0.550000 | 0.600000 | 18 | 0.570398 | 0.555556 | 0.014842 |
| 0.600000 | 0.650000 | 13 | 0.634117 | 0.384615 | 0.249502 |
| 0.650000 | 0.700000 | 32 | 0.674698 | 0.656250 | 0.018448 |
| 0.700000 | 0.750000 | 53 | 0.726282 | 0.584906 | 0.141376 |
| 0.750000 | 0.800000 | 50 | 0.776764 | 0.840000 | -0.063236 |
| 0.800000 | 0.850000 | 74 | 0.828205 | 0.878378 | -0.050173 |
| 0.850000 | 0.900000 | 126 | 0.878516 | 0.936508 | -0.057991 |
| 0.900000 | 0.950000 | 178 | 0.920812 | 0.915730 | 0.005082 |
| 0.950000 | 1.000000 | 54 | 0.955508 | 0.944444 | 0.011064 |

### RL+cumulative

Pooled ECE: 0.031043; mean participant ECE: 0.064628.

| Lower | Upper | n | Mean confidence | Empirical accuracy | Overconfidence |
| --- | --- | --- | --- | --- | --- |
| 0.500000 | 0.550000 | 10 | 0.542687 | 0.600000 | -0.057313 |
| 0.550000 | 0.600000 | 3 | 0.595437 | 1.000000 | -0.404563 |
| 0.600000 | 0.650000 | 46 | 0.616356 | 0.586957 | 0.029400 |
| 0.650000 | 0.700000 | 13 | 0.688784 | 0.692308 | -0.003523 |
| 0.700000 | 0.750000 | 38 | 0.738393 | 0.657895 | 0.080498 |
| 0.750000 | 0.800000 | 86 | 0.760502 | 0.755814 | 0.004688 |
| 0.800000 | 0.850000 | 73 | 0.832518 | 0.904110 | -0.071592 |
| 0.850000 | 0.900000 | 185 | 0.888398 | 0.918919 | -0.030521 |
| 0.900000 | 0.950000 | 153 | 0.932528 | 0.941176 | -0.008649 |
| 0.950000 | 1.000000 | 0 | NA | NA | NA |

| Model | Mean absolute reward logit | Mean absolute history logit | History dominates fraction | Mean ratio (abs reward >1e-6) | Excluded ratio trials |
| --- | --- | --- | --- | --- | --- |
| RL | 1.678184 | 0.000000 | 0.000000 | 0.000000 | 0 |
| RL+CK | 0.893566 | 1.176092 | 0.684134 | 1.419437 | 0 |
| RL+cumulative | 0.893823 | 1.080637 | 0.652019 | 1.293829 | 0 |

Raw DeltaH and full per-participant distributions are saved alongside weighted contributions. The ratio can be large near zero reward logits; the >1e-6 denominator rule and excluded count are explicit. History dominance in equal-value decisions is not empirically estimable here; unobserved probes cannot fill that gap.

## Primary S5 model disagreements

| abs(Pck−Pcum) threshold | n | Fraction | CK assigns higher P to actual choice | Cumulative higher | CK LL gain / selected trial |
| --- | --- | --- | --- | --- | --- |
| 0.1 | 31 | 0.051071 | 14 | 17 | -0.065873 |
| 0.2 | 0 | 0.000000 | 0 | 0 | NA |

Threshold 0.1: positions {'early': 10, 'late': 13, 'middle': 8}; pair counts {'1-2': 6, '1-3': 2, '2-4': 3, '3-5': 11, '4-6': 1, '5-7': 8, '6-8': 0, '7-8': 0}; reward gaps {'1.0': 31}.

Threshold 0.2: positions {'early': 0, 'late': 0, 'middle': 0}; pair counts {'1-2': 0, '1-3': 0, '2-4': 0, '3-5': 0, '4-6': 0, '5-7': 0, '6-8': 0, '7-8': 0}; reward gaps {'1.0': 0}.

Exploratory association (no individual psychological interpretation): alpha_CK versus CK gain over cumulative: {'n': 4, 'pearson': -0.5452870026302901, 'spearman': -0.39999999999999997}; finite half-life versus CK gain: {'n': 4, 'pearson': 0.6514248799380831, 'spearman': 0.39999999999999997}. Both quantities are fitted or scored on the same participant, so these are descriptive associations, not causal tests of recency.

## Reproducibility and boundaries

Full precision participant/fold summaries, parameters, start outcomes, confidence-bin counts, pair probes, and bootstrap intervals are in the companion JSON. Every validation trial’s pre-choice Q/H, raw and weighted logits, predicted probability, loss, and post-choice history change is recorded in the local ignored NPZ archive: `data/derived/history_diagnostics_stability_traces.npz`. Its SHA-256 and key schema are in JSON. The RData container is deserialized by the loader, but the pipeline never accesses the final-test object. Phase guards reject test rows even if they carry session label 5, as well as all session-6 rows. Protected model source hashes are checked before and after fitting. No original model equations or optimizer settings were edited.

These comparisons do not causally isolate recency from CK’s additional free update-rate parameter, its initialization, or representation scale. They also cannot establish that a deficit is *specific* to the training-to-test transition, because this milestone deliberately does not score test. No new history model or planning model was fitted. No commit or push was made.

<!-- Reviewed findings -->


