# Training-only phase-resolved history diagnostics

This analysis refits only the three existing models. It adds no mechanism, planning, regularization, reset, or new exclusion. The previously inspected final test is not used for fitting, scoring, diagnostics, or selection. Primary validation is S1–S4 → S5; secondary folds are S1 → S2, S1–S2 → S3, S1–S3 → S4. All held-out sessions still contain reward feedback: Q and histories continue updating after each scored response, without parameter refitting.

## Protocol and data support

213 participants; 164,087 retained training rows. Session counts: {'1': 31815, '2': 32739, '3': 33052, '4': 33160, '5': 33321}. No missing trials are reconstructed. Primary restart count 200; rolling count 50. Runtime 284.69 s, including trace compression but excluding final JSON/report serialization. Fits: {'attempted': 2556, 'failed': 0, 'successful_starts': 223521, 'unsuccessful_starts': 129}.

Unchanged bounded L-BFGS-B and analytic gradients are called through existing fitters. Authors starts use the established participant-index RNG convention (R seed 1234+17×index by default); cumulative starts use SeedSequence([20261015,index]). No five-session fitted parameters or final-test results are imported. Q/CK initialization uses the existing released-code-intended pre-training inputs; cumulative counts start at zero. pi=0 throughout.

Gain A over B = loss(B) − loss(A): positive always favors A. Inferential means weight participants equally. Bootstrap: 10,000 paired participant resamples, seed 20261015; percentile 95% intervals. Numerical ties: 1e-8 nats/choice. Rolling-session and phase comparisons are diagnostic, not multiplicity-adjusted confirmatory tests.

Observed unordered training pairs: {'1-2': 10094, '1-3': 30739, '2-4': 30644, '3-5': 9858, '4-6': 10071, '5-7': 30787, '6-8': 31449, '7-8': 10445}. Reward-gap counts: {'1.0': 164087}.

**Support limitation:** equal-value pairs absent from this run: ['2-3', '4-5', '6-7']. Observed reward gaps: ['1.0']. For absent pairs, empirical preference, conditional loss, calibration, and observed overconfidence are unavailable in training. Comparisons with absent reward-gap categories are also unavailable. We do not fabricate observations or recover them from final test. The state probes below are explicitly unobserved counterfactual contrasts, not additional validation data.

## Rolling-origin held-out performance

| Validation session | RL loss | RL+CK loss | Cumulative loss | Cumulative over RL | Cumulative over CK | CK over RL |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.484298 | 0.445734 | 0.439599 | 0.044699 | 0.006136 | 0.038564 |
| 3 | 0.439530 | 0.396493 | 0.390642 | 0.048888 | 0.005851 | 0.043037 |
| 4 | 0.411502 | 0.369354 | 0.367064 | 0.044437 | 0.002290 | 0.042147 |
| 5 | 0.408043 | 0.366074 | 0.364685 | 0.043358 | 0.001390 | 0.041969 |

### Session 2

Complete paired participants: 213; failed: [].

| Model | NLL | Trials | Mean participant loss | Median | Brier (participant mean) |
| --- | --- | --- | --- | --- | --- |
| RL | 15839.885434 | 32739 | 0.484298 | 0.492673 | 0.153771 |
| RL+CK | 14574.024750 | 32739 | 0.445734 | 0.451601 | 0.140605 |
| RL+cumulative | 14374.182807 | 32739 | 0.439599 | 0.443024 | 0.138238 |

| Gain | Mean | Median | P5 / P95 | 95% participant-bootstrap CI | Better / worse / tied |
| --- | --- | --- | --- | --- | --- |
| cumulative_over_rl | 0.044699 | 0.037434 | -0.003767 / 0.122765 | [0.03933758394955213, 0.0500643031063503] | 196 / 17 / 0 |
| cumulative_over_ck | 0.006136 | 0.005888 | -0.027301 / 0.037776 | [0.0032788844108552198, 0.009073431810205423] | 134 / 79 / 0 |
| ck_over_rl | 0.038564 | 0.031089 | -0.010207 / 0.123180 | [0.033180173772153376, 0.04412140951956156] | 184 / 24 / 5 |

Lowest-loss model counts: {'rl': 9, 'ck': 73, 'cumulative': 129}; numerical best-model ties: 2.

| Phase | RL loss | RL+CK loss | Cumulative loss | Cumulative over RL | Cumulative over CK |
| --- | --- | --- | --- | --- | --- |
| early | 0.522760 | 0.485143 | 0.478852 | 0.043907 | 0.006290 |
| middle | 0.465512 | 0.424280 | 0.418881 | 0.046632 | 0.005399 |
| late | 0.464232 | 0.427358 | 0.420634 | 0.043598 | 0.006724 |

Within-person late-minus-early change in cumulative-over-CK gain: {'n': 213, 'mean': 0.00043411960459588996, 'median': -0.00042969004999909854, 'sd': 0.031474546733034445, 'minimum': -0.09220343148316812, 'maximum': 0.1747656555489332, 'quantiles': {'p05': -0.042751782154951255, 'p25': -0.01599631563665943, 'p75': 0.014998491098428213, 'p95': 0.050607376704016896}}.

### Session 3

Complete paired participants: 213; failed: [].

| Model | NLL | Trials | Mean participant loss | Median | Brier (participant mean) |
| --- | --- | --- | --- | --- | --- |
| RL | 14508.572651 | 33052 | 0.439530 | 0.441611 | 0.135195 |
| RL+CK | 13089.743377 | 33052 | 0.396493 | 0.395476 | 0.121318 |
| RL+cumulative | 12894.670834 | 33052 | 0.390642 | 0.387265 | 0.119462 |

| Gain | Mean | Median | P5 / P95 | 95% participant-bootstrap CI | Better / worse / tied |
| --- | --- | --- | --- | --- | --- |
| cumulative_over_rl | 0.048888 | 0.041294 | 0.001941 / 0.115615 | [0.04408847629207278, 0.05369430246958741] | 207 / 6 / 0 |
| cumulative_over_ck | 0.005851 | 0.005607 | -0.016125 / 0.026907 | [0.004179486061787989, 0.00754703042281137] | 154 / 59 / 0 |
| ck_over_rl | 0.043037 | 0.037542 | -0.000300 / 0.112459 | [0.03850232785315738, 0.0476536756335975] | 201 / 12 / 0 |

Lowest-loss model counts: {'rl': 5, 'ck': 57, 'cumulative': 151}; numerical best-model ties: 0.

| Phase | RL loss | RL+CK loss | Cumulative loss | Cumulative over RL | Cumulative over CK |
| --- | --- | --- | --- | --- | --- |
| early | 0.472301 | 0.432410 | 0.424909 | 0.047393 | 0.007501 |
| middle | 0.425335 | 0.382706 | 0.378744 | 0.046591 | 0.003963 |
| late | 0.420633 | 0.373972 | 0.367894 | 0.052739 | 0.006079 |

Within-person late-minus-early change in cumulative-over-CK gain: {'n': 213, 'mean': -0.00142201592042409, 'median': -0.0010023278223007526, 'sd': 0.02432903761060837, 'minimum': -0.08260302513858842, 'maximum': 0.0748635539293766, 'quantiles': {'p05': -0.0403897957783654, 'p25': -0.015476535496405486, 'p75': 0.01200075750136853, 'p95': 0.03699553749472641}}.

### Session 4

Complete paired participants: 213; failed: [].

| Model | NLL | Trials | Mean participant loss | Median | Brier (participant mean) |
| --- | --- | --- | --- | --- | --- |
| RL | 13630.602717 | 33160 | 0.411502 | 0.409571 | 0.123686 |
| RL+CK | 12230.517432 | 33160 | 0.369354 | 0.358692 | 0.110854 |
| RL+cumulative | 12153.372554 | 33160 | 0.367064 | 0.357605 | 0.110489 |

| Gain | Mean | Median | P5 / P95 | 95% participant-bootstrap CI | Better / worse / tied |
| --- | --- | --- | --- | --- | --- |
| cumulative_over_rl | 0.044437 | 0.037915 | -0.003152 / 0.104373 | [0.03965914137231729, 0.049392286109975356] | 194 / 19 / 0 |
| cumulative_over_ck | 0.002290 | 0.003492 | -0.019630 / 0.019149 | [0.0005433683673541367, 0.003956076621216616] | 133 / 80 / 0 |
| ck_over_rl | 0.042147 | 0.032816 | -0.005041 / 0.112695 | [0.037363472924292096, 0.047040324470160086] | 193 / 20 / 0 |

Lowest-loss model counts: {'rl': 12, 'ck': 75, 'cumulative': 126}; numerical best-model ties: 0.

| Phase | RL loss | RL+CK loss | Cumulative loss | Cumulative over RL | Cumulative over CK |
| --- | --- | --- | --- | --- | --- |
| early | 0.438435 | 0.397189 | 0.393962 | 0.044473 | 0.003227 |
| middle | 0.398170 | 0.356720 | 0.355206 | 0.042964 | 0.001514 |
| late | 0.397726 | 0.353937 | 0.351822 | 0.045904 | 0.002115 |

Within-person late-minus-early change in cumulative-over-CK gain: {'n': 213, 'mean': -0.0011127802637440835, 'median': -0.001156915442779738, 'sd': 0.024808184029275787, 'minimum': -0.10803739874536139, 'maximum': 0.08255186497037575, 'quantiles': {'p05': -0.03622507435018776, 'p25': -0.011968710109216019, 'p75': 0.010125085406805623, 'p95': 0.03706040017432924}}.

### Session 5

Complete paired participants: 213; failed: [].

| Model | NLL | Trials | Mean participant loss | Median | Brier (participant mean) |
| --- | --- | --- | --- | --- | --- |
| RL | 13587.557172 | 33321 | 0.408043 | 0.409473 | 0.122552 |
| RL+CK | 12187.871204 | 33321 | 0.366074 | 0.359593 | 0.109876 |
| RL+cumulative | 12140.917812 | 33321 | 0.364685 | 0.358122 | 0.109626 |

| Gain | Mean | Median | P5 / P95 | 95% participant-bootstrap CI | Better / worse / tied |
| --- | --- | --- | --- | --- | --- |
| cumulative_over_rl | 0.043358 | 0.036185 | -0.003956 / 0.116877 | [0.038241463775320846, 0.04844356917868386] | 197 / 16 / 0 |
| cumulative_over_ck | 0.001390 | 0.002430 | -0.016189 / 0.017553 | [-0.00012978893626317323, 0.0028313649454004045] | 129 / 84 / 0 |
| ck_over_rl | 0.041969 | 0.031641 | -0.004422 / 0.118702 | [0.036878070567501385, 0.047159114508183476] | 193 / 20 / 0 |

Lowest-loss model counts: {'rl': 13, 'ck': 74, 'cumulative': 126}; numerical best-model ties: 0.

| Phase | RL loss | RL+CK loss | Cumulative loss | Cumulative over RL | Cumulative over CK |
| --- | --- | --- | --- | --- | --- |
| early | 0.436928 | 0.397549 | 0.395840 | 0.041088 | 0.001709 |
| middle | 0.390416 | 0.345110 | 0.343232 | 0.047184 | 0.001879 |
| late | 0.396538 | 0.355304 | 0.354721 | 0.041817 | 0.000583 |

Within-person late-minus-early change in cumulative-over-CK gain: {'n': 213, 'mean': -0.0011257010718664342, 'median': -5.8157633033240064e-05, 'sd': 0.025484805136290704, 'minimum': -0.12233627160135802, 'maximum': 0.08239476100657983, 'quantiles': {'p05': -0.041288451925701866, 'p25': -0.013474465040088257, 'p75': 0.011298738768616173, 'p95': 0.034447358546607396}}.

## Memory dynamics

For a seen stimulus, cumulative R changes by `(chosen_indicator − R)/(N+1)`; unseen stimuli begin at R=.5 and first update by ±.5. An old cumulative choice has weight 1/N, not a fixed-size response to a new choice. CK changes by `alpha_CK*(chosen_indicator − H)`, only when that stimulus is displayed. Its old-state multiplier after k relevant presentations is `(1-alpha_CK)^k`. These are encounter counts, not total trials, days, or a calibrated psychological memory duration.

Interior CK half-life is log(.5)/log(1−alpha). Alpha=0 is infinite/no updating (also no new choice enters); alpha=1 erases old state at the first update, reported as one update. Interior fractional half-lives are kept as such. Finite-only means explicitly exclude infinite cases. CK pre_val can initially exceed 1; initial raw-state magnitudes therefore need caution.

| Held-out session | Mean alpha_CK | Median alpha_CK | Mean finite half-life | Median | Q25 / Q75 | Infinite / alpha=1 |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.341510 | 0.252416 | 9.617324 | 2.230725 | 1.086597 / 4.631636 | 13 / 21 |
| 3 | 0.240066 | 0.179582 | 8.560469 | 3.484843 | 2.031221 / 5.935902 | 1 / 7 |
| 4 | 0.203885 | 0.146257 | 12.765055 | 4.383533 | 2.701084 / 7.532828 | 0 / 4 |
| 5 | 0.178073 | 0.128624 | 10.911908 | 5.034428 | 3.031376 / 8.053478 | 0 / 2 |

Fixed S1–S4-fitted parameters replayed over all five sessions: S1–S4 here are **descriptive fitted-state replay**, not validation. This separates the count-denominator effect from changing fitted parameters across folds. Means below average participant mean changes over displayed-stimulus updates; undisplayed zeros are not included. All stimulus-by-session values and strict rolling-fold update summaries are saved in JSON.

| Session | CK raw update | Cumulative raw update | CK weighted update | Cumulative weighted update |
| --- | --- | --- | --- | --- |
| 1 | 0.072746 | 0.048565 | 0.095058 | 0.089405 |
| 2 | 0.060544 | 0.006580 | 0.076250 | 0.011996 |
| 3 | 0.058341 | 0.003652 | 0.073684 | 0.006653 |
| 4 | 0.058304 | 0.002523 | 0.073418 | 0.004591 |
| 5 | 0.057491 | 0.001932 | 0.073707 | 0.003524 |

## Unobserved equal-value state contrasts

Each actual pre-choice state supplies hypothetical frequent-minus-rare Q and H contrasts for 3−2, 4−5, 7−6. No hypothetical choice or feedback is generated, and states are not updated for these probes. Their predicted preference cannot be compared with an observed training equal-pair rate, because no such rate exists.

| Session | Pair | CK history logit | Cumulative history logit | RL hypothetical P(frequent) | CK hypothetical P | Cumulative hypothetical P |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 2-3 | 0.499520 | 0.518778 | 0.507395 | 0.614925 | 0.622656 |
| 2 | 4-5 | 0.535437 | 0.551433 | 0.502148 | 0.622888 | 0.627211 |
| 2 | 6-7 | 0.555621 | 0.539076 | 0.504562 | 0.627198 | 0.627622 |
| 3 | 2-3 | 0.657172 | 0.706741 | 0.505580 | 0.650717 | 0.663053 |
| 3 | 4-5 | 0.733022 | 0.767839 | 0.501903 | 0.666070 | 0.674679 |
| 3 | 6-7 | 0.690279 | 0.739880 | 0.503626 | 0.658738 | 0.671498 |
| 4 | 2-3 | 0.717840 | 0.786125 | 0.505465 | 0.664295 | 0.680585 |
| 4 | 4-5 | 0.803852 | 0.862254 | 0.502540 | 0.682184 | 0.695416 |
| 4 | 6-7 | 0.763416 | 0.832176 | 0.503525 | 0.675400 | 0.691540 |
| 5 | 2-3 | 0.758583 | 0.825051 | 0.504237 | 0.673841 | 0.688909 |
| 5 | 4-5 | 0.821653 | 0.903069 | 0.502665 | 0.686053 | 0.704543 |
| 5 | 6-7 | 0.775879 | 0.869274 | 0.503020 | 0.678610 | 0.699920 |

## Reward-gap support in primary S5

| Reward category | Model | Trials | Participant mean log loss |
| --- | --- | --- | --- |
| equal | RL | 0 | NA |
| equal | RL+CK | 0 | NA |
| equal | RL+cumulative | 0 | NA |
| small | RL | 33321 | 0.408043 |
| small | RL+CK | 33321 | 0.366074 |
| small | RL+cumulative | 33321 | 0.364685 |
| larger | RL | 0 | NA |
| larger | RL+CK | 0 | NA |
| larger | RL+cumulative | 0 | NA |

NA denotes no observations, not zero loss. These training data cannot establish whether a deficit is concentrated in equal-reward choices or depends on larger reward gaps.

## Calibration across rolling sessions

| Session | RL pooled ECE | CK pooled ECE | Cumulative pooled ECE |
| --- | --- | --- | --- |
| 2 | 0.045815 | 0.039467 | 0.035603 |
| 3 | 0.040886 | 0.033774 | 0.030032 |
| 4 | 0.039258 | 0.031581 | 0.027066 |
| 5 | 0.026968 | 0.019981 | 0.016075 |

ECE is the predeclared-bin weighted absolute confidence/accuracy difference, not an optimized criterion. Pooled ECE can conceal participant heterogeneity; per-participant ECE distributions remain in JSON.

## Primary S5 calibration and logit decomposition

Calibration uses favored-option confidence max(Pleft,1−Pleft), with left favored at exactly .5. Bins are [.50,.55), …, [.95,1.00]. Overconfidence is mean confidence minus observed favored-option accuracy. These tables pool trials descriptively; participant Brier scores and participant ECE are also stored. No calibration results change the fitted models.

### RL

Pooled ECE: 0.026968; mean participant ECE: 0.043915.

| Lower | Upper | n | Mean confidence | Empirical accuracy | Overconfidence |
| --- | --- | --- | --- | --- | --- |
| 0.500000 | 0.550000 | 2 | 0.539430 | 0.500000 | 0.039430 |
| 0.550000 | 0.600000 | 8 | 0.575337 | 0.625000 | -0.049663 |
| 0.600000 | 0.650000 | 30 | 0.630440 | 0.233333 | 0.397106 |
| 0.650000 | 0.700000 | 969 | 0.678954 | 0.712074 | -0.033120 |
| 0.700000 | 0.750000 | 1976 | 0.732207 | 0.786943 | -0.054736 |
| 0.750000 | 0.800000 | 6491 | 0.779180 | 0.815899 | -0.036719 |
| 0.800000 | 0.850000 | 12224 | 0.828484 | 0.858475 | -0.029991 |
| 0.850000 | 0.900000 | 8155 | 0.874472 | 0.888903 | -0.014430 |
| 0.900000 | 0.950000 | 3305 | 0.915490 | 0.921936 | -0.006447 |
| 0.950000 | 1.000000 | 161 | 0.962881 | 0.950311 | 0.012570 |

### RL+CK

Pooled ECE: 0.019981; mean participant ECE: 0.068010.

| Lower | Upper | n | Mean confidence | Empirical accuracy | Overconfidence |
| --- | --- | --- | --- | --- | --- |
| 0.500000 | 0.550000 | 936 | 0.524936 | 0.520299 | 0.004637 |
| 0.550000 | 0.600000 | 927 | 0.574252 | 0.573894 | 0.000357 |
| 0.600000 | 0.650000 | 1017 | 0.624868 | 0.647001 | -0.022133 |
| 0.650000 | 0.700000 | 1247 | 0.675944 | 0.676824 | -0.000880 |
| 0.700000 | 0.750000 | 1780 | 0.726300 | 0.739888 | -0.013588 |
| 0.750000 | 0.800000 | 2782 | 0.777070 | 0.807333 | -0.030263 |
| 0.800000 | 0.850000 | 4792 | 0.827693 | 0.853506 | -0.025813 |
| 0.850000 | 0.900000 | 8356 | 0.877301 | 0.905098 | -0.027797 |
| 0.900000 | 0.950000 | 9349 | 0.923273 | 0.940421 | -0.017149 |
| 0.950000 | 1.000000 | 2135 | 0.961673 | 0.967681 | -0.006009 |

### RL+cumulative

Pooled ECE: 0.016075; mean participant ECE: 0.067011.

| Lower | Upper | n | Mean confidence | Empirical accuracy | Overconfidence |
| --- | --- | --- | --- | --- | --- |
| 0.500000 | 0.550000 | 966 | 0.525142 | 0.552795 | -0.027653 |
| 0.550000 | 0.600000 | 1094 | 0.574884 | 0.606033 | -0.031149 |
| 0.600000 | 0.650000 | 977 | 0.625272 | 0.633572 | -0.008300 |
| 0.650000 | 0.700000 | 992 | 0.676692 | 0.668347 | 0.008345 |
| 0.700000 | 0.750000 | 1680 | 0.726262 | 0.739286 | -0.013023 |
| 0.750000 | 0.800000 | 2246 | 0.775349 | 0.789403 | -0.014054 |
| 0.800000 | 0.850000 | 4560 | 0.830183 | 0.853728 | -0.023545 |
| 0.850000 | 0.900000 | 8046 | 0.877707 | 0.897465 | -0.019757 |
| 0.900000 | 0.950000 | 10133 | 0.923717 | 0.935360 | -0.011643 |
| 0.950000 | 1.000000 | 2627 | 0.960908 | 0.968786 | -0.007878 |

| Model | Mean absolute reward logit | Mean absolute history logit | History dominates fraction | Mean ratio (abs reward >1e-6) | Excluded ratio trials |
| --- | --- | --- | --- | --- | --- |
| RL | 1.636030 | 0.000000 | 0.000000 | 0.000000 | 0 |
| RL+CK | 1.221998 | 0.838660 | 0.295593 | 0.801317 | 0 |
| RL+cumulative | 1.200349 | 0.905874 | 0.312549 | 0.865962 | 0 |

Raw DeltaH and full per-participant distributions are saved alongside weighted contributions. The ratio can be large near zero reward logits; the >1e-6 denominator rule and excluded count are explicit. History dominance in equal-value decisions is not empirically estimable here; unobserved probes cannot fill that gap.

## Primary S5 model disagreements

| abs(Pck−Pcum) threshold | n | Fraction | CK assigns higher P to actual choice | Cumulative higher | CK LL gain / selected trial |
| --- | --- | --- | --- | --- | --- |
| 0.1 | 2039 | 0.061193 | 913 | 1126 | 0.013012 |
| 0.2 | 269 | 0.008073 | 119 | 150 | 0.007040 |

Threshold 0.1: positions {'early': 692, 'late': 708, 'middle': 639}; pair counts {'1-2': 130, '1-3': 149, '2-4': 358, '3-5': 367, '4-6': 474, '5-7': 376, '6-8': 90, '7-8': 95}; reward gaps {'1.0': 2039}.

Threshold 0.2: positions {'early': 95, 'late': 93, 'middle': 81}; pair counts {'1-2': 2, '1-3': 14, '2-4': 61, '3-5': 50, '4-6': 73, '5-7': 59, '6-8': 9, '7-8': 1}; reward gaps {'1.0': 269}.

Exploratory association (no individual psychological interpretation): alpha_CK versus CK gain over cumulative: {'n': 213, 'pearson': -0.10017048278067968, 'spearman': -0.1416435924445146}; finite half-life versus CK gain: {'n': 213, 'pearson': 0.30366993105192125, 'spearman': 0.14284317259146323}. Both quantities are fitted or scored on the same participant, so these are descriptive associations, not causal tests of recency.

## Reproducibility and boundaries

Full precision participant/fold summaries, parameters, start outcomes, confidence-bin counts, pair probes, and bootstrap intervals are in the companion JSON. Every validation trial’s pre-choice Q/H, raw and weighted logits, predicted probability, loss, and post-choice history change is recorded in the local ignored NPZ archive: `data/derived/history_diagnostics_results_traces.npz`. Its SHA-256 and key schema are in JSON. The RData container is deserialized by the loader, but the pipeline never accesses the final-test object. Phase guards reject test rows even if they carry session label 5, as well as all session-6 rows. Protected model source hashes are checked before and after fitting. No original model equations or optimizer settings were edited.

These comparisons do not causally isolate recency from CK’s additional free update-rate parameter, its initialization, or representation scale. They also cannot establish that a deficit is *specific* to the training-to-test transition, because this milestone deliberately does not score test. No new history model or planning model was fitted. No commit or push was made.

<!-- Reviewed findings -->

## Reviewed findings: the training-only result does not reproduce the deficit

**Cumulative history does not fail to generalize within the observed reinforced
training schedule.** It improves on RL substantially in all four rolling folds.
It has lower mean loss than RL+CK in every fold, although the primary S5 difference
is small and its bootstrap interval spans zero. This does not establish formal
equivalence between the two history models.

For the primary S1–S4 → S5 split, 130,766 prefix-training rows supply the fits and
33,321 previously unfitted S5 responses supply validation. Cumulative gains
0.043358 nats/choice over RL (95% participant CI 0.038241–0.048444; 197 wins,
16 losses), but only 0.001390 over CK (CI −0.000130–0.002831; 129 wins, 84 losses).
CK itself gains 0.041969 over RL (CI 0.036878–0.047159; 193 wins, 20 losses).
All comparisons have zero numerical ties in S5. Lowest-loss three-model counts
are 126 cumulative, 74 CK, and 13 RL.

### When does CK overtake cumulative history?

**Not at the cohort mean in any observed training fold.** Cumulative's mean
advantage over CK narrows from 0.006136 in S2 to 0.005851 in S3, 0.002290 in S4,
and 0.001390 in S5. CK catches up, but does not overtake it in these point estimates.
The earlier rolling intervals favor cumulative, but are descriptive secondary
comparisons, not a collection of multiplicity-adjusted confirmatory tests.

Within S5, cumulative-over-CK gains are +0.001709 early, +0.001879 middle, and
+0.000583 late; all three phase-specific intervals include zero. Cumulative loss
is 0.395840, 0.343232, and 0.354721 across those terciles, respectively. Thus the
late tercile is somewhat worse than the middle one but better than the early one;
CK shows the same broad pattern. There is no monotonic within-session
deterioration unique to cumulative history. Absolute loss decreases across the
rolling sessions for all models; comparing sessions also changes the fitted
prefix length, so this trend alone cannot identify a temporal mechanism.

### Measured inertia is real; harmful inertia in training is not established

With fixed S1–S4-fitted parameters, cumulative raw per-presentation changes shrink
from 0.048565 in S1 to 0.001932 in S5. CK changes shrink much less, from 0.072746
to 0.057491. In directly comparable weighted-logit units, the corresponding S5
changes are 0.003524 cumulative versus 0.073707 CK: CK is about **21 times more
responsive per displayed-stimulus update**. These are mean absolute realized
updates, not hypothetical changes for undisplayed stimuli.

S5-prefix-fitted alpha_CK has mean 0.178073, median 0.128624, SD 0.175985,
quartiles 0.082468/0.204399, and range 0.001889–1. There are no alpha=0 cases and
two alpha=1 cases. The finite half-life distribution has mean 10.9119 relevant
presentations, median 5.0344, quartiles 3.0314/8.0535, 5th–95th percentiles
0.9926/33.6813, and range 0.2355–366.5876 under the stated boundary convention.
The long upper tail makes the mean unrepresentative of many fitted trajectories.
These are model-dynamic descriptions, not individually validated memory estimates.

The exploratory associations do not support a simple “shorter CK memory is
better” story during training: alpha_CK versus CK's advantage over cumulative
has Pearson r=−0.100 and Spearman rho=−0.142; finite half-life has r=+0.304 and
rho=+0.143. The weaker rank correlations and long half-life tail warrant caution.
Both fitted quantities depend on the same participant data; there is no causal
or individual psychological inference here.

### Calibration does not show growing late-training overconfidence

Cumulative pooled ECE improves from 0.035603 (S2) to 0.030032 (S3), 0.027066
(S4), and 0.016075 (S5). Its participant-mean ECE also declines, from 0.090341 to
0.067011. S5 pooled ECE is 0.019981 for CK and 0.026968 for RL. The different
pooled versus participant ECE rankings illustrate why aggregation matters;
binning and finite per-person sample sizes also affect ECE.

S5 participant-mean Brier scores are 0.122552 RL, 0.109876 CK, and 0.109626
cumulative. Cumulative confidence is below observed favored-option accuracy in
nine of the ten pooled bins; the exception is the 0.65–0.70 bin (overconfidence
0.008345). This is not the pronounced frequent-member overprediction suggested
by the *previously documented final-test motivation*. That earlier test is not
recomputed or used to select a new model here.

Cumulative's mean absolute S5 history logit is 0.905874 versus 0.838660 for CK;
reward-logit magnitudes are 1.200349 and 1.221998, respectively. History exceeds
reward magnitude on 31.25% of cumulative trials and 29.56% of CK trials. A somewhat
larger history contribution is therefore observed, but does not imply
miscalibration: the combined model predicts reinforced training well.

### What the equal-pair diagnostics can and cannot say

There are **zero observed equal-value comparisons in S2, S3, S4, or S5**, and
zero larger-gap training trials. All 33,321 S5 observations have supplied reward
gap 1. The requested equal-pair observed rate, conditional loss, calibration
error, observed deviation from .5, and larger-gap model comparison are therefore
**not estimable**, rather than merely nonsignificant.

Unobserved state contrasts show elevated and increasing frequent-minus-rare
history logits for **both** history models. S5 cumulative logits for 3−2, 4−5,
and 7−6 average 0.8251, 0.9031, and 0.8693; CK averages 0.7586, 0.8217, and
0.7759. Their hypothetical frequent-member probabilities are about 68.9–70.5%
for cumulative and 67.4–68.6% for CK. These are not observed choices and cannot
show that preference has stopped growing, reversed, or become overconfident.
No probes are inserted into the sequence and no probe actions update history.

### Large prediction disagreements are uncommon and have mixed implications

S5 absolute probability differences reach 0.10 on 2,039/33,321 trials (6.12%),
and 0.20 on 269 trials (0.81%). At the 0.10 threshold, cumulative assigns higher
probability to the observed choice on 1,126 trials and CK on 913. At 0.20, the
counts are 150 cumulative and 119 CK. Despite winning fewer such trials, CK has
positive summed log-likelihood gain within these selected subsets (0.013012 and
0.007040 nats per selected trial), because magnitudes matter as well as counts.
This subset result does not reverse the full-S5 comparison. Selection is based
only on the predeclared probability-distance thresholds, not on which model won.

Disagreements are present throughout each tercile, rather than concentrated at a
single session position. The pair-count tables above are descriptive counts, not
normalized evidence that one pair is intrinsically harder; training pair
frequencies differ. Every disagreement still concerns reward gap 1.

## Narrow scientific answers

1. **Is the cumulative failure already visible within reinforced training?**
   No at the cohort level tested here. It predicts S5 better than RL, and CK has
   no established S5 mean advantage over cumulative. Individual exceptions exist.

2. **Is the evidence consistent with excessive persistence/insufficient forgetting?**
   It confirms increasing cumulative persistence and much faster CK adaptation.
   It does **not** establish that this persistence is excessive under reinforced
   training or causes the historical final-test deficit. The stronger claim
   requires transfer conditions not represented in this training-only validation.

3. **What aspect of CK appears useful?**
   Its finite encounter-based timescale preserves local adaptability even after
   many trials. That is a measured functional difference. It is not shown to be
   the cause of a predictive advantage here, nor isolated from CK's additional
   free update-rate parameter, different initialization, and feature definition.

4. **Does this motivate a new history representation?**
   Not yet on these training-only results. They narrow the problem toward
   transfer/generalization across conditions, rather than demonstrating a need
   to repair cumulative history during stable reinforced training. Pair novelty,
   reward-gap support, and feedback removal must not be conflated.

5. **Does anything justify planning now?**
   No. This milestone neither fitted nor identified forward planning, and a
   planning extension would not resolve the design's missing transfer comparisons.

## Exactly one recommended next modeling experiment

Run a **predeclared synthetic 2×2 transfer stress test of the unchanged CK and
cumulative models**, crossing familiar versus novel/equal-value pair sets with
feedback present versus absent. Use reciprocal known generators, fit parameters
only before transfer, and evaluate causal one-step transfer loss plus adaptation
trajectories on fresh independent agents. Fix parameter settings and evaluation
windows before simulation; do not match the already-inspected human test numbers.
This single factorial experiment can separate schedule change from feedback
removal and establish which existing memory dynamics generate a transfer gap.
It would still be a synthetic mechanism check, not proof of a human cause.
Do not add a new human history representation or planning model in that experiment.

## Verification and exact run protocol

- The initial suite passed **240 tests** before fitting. The final suite passed
  **242 tests in 45.53 seconds** (216 unchanged existing cases plus 26 new cases).
  New tests cover prefix-only optimizer inputs, future-choice invariance, the
  final-test phase/session guards, feedback-bearing validation, state carryover,
  Q and history updates, half-life boundaries, chronological terciles, reward and
  pair classification, logit identity, inclusive disagreement thresholds, and
  deterministic participant-bootstrap counting.
- The three-model E11T9A ten-start S5 smoke test took 0.92 seconds. The structural
  stability sample used r01b4b, e05w0s, s04t2z, and E11T9A, selected by shortest,
  middle, longest retained-training rank plus the previous reference participant,
  with lexicographic ID tie-breaking. No behavioral performance selected them.
- Across the 12 subset fits, 50 versus 200 starts differed by at most 1.48e-12
  deviance, below the predeclared 1e-6 tolerance; runtime was 6.84 seconds.
  Thus S2–S4 used 50 starts, while primary S5 retained 200. In the completed full
  S5 cohort, every 50-start prefix again matched 200 within 1.48e-12. This does
  not prove global optimality or certify all early-fold optima at 200 starts.
- The main run made **2,556 fits over 852 participant/folds**, not 852 independent
  people. It attempted **223,650 starts**, with **zero failed fits** and **129
  unsuccessful starts** (101 RL, 9 CK, 19 cumulative; `ABNORMAL` termination).
  Unsuccessful starts were retained in diagnostics and not accepted as optima.
  Runtime was **284.69 seconds** with four workers.
- The saved trace audit checked all **396,816 model-trial predictions**: direct
  logit-derived probabilities/losses agreed with stored results within 8.89e-16.
  Every validation Q feedback update, CK update, and history-state continuation
  passed; recomputed participant/bootstrap summaries matched the saved JSON
  exactly. Trace checksum and protected model-source hashes matched.
- **No final-test or session-6 rows were scored or fitted.** The final-test object
  is never requested from the deserialized RData mapping. Tests verify this with
  a mapping that raises on access. Phase guards also reject final-test rows when
  their session label is 5; a numeric session filter alone would be insufficient.
  Pre-training rating/pretest-choice initialization remains exactly the required
  released-code-intended convention. There are no new participant exclusions.

```sh
.venv/bin/python -m pytest -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python scripts/run_history_diagnostics.py --stability-check \
  --primary-restarts 200 --workers 4 --output docs/history_diagnostics_stability.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python scripts/run_history_diagnostics.py --primary-restarts 200 \
  --rolling-restarts 50 --workers 4 --output docs/history_diagnostics_results.json
```

Regenerate this report without loading data or refitting:

```sh
.venv/bin/python scripts/run_history_diagnostics.py --report-only \
  --output docs/history_diagnostics_results.json
```

The reviewed interpretation section is preserved by report regeneration.
No original model implementation or prior result file was modified. Raw/reference
data and the new local trial-trace archives remain Git-ignored. No commit or push.
