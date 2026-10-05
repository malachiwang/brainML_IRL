# Causal history baseline: actual synthetic results

Completed 2026-10-05. This experiment uses the real **displayed-pair scaffold**
and entirely synthetic actions. It does not fit human choices or human alpha.
See [the model explanation](history_model.md) and
[full numerical results](history_baseline_results.json).

## Reproduction and configuration

```sh
source .venv/bin/activate
python -m pytest -q
python scripts/run_history_baseline.py \
  --participant E11T9A --beta 1 \
  --alphas 0 0.5 1 --seeds 0 1 2 3 4 --rollouts 1 5 20 \
  --output docs/history_baseline_results.json
```

The loader selected `data_4_analysis_RP_training` from the existing local RData.
The scaffold has 783 retained trials: session counts **153, 157, 156, 159, 158**.
No preprocessing or exclusions changed. Raw/reference files remain ignored.
The source SHA-256 is
`037dde9d968dd8d0705891a31315930d70f475fd33a99c4743712eed89e9c5e1`.
Environment: Python 3.12.4, NumPy 2.5.3, pandas 3.0.6, SciPy 1.18.1,
rdata 1.1.0, pytest 9.1.1. No dependencies added.

The true utilities are `[1,2,2,3,3,4,4,5]`, reference-aligned to
`[0,1,1,2,2,3,3,4]`. The alpha values, seed list, and rollout counts above were
fixed before the experiment and were not changed to improve results.

Each seed spawns 20 independent RNG streams using `SeedSequence(seed).spawn(20)`.
Each agent starts with fresh history. Counts 1, 5, and 20 use nested subsets of
those agents. The same streams across alpha conditions provide common random
draws, although the choices and histories diverge. There are 100 distinct agents
per alpha condition (five batches of twenty), not 130: smaller nested subsets
are not extra agents. These streams differ from the old concatenated-exposure
baseline, so its particular seed results are not reused here.

## Exact model, likelihood, and chronology

Before each choice, `H_i=C_i/N_i-0.5` for previously presented stimuli and zero
for unseen ones. `C_i` counts previous synthetic selections and `N_i` previous
retained trials displaying i. Then:

```text
P(left_t) = sigmoid(beta * [U_L-U_R + alpha*(H_L(t)-H_R(t))])
```

Only after sampling/scoring the action are presentation counts incremented for
both stimuli and the chosen count incremented for the selected stimulus.
History carries across all sessions and gaps in this primary experiment.
Missing trials cause no updates. All-session recovery uses all retained trials.
For predictive evaluation, separate parameters are fitted to sessions 1–4
(625 trials/agent); session 5 contributes 158 held-out responses/agent.
The predictor retains the observed training history and updates it with each
earlier held-out response, never the current or future response. It does not
refit parameters on session 5. This is conditional one-step prediction.

Both models receive the same synthetic choices. The static model uses the
existing BT estimator, alpha fixed to zero. The history model jointly fits
seven free utilities and a signed, unconstrained alpha with `w_1=0`. Its
histories are reconstructed from actions, not taken from supplied feature
columns or recomputed from proposed parameters. Beta is fixed at 1.

The objective is unregularized mean sequential negative log likelihood, using
stable `logaddexp` calculations, an analytic gradient, zero initialization, and
BFGS (`gtol=1e-8`, maximum 1,000 iterations). Reported NLL is total NLL. No true
alpha/utility, equality information, reward column, or evaluation metric enters
optimization. No constraints or priors were added after inspecting results.

## Recovery across five seed batches

Means and sample SDs describe five estimates at each configuration; they are
not confidence intervals. Errors compare all-session fits with hidden truth.

| True alpha | Independent rollouts per fit | Recovered alpha, mean ± SD | Mean absolute alpha error | History utility MAE | Static utility MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | −1.032397 ± 0.895378 | 1.032397 | 0.306157 | 0.206994 |
| 0 | 5 | −0.155970 ± 0.295501 | 0.212935 | 0.120555 | 0.080095 |
| 0 | 20 | −0.068829 ± 0.150563 | 0.137806 | 0.054926 | 0.041799 |
| 0.5 | 1 | −0.433171 ± 0.681083 | 0.933171 | 0.312326 | 0.249297 |
| 0.5 | 5 | 0.411662 ± 0.356973 | 0.202914 | 0.110267 | 0.167028 |
| 0.5 | 20 | 0.413408 ± 0.121599 | 0.131046 | 0.053637 | 0.138104 |
| 1 | 1 | 0.116498 ± 0.853232 | 0.962457 | 0.345485 | 0.368110 |
| 1 | 5 | 0.838776 ± 0.305409 | 0.255340 | 0.141629 | 0.328032 |
| 1 | 20 | 0.995470 ± 0.085052 | 0.071526 | 0.054270 | 0.286562 |

Both models' strict ordering of the 25 truly unequal utility pairs was correct
in every full-session fit. That does not imply accurate alpha or exact recovery
of tied utilities. Estimated vectors, correlations, true-tie gaps, and all
training NLLs are in JSON and printed by the CLI.

The 20-rollout alpha estimates, in seed order 0–4, were:

| True alpha | Five recovered alpha values |
| --- | --- |
| 0 | 0.172442, −0.057414, −0.191532, −0.067375, −0.200269 |
| 0.5 | 0.611136, 0.450272, 0.346046, 0.320593, 0.338993 |
| 1 | 1.036844, 0.889471, 1.048115, 0.920388, 1.082529 |

For alpha=1 with 20 agents, mean recovered base utilities across seeds were:

```text
truth:   [0, 1,        1,        2,        2,        3,        3,        4       ]
history: [0, 1.035721, 1.024125, 2.025436, 2.024383, 2.999653, 3.023675, 3.998249]
static:  [0, 1.175560, 1.505041, 2.444610, 2.118390, 3.085973, 3.417938, 4.524284]
```

The static model can absorb persistent history bonuses into its estimated
utilities. Its ability to predict does not imply recovery of the underlying
base reward vector.

## Negative control: an important small-sample limitation

Simulation with alpha=0 matches the existing Fake Human's probabilities and
choices exactly for identical RNG draws. It still develops unequal empirical
selection rates, but those rates have no causal effect on its decisions.

Estimation is less reassuring with **one agent**: all five zero-truth estimates
were negative, ranging from −2.480201 to −0.248573. Thus the desired statement
“the richer model never invents a strong effect” is **not supported at one
rollout**. Wrong-sign estimates also occur under positive truth. We report
this rather than imposing a positive-alpha bound or tuning the true alphas.

Across more independent agents, both alpha error and dispersion decrease in
the zero control. At 20 agents its mean is −0.068829 with SD 0.150563, and
held-out advantage is essentially zero. These five batches alone do not prove
unbiased estimation or calibrate a hypothesis test. The static model has lower
utility MAE throughout the zero condition, as expected when the extra parameter
is unnecessary and adds estimation variance.

## Held-out session-5 prediction

Natural-log loss in nats per response, averaged across the five seed batches.
Gain = static log loss minus history log loss; positive favors history.

| True alpha | Agents | Static log loss | History log loss | Gain | Seeds favoring history |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0.584230 | 0.583231 | +0.000999662 | 3/5 |
| 0 | 5 | 0.582817 | 0.582951 | −0.000133886 | 3/5 |
| 0 | 20 | 0.587564 | 0.587563 | +0.000000231 | 2/5 |
| 0.5 | 1 | 0.555097 | 0.554402 | +0.000694826 | 4/5 |
| 0.5 | 5 | 0.557343 | 0.557960 | −0.000617251 | 0/5 |
| 0.5 | 20 | 0.561489 | 0.561386 | +0.000102758 | 5/5 |
| 1 | 1 | 0.532797 | 0.530841 | +0.001955658 | 5/5 |
| 1 | 5 | 0.523518 | 0.523749 | −0.000230412 | 3/5 |
| 1 | 20 | 0.525034 | 0.524302 | +0.000731990 | 5/5 |

At 20 agents, mean total held-out NLL (3,160 responses per fit) was:

| True alpha | Static NLL | History NLL |
| --- | ---: | ---: |
| 0 | 1856.700809 | 1856.700080 |
| 0.5 | 1774.305240 | 1773.980525 |
| 1 | 1659.107645 | 1656.794558 |

History improves average prediction for both positive alphas at 20 agents,
but the gains are small, especially for alpha=0.5. At five agents it loses
on average in both positive-alpha conditions. Prediction is not monotonically
improved by more agents in these finite samples. The larger model improves
training NLL even at zero truth; that alone is not predictive evidence.

For example, at 20 agents mean full-session NLL improvements were 0.656393,
6.979578, and 52.070642 for true alpha 0, 0.5, and 1. Corresponding mean absolute
differences between the models' predicted probabilities on those observations
were 0.002323, 0.007693, and 0.021525.

## Frozen equal-value probes

Each agent's final history was frozen and queried analytically, without adding
trials or changing counters. Summary means use all 100 unique rollouts per alpha.
The direction comparator was calculated from the reward-only oracle's expected
choice rates over the scaffold: it identified 3 over 2, 4 over 5, and 7 over 6.
Every one of the 100 agents per condition had that measured history direction
for each pair. These winners were not hard-coded; a test with a different
opponent schedule verifies that the inferred direction can reverse.

| True alpha | Pair (left, right) | Final left rate | Final right rate | P(left) | P(right) |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | (2,3) | 0.383744 | 0.616786 | 0.500000 | 0.500000 |
| 0 | (4,5) | 0.613535 | 0.387917 | 0.500000 | 0.500000 |
| 0 | (6,7) | 0.384271 | 0.612021 | 0.500000 | 0.500000 |
| 0.5 | (2,3) | 0.367128 | 0.651786 | 0.464483 | 0.535517 |
| 0.5 | (4,5) | 0.641010 | 0.360104 | 0.535050 | 0.464950 |
| 0.5 | (6,7) | 0.346985 | 0.628653 | 0.464855 | 0.535145 |
| 1 | (2,3) | 0.350718 | 0.701480 | 0.413265 | 0.586735 |
| 1 | (4,5) | 0.670556 | 0.325208 | 0.585390 | 0.414610 |
| 1 | (6,7) | 0.308291 | 0.646477 | 0.416328 | 0.583672 |

Concrete alpha=1, seed=0, agent=0 example:

| Stimulus | Base U | Chosen/presented | Rate | H | Effective U |
| --- | ---: | --- | ---: | ---: | ---: |
| 2 | 2 | 68/195 | 0.348718 | −0.151282 | 1.848718 |
| 3 | 2 | 132/196 | 0.673469 | +0.173469 | 2.173469 |
| 4 | 3 | 141/198 | 0.712121 | +0.212121 | 3.212121 |
| 5 | 3 | 69/192 | 0.359375 | −0.140625 | 2.859375 |
| 6 | 4 | 63/199 | 0.316583 | −0.183417 | 3.816583 |
| 7 | 4 | 124/193 | 0.642487 | +0.142487 | 4.142487 |

Its preferences are P(3 over 2)=0.580482, P(4 over 5)=0.587283, and
P(7 over 6)=0.580762. Equal base rewards cancel; the different preexisting
choice rates explain these probabilities exactly. JSON includes this detail
for every unique rollout, plus predictions from both fitted models trained on
twenty rollouts. These analytic probes are not observed human test choices and
are not, by themselves, proof that static and history explanations can be
distinguished: static fitted utilities can also absorb some final preferences.

Mean absolute error of fitted probe probabilities against the known generative
probe probabilities, using the 20-rollout fits and all 100 agents per condition:

| True alpha | Static probe probability MAE | History probe probability MAE |
| --- | ---: | ---: |
| 0 | 0.011458 | 0.011641 |
| 0.5 | 0.013450 | 0.012803 |
| 1 | 0.016650 | 0.011885 |

## Identifiability, failures, and tests

All **45 configurations / 180 fits** succeeded: all-session and training-session
fits for both static and history models. All history designs had full column
rank and passed a full-design separation check. No priors, penalties, parameter
bounds, exclusions, or numerical fallback were introduced.

Observed-information condition numbers over all 90 history fits ranged from
**39.10 to 132.90**. The minimum smallest eigenvalue was **0.51645**. No fit
crossed the prespecified severe-conditioning reporting threshold of `1e8`.
These diagnostics indicate finite locally identified fits, not high precision
from one agent. As rates stabilize, their effects can resemble fixed utility
offsets; independent histories and early-trial variation help distinguish them.

The suite has **88 passing tests**, including all 65 prior tests. New tests
cover pre-action timing, current/future-response leakage, both stimulus counters,
orientation, session carry/reset, fresh independent histories, alpha-zero exact
equivalence, positive/negative effects, signed-alpha recovery, deficient designs,
separation, optimizer failure, and mean held-out behavior. Changing all session-5
responses leaves the sessions-1–4 fit and earlier probabilities unchanged.
The first altered response also leaves its own probability unchanged. Controlled
tests recover negative alpha without sign restrictions and verify that frozen
probes do not update history.

## Interpretation and next step

The mechanism behaves as defined: the real opponent schedule induces unequal
choice rates among equally rewarded stimuli, and positive alpha turns those
rates into later preference. With twenty independent synthetic agents, recovery
separates base utilities and history strength reasonably well, and held-out
gains appear for both positive effects. Single-agent recovery is unreliable,
and the zero control produces strong erroneous negative estimates at that scale.
This limitation must remain visible before any human analysis.

The conclusions are conditional on fixed known beta, correct running-rate
dynamics, shared parameters across artificial agents, and a matched simulator
and estimator. Retained-trial history is incomplete relative to the original
experiment. Only one scaffold participant and five seed batches were studied.
No human coefficient, validated habit mechanism, reinforcement learning of
base utility, or history-aware IRL was inferred or implemented.

Next: predeclare additional independent seed batches and scaffold participants
to measure small-sample alpha bias/dispersion and held-out gains, including
zero and negative controls, without using human actions. Establish the amount
and type of history variation needed for reliable recovery before deciding on
a formal history-augmented IRL model or real-human model comparison.
