# Explicit current-pair MDP: BT versus soft-IRL results

Run on 2026-10-04. All results below use synthetic choices on a real displayed
state scaffold. Human actions and reward columns are not estimator inputs.
The [method overview](method_overview.md) gives the plain-language explanation;
[irl_baseline_results.json](irl_baseline_results.json) preserves every fitted
vector, metric, and sensitivity check at full floating-point precision.

## Configuration and reproduction

```sh
source .venv/bin/activate
python -m pytest -q
python scripts/run_irl_baseline.py \
  --participant E11T9A --beta 1 --gamma 0.95 \
  --seeds 0 1 2 3 4 --repeats 1 5 20 \
  --output docs/irl_baseline_results.json
python scripts/run_baseline.py \
  --participant E11T9A --beta 1 --seeds 0 1 2 3 4 --repeats 1 5 20
```

The source file is the existing
`data/raw/02_comp_mod_RP_task_data_in.RData`, SHA-256
`037dde9d968dd8d0705891a31315930d70f475fd33a99c4743712eed89e9c5e1`.
Actual training object: `data_4_analysis_RP_training` (164,087 rows), explicitly
reported by the loader. No preprocessing or exclusions changed. The existing
`rdata` data.table-to-data.frame conversion warning remains visible.

Python 3.12.4, NumPy 2.5.3, pandas 3.0.6, SciPy 1.18.1, rdata 1.1.0,
pytest 9.1.1; no dependencies added. The complete suite has **65 passing tests**,
including all 24 original tests and 41 new test cases. Both full CLI experiments
completed successfully.

Fake Human uses utilities `[1,2,2,3,3,4,4,5]`, beta 1, and no learning or memory.
The reference-aligned answer key is `[0,1,1,2,2,3,3,4]`. Gamma is 0.95 for the
main IRL experiment. Both fits start at seven zero parameters and constrain
only `w_1=0`; no tie constraints, penalties, true-reward initialization, or
truth-based tuning are used. Choices are sampled **once per seed/repeat case**
and supplied unchanged to both estimators. Each repetition adds fresh choices
on the same sequence; it does not alter the transition estimate or link sessions.

## Exact transition construction

E11T9A contributes 783 retained displayed states across five sessions, covering
16 ordered pair states. There is one additional terminal marker with zero value
and no actions, reward, or entropy.

For each retained trial we count one destination:

1. If the next chronological row has the same participant and session and trial
   number exactly one larger, count its ordered pair as the next state.
2. Otherwise count termination. Distinguish session endings from gaps for reporting.
3. Normalize counts by the number of occurrences of each source state. Copy
   the resulting row to both actions, without consulting human or synthetic actions.

This gives **762 consecutive pair transitions, five session-ending terminal
counts, and sixteen gap-ending terminal counts**, totaling 783 outgoing counts.
There are 21 observed fragments. Relative to 800 nominal trials, 17 labels are
absent: 16 internal gaps and trial 1 of session 3. That leading missing trial
has no preceding retained row and therefore adds no terminal count. No
cross-session, cross-participant, or across-gap transition is introduced. All
783 choices remain in the likelihood. The two action kernels are exactly equal;
maximum row-sum error is `1.1102230246251565e-16`.

This is an empirical stationary model of observed fragments, not the exact
experiment scheduler. Since time/session are not state features, terminal
probabilities are pooled by displayed pair rather than triggered at a fixed
episode length. Termination before missing trials is a censoring convention.
Its particular values affect V, but not the relative action policy here.

## Mathematical formulation and independent implementation

The reward features select the chosen stimulus: `r_w(s,a)=w^T phi(s,a)`.
The IRL objective is the unweighted demonstration negative log likelihood
under a discounted entropy-regularized optimal policy:

```text
Q = r_w + gamma * P V
V(s) = logsumexp_a(beta Q(s,a)) / beta
log pi(a|s) = beta Q(s,a) - logsumexp_b(beta Q(s,b))
minimize_w -sum_t log pi_w(a_t|s_t), subject to w_1 = 0
```

`mdp.py` performs full soft value iteration. `irl.py` solves that MDP at each
objective evaluation and differentiates the fixed-point equations:

```text
P_pi = sum_a pi(a|s) P(.|s,a)
dV/dw = (I - gamma P_pi)^-1 sum_a pi(a|s) phi(s,a)
dQ/dw = phi + gamma P dV/dw
d log pi/dw = beta * (dQ/dw - E_pi[dQ/dw])
```

The terminal derivative is zero. A finite-difference test on an action-dependent
toy MDP independently verifies this gradient, including its continuation terms.
BT retains its separate logistic objective. Only input/identifiability checks
are shared. Both use BFGS on mean NLL with gradient tolerance `1e-8`; reported
likelihoods are **total** NLL. The soft planner's absolute Bellman residual
tolerance is `1e-10` with a 10,000-iteration cap. Failure is explicit; there is
no regularized fallback.

Because both actions have the same P, their continuation terms are equal:
`Q(s,left)-Q(s,right)=w_L-w_R`. Softmax therefore reduces exactly to BT's
logistic probabilities. This proof concerns the specified v0 model, not general
IRL. The implementation calculates Q and V first; the logistic identity is used
as an independent comparison, not as the IRL forward solver.

## Recovered values: seed 0, one exposure

| Stimulus | True, aligned | Bradley–Terry | Soft-IRL |
| --- | ---: | ---: | ---: |
| 1 | 0 | 0.00000000 | 0.00000000 |
| 2 | 1 | 0.81604348 | 0.81604348 |
| 3 | 1 | 0.86388680 | 0.86388680 |
| 4 | 2 | 1.76149752 | 1.76149752 |
| 5 | 2 | 1.72193630 | 1.72193630 |
| 6 | 3 | 2.47779932 | 2.47779932 |
| 7 | 3 | 2.57959174 | 2.57959174 |
| 8 | 4 | 3.46628787 | 3.46628787 |

Both total NLLs are `471.902428922866`. Maximum parameter difference is
`1.2878587085651816e-14`; maximum left-choice probability difference across all
16 decision states is `2.4424906541753444e-15`.

Both yield aligned MAE `0.28911962`, strict unequal-order agreement `1.0`,
Pearson correlation `0.99920306`, Spearman correlation `0.98198051`, and mean
estimated absolute gap for truly tied pairs `0.06306566`. The strict-order score
covers 25 unequal-value pairs and excludes the three true ties. Ties are not
forced by the optimizer.

## All seeds and repetitions

All **15/15** cases produced finite converged fits and passed equivalence checks.
Maximum differences over all cases:

| Quantity | Observed maximum | Acceptance threshold |
| --- | ---: | ---: |
| Absolute BT–IRL parameter difference | `3.4638958368304884e-14` | `< 1e-5` |
| Absolute probability difference over MDP states | `3.3584246494910985e-15` | `< 1e-7` |
| Absolute total NLL difference | `1.8189894035458565e-12` | `< 1e-6` |
| Error in Q-gap versus immediate reward gap | `6.661338147750939e-15` | Tested at `1e-12` |

Across-seed recovery metrics are indistinguishable for BT and IRL:

| Repetitions | Choices | MAE mean ± sample SD, both models | Strict order agreement, all seeds | True-tie gap mean ± sample SD, both models |
| --- | ---: | ---: | ---: | ---: |
| 1 | 783 | 0.223433 ± 0.127755 | 1.000000 | 0.297958 ± 0.380036 |
| 5 | 3,915 | 0.109586 ± 0.052536 | 1.000000 | 0.144061 ± 0.126228 |
| 20 | 15,660 | 0.049728 ± 0.022411 | 1.000000 | 0.069892 ± 0.021295 |

The SDs describe five seeds, not confidence intervals. Larger simulations share
the smaller simulation's random-choice prefix within a seed. Mean precision
improves, but individual seeds need not improve monotonically. Exact ties remain
uncertain; a high ordering score does not mean their equality was recovered.
The original pairwise CLI reproduced its earlier results after the shared
validation extraction.

## Gamma and transition-kernel sensitivity

For seed 0 / one exposure, independently refit IRL at each gamma using the same
choices and beta. Also hold recovered weights fixed and recalculate policies.

| Gamma | Max refitted parameter difference from BT | Fixed-weight policy difference from gamma 0.95 |
| --- | ---: | ---: |
| 0 | `8.88e-16` | `7.77e-16` |
| 0.5 | `3.55e-15` | `8.88e-16` |
| 0.95 | `1.29e-14` | `0` |
| 0.99 | `1.20e-14` | `1.55e-15` |

All refits converged; maximum total NLL difference from BT in this sweep was
`1.14e-13`. Thus gamma does not become identifiable from choices in this model.

At gamma 0.95, replace the empirical kernel while keeping recovered weights fixed:

| Kernel | Max probability difference from logistic | Max value difference from empirical V |
| --- | ---: | ---: |
| Empirical | `7.77e-16` | 0 |
| All terminal | `1.11e-16` | 32.14 |
| Self-loops | `9.99e-16` | 43.37 |
| Uniform over pair states and terminal | `4.44e-16` | 11.07 |

These deliberately different kernels substantially change V without changing
the policy. Additional tests check cycle transitions, several beta/gamma values,
terminal entropy handling, and an action-dependent negative control where equal
immediate rewards nevertheless give unequal action probabilities. That control
confirms the Bellman machinery actually uses future consequences.

## Interpretation, limitations, and next step

The explicit soft-IRL formulation recovers the same **relative immediate
stimulus preferences** as BT. It contributes no additional sequential reward
information because actions cannot affect future states or termination. The
equivalence is a successful result of this milestone, not evidence that IRL
outperforms BT or that generic IRL has been solved.

The model is a stationary approximation with known beta, fixed rewards, a
restricted one-hot reward class, matched simulator/estimator, and one scaffold
participant. No human actions were fitted, no habits were inferred, and no
learning/history/test-phase variables were introduced. Reward scale is not a
psychological measurement. Graph connectivity and finite-MLE checks concern
the demonstrated comparisons; unseen comparisons do not supply information.

Next: write down one minimal causal history-state variable and its action-driven
update, choose a synthetic reward/decision mechanism for which that history
matters, and test its identifiability and recovery on held-out synthetic
sequences. Do this before fitting any human rewards. Simply storing history
without making it relevant to rewards or decisions would not remove this
equivalence. That next model is intentionally not implemented in this ticket.
