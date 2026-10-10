# Planning with choice history: the first sequential IRL model

The earlier history-sensitive Bob remembers past selections, but acts myopically:
he compares today's effective values, `U_i + alpha H_i`. He does not account for
how today's choice changes tomorrow's history. That model and all its existing
code remain available, unchanged.

Planning Bob adds one step to the reasoning: an action changes chosen counts,
which can change future history-dependent rewards. He evaluates those future
consequences before choosing now. This is an explicitly sequential soft-planning
model, unlike the current-pair MDP whose continuation values canceled.

## State, reward, and exact update

For a known displayed-pair schedule, use state **(t, C)**. Here t is the zero-based
position in the retained sequence, and C contains eight previous chosen counts.
The current ordered pair and eight presentation counts N(t) are deterministic
from that schedule. They need not be independently enumerated in the state.
Session/trial labels locate retained observations; there is no session coefficient
or added behavioral feature.

The unchanged pre-action history is:

```text
H_i(t) = C_i(t)/N_i(t) - 0.5, if N_i(t)>0; otherwise 0
r(t,C,a) = w_selected + alpha * H_selected(t)
transition: (t,C) -> (t+1, C + one_hot(selected))
```

Both displayed stimuli's presentation counts advance with the schedule; chosen
counts advance only for the selected stimulus, after evaluating/sampling its
action. If a stimulus occupies both sides it is presented on one trial, not two.
The next pair is externally fixed, but **the next chosen-count state is not**.
No current or future observed choice enters its own pre-choice features.

Default history and planning both cross session boundaries. Missing original
trial labels make no updates; the next step is the next retained trial. Gamma
discounts each retained step, not elapsed time or the overnight break. This is
a synthetic schedule convention, not a model of missing human experiences.
Only the end of the supplied scaffold is terminal, with value and entropy zero.

## Exact soft planning and finite lookahead

For each action, recursively construct its hypothetical successor and calculate:

```text
Q(t,C,a) = r(t,C,a) + gamma * V(t+1, C + one_hot(selected))
V(t,C)   = logsumexp_a(beta * Q(t,C,a)) / beta
log pi(a|t,C) = beta * Q(t,C,a) - logsumexp_b(beta * Q(t,C,b))
```

Transitions are deterministic conditional on an action and the known schedule;
the recursion accounts for both possible future choices through the soft value.
No observed future responses are used. Stable shifted logits prevent overflow.

| Lookahead h | Decisions included in today's plan |
| ---: | --- |
| 0 | Current decision only: the old myopic policy |
| 1 | Current decision plus one future decision |
| 3 | Current decision plus three future decisions |
| 5 | Current decision plus five future decisions |

Set V=0 after the lookahead window or scaffold end, whichever comes first.
Replan with a fresh window after each actual action. This is **receding-horizon
planning**, not full-horizon or infinite-horizon optimization. Within a plan,
future nodes use decreasing remaining depth; today's calculation does not solve
a sophisticated model of its own later replanning. On an eight-trial toy
schedule, h=7 covers the entire remaining episode and is exact full-horizon DP.

Planning assumes access to the future *displayed-pair schedule*. It does not
assert that real participants had such knowledge. Gamma defaults to 0.95 and
beta to 1; both, and h, are fixed and known during fitting.

## A hand-checkable example

Use pairs `(2,3)` then `(2,1)`, all base rewards zero, alpha=2, beta=1,
gamma=0.95, initially no history. Immediate rewards at the first trial tie.

Choose 2 now: on the second trial, H2=+0.5, so its reward is +1. Stimulus 1 is
unseen and has history/reward zero. The next value is `log(exp(1)+1)`.

Choose 3 now: next H2=−0.5, its reward is −1, and the next value is
`log(exp(-1)+1)`. Those two values differ by exactly 1. Consequently today's
Q difference is 0.95, even though its immediate reward difference is zero.
The probability of choosing 2 becomes `sigmoid(0.95)`, about 0.7211 rather
than 0.5. Replacing the next pair with `(3,1)` reverses the planning preference.

Gamma=0 or h=0 removes the continuation effect and recovers myopic Bob.
Alpha=0 also makes history irrelevant to reward; different chosen-count states
then have identical continuation values. Distinct transitions alone do not
guarantee a behavioral planning effect.

## Why not enumerate the full 783-trial MDP?

Different choice paths can merge into identical `(t,C)` states. The recursive
reference solver memoizes these states, and the tests compare memoized and
unmemoized results. Nevertheless, the number of distinct count vectors can be
enormous. The [runtime/state-count study](history_irl_results.md) includes capped
prefix enumeration and a rigorous reachable-state lower bound based on a
spanning tree of comparison pairs, rather than extrapolating `2^783` paths.

For fitting, the fast solver compiles exact hypothetical reward features into
small binary windows and performs vectorized backward Bellman backups. It caches
parameter-independent features across optimizer calls, but recalculates values
for every candidate parameter vector. This is the same model, not a discretized
history or Monte Carlo approximation. The independent memoized scalar solver
checks Q values, policies, and terminal handling. The batch solver avoids Python
recursion overhead; merging small trees is not required for tractability.

Safety guards reject h>10 and batches exceeding two million allocated decision
nodes. The likelihood splits roots into bounded chunks and groups roots by
remaining episode length. Guards raise errors, never shorten a requested horizon.
No unbounded full-scaffold tree is launched.

## What planning-policy IRL estimates

Generate choices from Planning Bob with known parameters, then hide them.
The estimator receives the schedule, chronological synthetic actions, explicit
transition rule, beta, gamma, and h. It reconstructs C before each observation,
solves the candidate-parameter planning problem, and minimizes:

```text
-sum_t log pi_parameters(observed action_t | t, observed choices before t)
```

Fit `w2,...,w8,alpha`, fixing w1=0; alpha can be positive or negative. No equality
constraints, reward observations, truth-based initialization, regularization,
or joint beta/scale estimation are used. The answer key `[0,1,1,2,2,3,3,4]`
and true alpha are used only for generation and evaluation.

An analytic derivative accompanies each Bellman backup:

```text
dQ = immediate_reward_features + gamma * dV_child
dV = sum_a pi(a) * dQ(a)
d log pi(a) = beta * (dQ(a) - dV)
```

Counts are fixed observed-prefix or hypothetical-branch statistics during
optimization, not resimulated at each proposed parameter. Tests independently
compare these derivatives with finite differences. BFGS starts at eight zeros
and minimizes mean NLL with gradient tolerance 1e-8 and at most 1,000 iterations;
reported NLL is total. Failures/nonfinite results are explicit. A central
difference of the analytic gradient (step 1e-4) estimates the observed-information
Hessian at the optimum and flags singular/nonpositive cases.

Unlike myopic logistic regression, the planning-policy objective is not generally
guaranteed convex. Successful BFGS and positive local information are **not a
certificate of a global MLE or universal identifiability**. We do not reuse the
old linear-design separation test as if it applied to this nonlinear model.
This milestone is finite-lookahead soft-optimal **policy-likelihood IRL**,
not a separate trajectory-normalized MaxEnt or feature-occupancy algorithm.

## Prediction, mismatch, and equal-value probes

Matched recovery checks the planning generator against the same planning family.
Separate fits use sessions 1–4, then score session 5 one step ahead. History
carries from training and updates after each scored synthetic response. Planning
may see session-5 displayed pairs when looking forward from late session 4, but
not session-5 actions. Training parameters never use held-out response labels.

Mismatch diagnostics fit myopic data with planning and planning data with the
old myopic estimator. They ask how assumptions distort recovered parameters and
prediction, not which family describes humans.

For intermediate equal-value probes, replace the current pair with `(2,3)`,
`(4,5)`, or `(6,7)`, initialize with the observed pre-trial counts, then use the
actual subsequent displayed pairs within the window. Counterfactual presentation
and chosen counts update under the substituted probe. Report the immediate
history reward gap and the additional discounted continuation gap separately.
At final history, append a single terminal probe with **no invented future
schedule**; planning and myopic probabilities must then coincide.

This model is an explicit synthetic planning mechanism, not a validated habit
theory, human reward inference, or evidence that humans need to plan this way.
See [the measured effects and recovery results](history_irl_results.md).
