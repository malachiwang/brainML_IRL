"""Discounted soft-optimal policy likelihood IRL for the v0 pair MDP.

Each objective call solves the explicit soft Bellman equations. No truth,
equality constraints, BT fitted parameters, or reward observations enter.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .mdp import PairMDP, SoftSolution, SoftValueError, soft_policy_jacobian, soft_value_iteration
from .recovery import RecoveryError, RecoveryResult, validated_choices


@dataclass
class IRLResult(RecoveryResult):
    gamma: float
    solution: SoftSolution
    optimizer_iterations: int


def soft_action_objective(
    weights: np.ndarray, mdp: PairMDP, counts: np.ndarray, *, beta: float, gamma: float,
) -> tuple[float, np.ndarray]:
    """Mean negative action log-likelihood and its full eight-weight gradient.

    counts[s,a] are observed demonstration counts. Transitions are fixed, so
    their likelihood contributes no reward-dependent term. Observations have
    equal likelihood weight; gamma discounts planning, not observed choices.
    """
    counts = np.asarray(counts, dtype=float)
    if counts.shape != (len(mdp.states), 2) or not np.isfinite(counts).all() or (counts < 0).any() or counts.sum() <= 0:
        raise ValueError("Provide finite nonnegative state-action counts with a positive total.")
    solution = soft_value_iteration(mdp, weights, beta=beta, gamma=gamma)
    frequencies = counts / counts.sum()
    loss = -float(np.sum(frequencies * solution.log_policy))
    jacobian = soft_policy_jacobian(mdp, solution, beta=beta, gamma=gamma)
    gradient = -np.einsum("sa,sak->k", frequencies, jacobian)
    return loss, gradient


def recover_rewards(
    choices: pd.DataFrame, mdp: PairMDP, *, beta: float = 1.0,
    gamma: float = 0.95, max_iterations: int = 1000,
) -> IRLResult:
    """Fit w_2..w_8 with w_1=0, fixed beta/gamma, and no regularization.

    Only the v0 action-independent environment is supported for inference.
    The shared comparison-graph check is sufficient in precisely that case.
    """
    if not mdp.action_independent:
        raise ValueError("v0 IRL requires action-independent transitions.")
    # Validate configuration before beginning optimization.
    soft_value_iteration(mdp, np.zeros(8), beta=beta, gamma=gamma)
    if max_iterations < 1:
        raise ValueError("Optimizer iteration limit must be positive.")
    pairs, chose_left = validated_choices(choices)
    index = {state: i for i, state in enumerate(mdp.states)}
    try:
        state_indices = np.array([index[tuple(pair)] for pair in pairs])
    except KeyError as exc:
        raise ValueError(f"Demonstration state {exc.args[0]} is absent from the MDP.") from exc
    counts = np.zeros((len(mdp.states), 2))
    np.add.at(counts, (state_indices, (~chose_left).astype(int)), 1)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        weights = np.r_[0.0, parameters]
        loss, gradient = soft_action_objective(weights, mdp, counts, beta=beta, gamma=gamma)
        return loss, gradient[1:]

    try:
        fit = minimize(objective, np.zeros(7), jac=True, method="BFGS",
                       options={"gtol": 1e-8, "maxiter": max_iterations})
        if not fit.success or not np.isfinite(fit.x).all() or not np.isfinite(fit.fun):
            raise RecoveryError(f"Soft-IRL optimization failed: {fit.message}")
        weights = np.r_[0.0, fit.x]
        solution = soft_value_iteration(mdp, weights, beta=beta, gamma=gamma)
    except (SoftValueError, np.linalg.LinAlgError) as exc:
        raise RecoveryError(f"Soft-IRL planning/derivative failed: {exc}") from exc
    return IRLResult(
        utilities=pd.Series(weights, index=pd.Index(range(1, 9), name="stimulus"), name="recovered"),
        beta=beta, reference=1, negative_log_likelihood=-float(np.sum(counts * solution.log_policy)),
        n_trials=len(pairs), gamma=gamma, solution=solution, optimizer_iterations=int(fit.nit),
    )
