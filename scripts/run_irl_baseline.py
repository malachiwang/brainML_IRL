"""Compare BT and explicit soft-IRL on exactly the same synthetic choices."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reward_pairs.data import load_rdata, normalize_training, select_training_object
from reward_pairs.evaluation import compare_utilities
from reward_pairs.irl import recover_rewards
from reward_pairs.mdp import PairMDP, build_empirical_mdp, soft_value_iteration
from reward_pairs.recovery import RecoveryError, recover_utilities
from reward_pairs.simulation import DEFAULT_UTILITIES, left_probability, simulate_choices
from reward_pairs.trajectories import build_trajectory

PARAMETER_TOLERANCE = 1e-5
PROBABILITY_TOLERANCE = 1e-7
NLL_TOLERANCE = 1e-6  # total, not mean, negative log likelihood


def sensitivity_checks(mdp, choices, bt, irl, beta):
    """Fixed-weight policies and independently re-fitted parameters over gamma."""
    rows = []
    for gamma in (0.0, 0.5, 0.95, 0.99):
        fixed = soft_value_iteration(mdp, irl.utilities, beta=beta, gamma=gamma)
        row = {"gamma": gamma, "fixed_weight_policy_delta": float(np.max(
            np.abs(fixed.policy - irl.solution.policy)))}
        try:
            fitted = recover_rewards(choices, mdp, beta=beta, gamma=gamma)
            row.update(status="ok", parameter_delta_vs_bt=float(np.max(np.abs(fitted.utilities - bt.utilities))),
                       nll_delta_vs_bt=abs(fitted.negative_log_likelihood - bt.negative_log_likelihood),
                       bellman_residual=fitted.solution.bellman_residual)
        except RecoveryError as exc:
            row.update(status=str(exc))
        rows.append(row)
    print("\nGamma sensitivity, first successful matched fit (same choices and beta):")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3e}"))

    n = len(mdp.states)
    terminal = np.zeros((n, n + 1))
    terminal[:, -1] = 1
    self_loop = np.column_stack([np.eye(n), np.zeros(n)])
    uniform = np.full((n, n + 1), 1 / (n + 1))
    kernels = {"empirical": mdp.transitions[:, 0], "all_terminal": terminal,
               "self_loop": self_loop, "uniform_including_terminal": uniform}
    kernel_rows = []
    for name, p in kernels.items():
        alternative = PairMDP(mdp.states, np.repeat(p[:, None, :], 2, axis=1))
        solution = soft_value_iteration(alternative, irl.utilities, beta=beta, gamma=irl.gamma)
        expected = left_probability(mdp.states, irl.utilities, beta=beta)
        kernel_rows.append({"kernel": name, "policy_delta_vs_logistic": float(np.max(np.abs(solution.policy[:, 0] - expected))),
                            "value_delta_vs_empirical": float(np.max(np.abs(solution.values - irl.solution.values)))})
    print("\nTransition-kernel sensitivity, fixed recovered weights and configured gamma:")
    print(pd.DataFrame(kernel_rows).to_string(index=False, float_format=lambda x: f"{x:.3e}"))
    return rows, kernel_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Explicit current-pair MDP: synthetic BT vs soft-IRL")
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data/raw/02_comp_mod_RP_task_data_in.RData")
    parser.add_argument("--object", dest="object_name")
    parser.add_argument("--participant", help="Default: lexicographically first supplied participant")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--repeats", type=int, nargs="+", default=[1, 5, 20])
    parser.add_argument("--output", type=Path, help="Optional JSON report of synthetic results (no raw trials)")
    args = parser.parse_args()
    if not np.isfinite(args.beta) or args.beta <= 0 or not 0 <= args.gamma < 1:
        parser.error("Require finite beta > 0 and gamma in [0, 1).")
    if any(seed < 0 for seed in args.seeds) or any(n < 1 for n in args.repeats):
        parser.error("Seeds must be nonnegative and repeats positive.")
    try:
        name, source = select_training_object(load_rdata(args.data), args.object_name)
        frame = normalize_training(source)
        participant = args.participant if args.participant is not None else sorted(frame.participant.unique())[0]
        trajectory = build_trajectory(frame, participant)
        mdp = build_empirical_mdp(trajectory)
        states = np.array([trial.state for trial in trajectory])
        print(f"Actual R object: {name} ({len(frame):,} supplied rows; no additional filtering)")
        if name != "data_RP_training_clean":
            print("Known schema difference: using the supplied analysis table, not data_RP_training_clean.")
        print(f"Participant: {participant}; displayed states: {len(states)}; MDP decision states: {len(mdp.states)} + terminal")
        print("Transitions:", asdict(mdp.summary))
        print("Only consecutive retained trial numbers within a participant-session are linked.")
        print("Session endings and gaps terminate observed fragments; terminal value/entropy = 0.")
        print("Transition kernels P(next | state,left) == P(next | state,right):", mdp.action_independent,
              "; max row-sum error:", float(np.max(np.abs(mdp.transitions.sum(axis=2) - 1))))
        print(f"Fixed beta={args.beta}; gamma={args.gamma}; reference stimulus 1=0; no regularization.")
        truth = np.array(DEFAULT_UTILITIES) - DEFAULT_UTILITIES[0]
        print("True aligned synthetic rewards (evaluation only):", truth.tolist())
        print(f"Equivalence tolerances: parameters < {PARAMETER_TOLERANCE:g}, probabilities < {PROBABILITY_TOLERANCE:g}, total NLL < {NLL_TOLERANCE:g}.")
        rows, details = [], []
        first_success = None
        for repeat in args.repeats:
            repeated_states = np.tile(states, (repeat, 1))
            for seed in args.seeds:
                # Generate ONCE. Both optimizers receive this exact same table.
                choices = simulate_choices(repeated_states, seed=seed, beta=args.beta)
                row = {"repeats": repeat, "seed": seed, "n_trials": len(choices)}
                detail = dict(row)
                fitted = {}
                print(f"\nrepeats={repeat}, seed={seed}, synthetic trials={len(choices)}")
                for label, fit_function in (
                    ("bt", lambda: recover_utilities(choices, beta=args.beta)),
                    ("irl", lambda: recover_rewards(choices, mdp, beta=args.beta, gamma=args.gamma)),
                ):
                    try:
                        result = fit_function()
                        fitted[label] = result
                        _, metrics = compare_utilities(result)
                        row.update({f"{label}_{key}": value for key, value in metrics.items()})
                        row[f"{label}_nll"] = result.negative_log_likelihood
                        detail[f"{label}_weights"] = result.utilities.tolist()
                    except RecoveryError as exc:
                        row[f"{label}_failure"] = str(exc)
                        print(f"{label.upper()} FAILED: {exc}")
                if len(fitted) == 2:
                    bt, irl = fitted["bt"], fitted["irl"]
                    print(pd.DataFrame({"true_aligned": truth, "BT": bt.utilities, "IRL": irl.utilities}).to_string(float_format=lambda x: f"{x:.8f}"))
                    delta_w = float(np.max(np.abs(bt.utilities - irl.utilities)))
                    delta_nll = abs(bt.negative_log_likelihood - irl.negative_log_likelihood)
                    delta_p = float(np.max(np.abs(left_probability(mdp.states, bt.utilities, beta=args.beta) - irl.solution.policy[:, 0])))
                    immediate = mdp.rewards(irl.utilities)
                    q_gap_error = float(np.max(np.abs(np.diff(irl.solution.q_values, axis=1) - np.diff(immediate, axis=1))))
                    row.update(max_parameter_delta=delta_w, nll_delta=delta_nll, max_probability_delta=delta_p,
                               q_gap_error=q_gap_error, bellman_residual=irl.solution.bellman_residual,
                               status="ok" if delta_w < PARAMETER_TOLERANCE and delta_nll < NLL_TOLERANCE and delta_p < PROBABILITY_TOLERANCE else "MISMATCH")
                    print(f"BT NLL={bt.negative_log_likelihood:.12f}; IRL NLL={irl.negative_log_likelihood:.12f}")
                    print(f"Max parameter delta={delta_w:.3e}; probability delta={delta_p:.3e}; NLL delta={delta_nll:.3e}; Q-gap error={q_gap_error:.3e}; {row['status']}")
                    if first_success is None:
                        first_success = (choices, bt, irl)
                else:
                    row["status"] = "FAILED"
                rows.append(row)
                details.append({**detail, **row})

        runs = pd.DataFrame(rows)
        print("\nAll runs (both estimators' metrics; no failed fits removed):")
        print(runs.to_string(index=False, float_format=lambda x: f"{x:.8g}"))
        successful = runs.loc[runs.status == "ok"]
        print(f"\nMatching finite fits: {len(successful)}/{len(runs)}")
        if not successful.empty:
            metrics = [f"{model}_{metric}" for model in ("bt", "irl")
                       for metric in ("mae", "strict_order_agreement", "true_tie_mean_gap")]
            print("Across-seed summaries of matching fits (SD across seeds, not confidence intervals):")
            print(successful.groupby(["repeats", "n_trials"])[metrics].agg(["mean", "std", "count"]).to_string(float_format=lambda x: f"{x:.6f}"))
        gamma_rows, kernel_rows = sensitivity_checks(mdp, *first_success, args.beta) if first_success else ([], [])
        if args.output:
            report = {"training_object": name, "participant": participant, "displayed_states": len(states),
                      "mdp_states": mdp.states, "transitions": asdict(mdp.summary), "beta": args.beta,
                      "gamma": args.gamma, "reference": 1, "true_aligned": truth.tolist(),
                      "runs": details, "gamma_checks": gamma_rows, "kernel_checks": kernel_rows}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(f"Synthetic results written to {args.output}")
        print("\nThis experiment tests reward recovery under the current-pair MDP representation using synthetic demonstrations. Human reward inference and history-dependent IRL are not performed.")
        sensitivity_failed = any(r["status"] != "ok" or r.get("parameter_delta_vs_bt", float("inf")) >= PARAMETER_TOLERANCE
                                 or r.get("nll_delta_vs_bt", float("inf")) >= NLL_TOLERANCE
                                 or r["fixed_weight_policy_delta"] >= PROBABILITY_TOLERANCE for r in gamma_rows)
        sensitivity_failed |= any(r["policy_delta_vs_logistic"] >= PROBABILITY_TOLERANCE for r in kernel_rows)
        if len(successful) != len(runs) or sensitivity_failed:
            parser.exit(1, "At least one fit/equivalence check failed; see the explicit results above.\n")
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"IRL baseline error: {exc}\n")


if __name__ == "__main__":
    main()
