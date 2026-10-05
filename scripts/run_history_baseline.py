"""Causal history recovery on independent synthetic agents; no human fitting."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reward_pairs.data import load_rdata, normalize_training, select_training_object
from reward_pairs.history_evaluation import compare_history_models, equal_value_probes, schedule_expected_rates, summary_table
from reward_pairs.history_simulation import independent_rollouts
from reward_pairs.simulation import DEFAULT_UTILITIES
from reward_pairs.trajectories import build_trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data/raw/02_comp_mod_RP_task_data_in.RData")
    parser.add_argument("--participant", default="E11T9A")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--rollouts", type=int, nargs="+", default=[1, 5, 20])
    parser.add_argument("--reset-history-each-session", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not np.isfinite(args.beta) or args.beta <= 0 or not np.isfinite(args.alphas).all():
        parser.error("Require finite beta > 0 and finite alpha values (either sign allowed).")
    if min(args.seeds) < 0 or min(args.rollouts) < 1:
        parser.error("Seeds must be nonnegative and rollout counts positive.")
    try:
        name, source = select_training_object(load_rdata(args.data))
        trajectory = build_trajectory(normalize_training(source), args.participant)
        session_counts = {s: sum(t.session == s for t in trajectory) for s in range(1, 6)}
        if not all(session_counts.values()):
            raise ValueError("The predeclared 1..4/5 comparison requires all five sessions; choose another scaffold explicitly.")
        print(f"Actual R object: {name}; scaffold={args.participant}; {len(trajectory)} retained states")
        print("Session rows:", session_counts)
        print(f"beta={args.beta}; history resets per session={args.reset_history_each_session}; fresh history per independent rollout")
        truth = np.array(DEFAULT_UTILITIES) - DEFAULT_UTILITIES[0]
        print("True aligned utilities (evaluation only):", truth.tolist())
        print("Recovery: all five sessions. Prediction: fit 1..4, score 5 causally with observed past synthetic actions.")
        # With session resets the final probe uses session-5 histories only.
        probe_scaffold = [t for t in trajectory if t.session == 5] if args.reset_history_each_session else trajectory
        expected = schedule_expected_rates(probe_scaffold, beta=args.beta)
        runs, probes = [], []
        for alpha in args.alphas:
            for seed in args.seeds:
                agents = independent_rollouts(trajectory, max(args.rollouts), seed=seed, alpha=alpha,
                                             beta=args.beta, reset_history_each_session=args.reset_history_each_session)
                for count in sorted(set(args.rollouts)):
                    choices = pd.concat([r.choices for r in agents[:count]], ignore_index=True)
                    run, fits = compare_history_models(choices, alpha_true=alpha, beta=args.beta,
                                                       reset_history_each_session=args.reset_history_each_session)
                    run.update(alpha_true=alpha, seed=seed, rollouts=count)
                    runs.append(run)
                    print(f"\nalpha_true={alpha}; seed={seed}; independent rollouts={count}; {run['status']}", flush=True)
                    for key, fit in run["fits"].items():
                        if fit["status"] != "ok":
                            print(key, "FAILED:", fit["status"])
                        else:
                            print(f"{key}: w={np.round(fit['utilities'], 6).tolist()}; alpha={fit['alpha']:.6f}; "
                                  f"NLL={fit['nll']:.6f}; utility MAE={fit['mae']:.6f}; alpha error={fit['alpha_absolute_error']:.6f}")
                            if "test_log_loss" in fit:
                                print(f"  Held-out NLL={fit['test_nll']:.6f}; mean log loss={fit['test_log_loss']:.8f}")
                    if count == max(args.rollouts):
                        batch = equal_value_probes(agents, alpha=alpha, beta=args.beta, expected_rates=expected,
                                                  static_fit=fits.get("all_static"), history_fit=fits.get("all_history"))
                        probes.extend({**p, "alpha_true": alpha, "seed": seed, "fit_rollouts": count} for p in batch)
                        print("Frozen equal-value probes, first rollout (all rollouts saved in JSON):")
                        print(pd.DataFrame(batch[:3]).to_string(index=False))
        table = summary_table(runs)
        print("\nEvery configuration (positive heldout_gain favors history):\n", table.to_string(index=False))
        numeric = [c for c in table.columns if c not in ("alpha_true", "seed", "rollouts", "status")]
        aggregate = table.loc[table.status == "ok"].groupby(["alpha_true", "rollouts"])[numeric].agg(["mean", "std", "count"])
        print("\nAcross-seed summaries, successful configurations only (sample SD, not confidence intervals):\n", aggregate.to_string())
        probe_frame = pd.DataFrame(probes)
        print("\nUnique-rollout probe summaries (nested rollout sets are not counted twice):\n",
              probe_frame.groupby(["alpha_true", "left", "right"])[["left_rate", "right_rate", "probability_left", "matches_schedule_direction"]].mean().to_string())
        failed = sum(r["status"] != "ok" for r in runs)
        print(f"\nFailed configurations: {failed}/{len(runs)}; no priors or penalties added.")
        if args.output:
            payload = dict(training_object=name, participant=args.participant, retained_states=len(trajectory),
                           session_counts=session_counts, beta=args.beta, reset_history_each_session=args.reset_history_each_session,
                           true_utilities=list(DEFAULT_UTILITIES), true_aligned=truth.tolist(),
                           rng="SeedSequence(seed).spawn(max_rollouts); fresh history per stream; nested rollout sets",
                           expected_static_rates=expected.tolist(), runs=runs, probes=probes)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            print("Saved synthetic results:", args.output)
        print("This is a synthetic sequential-choice history baseline, not human reward inference or history-aware IRL.")
        if failed:
            parser.exit(1, "Some fits failed; their configurations and errors are retained above.\n")
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(1, f"History baseline error: {exc}\n")


if __name__ == "__main__":
    main()
