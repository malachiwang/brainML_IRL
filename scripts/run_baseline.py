"""Run after `python -m pip install -e '.[dev]'` from the project root."""

import argparse
from pathlib import Path

from reward_pairs.data import load_rdata, normalize_training, select_training_object, validate_training
from reward_pairs.evaluation import compare_utilities, recovery_stability
from reward_pairs.recovery import RecoveryError, recover_utilities
from reward_pairs.simulation import simulate_trajectory
from reward_pairs.trajectories import build_trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description="Reward Pairs synthetic pairwise-choice baseline (not full IRL)")
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data/raw/02_comp_mod_RP_task_data_in.RData")
    parser.add_argument("--object", dest="object_name", help="Exact R training object name; default prefers data_RP_training_clean")
    parser.add_argument("--participant", help="Default: lexicographically first supplied participant; no accuracy selection")
    parser.add_argument("--beta", type=float, default=1.0, help="Known inverse temperature, fixed in simulation and recovery")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--repeats", type=int, nargs="+", default=[1], help="Synthetic repeats of the entire state sequence, e.g. 1 5 20")
    parser.add_argument("--show-participants", action="store_true", help="Print the complete per-participant count table")
    args = parser.parse_args()
    if not 0 < args.beta < float("inf") or any(n < 1 for n in args.repeats) or any(s < 0 for s in args.seeds):
        parser.error("beta must be finite and positive; repeats positive; seeds nonnegative.")
    try:
        objects = load_rdata(args.data)
        print("R objects:", ", ".join(f"{name} {getattr(value, 'shape', '')}" for name, value in objects.items()))
        name, source = select_training_object(objects, args.object_name)
        print(f"Training object: {name}")
        if name != "data_RP_training_clean":
            print("SCHEMA NOTE: using the named analysis variant; data_RP_training_clean is not selected. No additional filtering.")
        frame = normalize_training(source)
        report = validate_training(frame)
        print(f"\nValidation: {report.row_count:,} rows; {len(report.participant_counts)} participants; "
              f"{len(report.session_counts)} participant-sessions; {report.missing_actions} missing actions.")
        print("Input already sorted by participant/session/trial:", report.input_sorted)
        print("Stimuli:", sorted(set(frame.left_stimulus) | set(frame.right_stimulus)))
        print("Actions:", frame.action.value_counts(dropna=False).to_dict())
        print("Rows by training session:", frame.session.value_counts().sort_index().to_dict())
        print("\nParticipant row/session counts:\n", report.participant_counts.describe().to_string())
        print("\nRows per participant-session:\n", report.session_counts.rows.describe().to_string())
        if args.show_participants:
            print("\nAll participant counts:\n", report.participant_counts.to_string())
        print("\nObserved unordered pairs (states themselves remain ordered):\n",
              report.pair_counts.sort_values(["stimulus_a", "stimulus_b"]).to_string(index=False))
        print("Flags (not reapplied):", report.flag_counts)
        for note in report.notes:
            print("Note:", note)

        participant = args.participant if args.participant is not None else sorted(frame.participant.unique())[0]
        trajectory = build_trajectory(frame, participant)
        print(f"\nSelected participant: {participant}; {len(trajectory)} trials; "
              f"{len({trial.session for trial in trajectory})} sessions.")
        print(f"Fixed beta={args.beta}; reference stimulus 1 = 0; first seed={args.seeds[0]}.")
        choices = simulate_trajectory(trajectory, seed=args.seeds[0], beta=args.beta)
        try:
            fit = recover_utilities(choices, beta=args.beta)
            comparison, metrics = compare_utilities(fit)
            print("\nSingle-trajectory recovery:\n", comparison.to_string(float_format=lambda x: f"{x:.4f}"))
            print("Metrics:", ", ".join(f"{key}={value:.4f}" for key, value in metrics.items()))
        except RecoveryError as exc:
            print(f"Single-trajectory recovery unavailable: {exc}")

        runs = recovery_stability([trial.state for trial in trajectory], seeds=args.seeds,
                                  repeats=args.repeats, beta=args.beta)
        print("\nSeed stability (repeats are fresh simulated choices on the same state sequence):")
        print(runs.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        successful = runs.loc[runs.status == "ok"]
        print(f"Successful fits: {len(successful)}/{len(runs)}; failures remain listed above.")
        if not successful.empty:
            metrics = ["mae", "strict_order_agreement", "spearman", "pearson", "true_tie_mean_gap"]
            print("\nAcross-seed summaries (successful fits only; std undefined for one seed):")
            print(successful.groupby(["repeats", "n_trials"])[metrics].agg(["mean", "std", "count"]).to_string(float_format=lambda x: f"{x:.4f}"))
        else:
            parser.exit(1, "No finite identified recovery fits; see reported failures.\n")
        print("\nThis tests a stationary pairwise-choice baseline, not sequential IRL or human reward recovery.")
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(1, f"Baseline error: {exc}\n")


if __name__ == "__main__":
    main()
