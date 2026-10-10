"""Replicate the unchanged causal-history model on structurally selected scaffolds."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

from reward_pairs.data import load_rdata, select_training_object
from reward_pairs.history_replication import (
    RNG_RULE, SELECTION_RULE, experiment_grid, run_replication, scaffold_summary,
    scaffold_trajectory, select_scaffolds, structural_training,
)
from reward_pairs.simulation import DEFAULT_UTILITIES


def print_summaries(result):
    print("\nCondition summaries: means across scaffold x batch; SD is descriptive, not an SE.")
    print("phase alpha n alpha_hat SD alpha_MAE U_history U_static test_static test_history gain win_rate failed_fits")
    for row in result["summaries"]["conditions"]:
        m = row["metrics"]
        names = ("all_history.alpha", "all_history.alpha_absolute_error", "all_history.mae", "all_static.mae",
                 "train_static.test_log_loss", "train_history.test_log_loss", "heldout_log_loss_gain")
        means = [m[k]["mean"] for k in names]
        print(row["experiment"], row["alpha_true"], row["rollouts"], means[0], m[names[0]]["sd"],
              *means[1:], row["history_win_rate"], row["fit_failures"])
        print("  alpha distribution:", m["all_history.alpha"])
        print("  held-out gain distribution:", m["heldout_log_loss_gain"])
        print("  zero-control thresholds:", row["zero_control"])
    print("\nScaffold-specific recovery and conditioning, every condition:")
    for row in result["summaries"]["scaffold_conditions"]:
        m = row["metrics"]
        print(row["participant"], row["experiment"], row["alpha_true"], row["rollouts"],
              {k: m[k]["mean"] for k in ("all_history.alpha", "all_history.alpha_absolute_error",
                  "all_history.mae", "all_static.mae", "heldout_log_loss_gain", "all_history.information_condition")},
              "wins", row["history_wins"], "/", row["heldout_pairs"], "failed fits", row["fit_failures"])
    print("\nEqual-value probes (unique agents only):")
    for row in result["probe_summaries"]:
        print(row)
    totals = result["summaries"]["totals"]
    print(f"\nConfigurations={totals['configurations']}; fits attempted={totals['fits_attempted']}; "
          f"fit failures={totals['fit_failures']}; failed configurations={totals['failed_configurations']}")
    for run in result["runs"]:
        for name, fit in run["fits"].items():
            if fit["status"] != "ok":
                print("FAILED:", {k: run[k] for k in ("participant", "alpha_true", "rollouts", "seed_batch")}, name, fit["status"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data", type=Path, default=root / "data/raw/02_comp_mod_RP_task_data_in.RData")
    parser.add_argument("--beta", type=float, default=1.)
    parser.add_argument("--n-scaffolds", type=int, default=12)
    parser.add_argument("--seed-batches", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--alphas", type=float, nargs="+", default=[0., .5, 1.])
    parser.add_argument("--rollouts", type=int, nargs="+", default=[1, 5, 20])
    parser.add_argument("--secondary-negative", action="store_true",
                        help="Predeclare alpha=-1 at the maximum rollout count on every scaffold/batch.")
    parser.add_argument("--output", type=Path, default=root / "docs/history_replication_results.json")
    args = parser.parse_args()
    started = perf_counter()
    try:
        name, source = select_training_object(load_rdata(args.data))
        frame = structural_training(source)
        participants = select_scaffolds(frame, args.n_scaffolds)
        trajectories = {p: scaffold_trajectory(frame, p) for p in participants}
        scaffolds = [scaffold_summary(t) for t in trajectories.values()]
        grid = experiment_grid(participants, args.alphas, args.rollouts, args.seed_batches,
                               secondary_negative=args.secondary_negative)
        print("Actual R object:", name, "; available retained rows:", len(frame), flush=True)
        print("Selection:", SELECTION_RULE)
        print("Chosen scaffolds:", participants)
        for scaffold in scaffolds:
            print(scaffold)
        print("RNG:", RNG_RULE)
        print(f"Predeclared grid: {len(grid)} configurations; {4 * len(grid)} fits; beta={args.beta}", flush=True)
        print("Primary alpha:", args.alphas, "; rollouts:", args.rollouts, "; seed batches:", args.seed_batches)
        print("Secondary -1 at max rollouts:", args.secondary_negative)
        print("No human choices; fresh independent histories; carry across sessions; fit 1-4, score 5.")

        def progress(p, phase, alpha, seed, completed, seconds):
            print(f"{p} {phase} alpha={alpha:g} batch={seed}: {completed}/{len(grid)} configurations; {seconds:.1f}s", flush=True)

        result = run_replication(trajectories, grid, beta=args.beta, progress=progress)
        model_files = [root / "src/reward_pairs" / name for name in (
            "history.py", "history_simulation.py", "history_recovery.py", "history_evaluation.py", "recovery.py")]
        payload = dict(schema_version=1, completed_utc=datetime.now(timezone.utc).isoformat(),
            command=[sys.executable, *sys.argv], training_object=name,
            source_sha256=sha256(args.data.read_bytes()).hexdigest(),
            model_sha256={p.name: sha256(p.read_bytes()).hexdigest() for p in model_files},
            environment=dict(python=platform.python_version(), **{p: version(p) for p in ("numpy", "pandas", "scipy", "rdata")}),
            selection_rule=SELECTION_RULE, rng_rule=RNG_RULE, beta=args.beta, reference_stimulus=1,
            true_utilities=list(DEFAULT_UTILITIES), true_aligned=[u - DEFAULT_UTILITIES[0] for u in DEFAULT_UTILITIES],
            reset_history_each_session=False, fit_sessions=[1, 2, 3, 4], test_sessions=[5],
            primary_alphas=args.alphas, rollout_counts=args.rollouts, seed_batches=args.seed_batches,
            secondary_negative=args.secondary_negative, spurious_alpha_thresholds=[.5, 1.],
            available_participants=int(frame.participant.nunique()), available_rows=len(frame),
            scaffolds=scaffolds, **result)
        print_summaries(result)
        payload["total_seconds_before_serialization"] = perf_counter() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        print(f"Saved {args.output}; experiment runtime={result['experiment_seconds']:.2f}s; "
              f"total runtime including serialization={perf_counter() - started:.2f}s")
        print("This is replication of a synthetic sequential-choice model, not human habit inference or history-aware IRL.")
        if result["summaries"]["totals"]["fit_failures"]:
            parser.exit(1, "Failed fits retained in JSON; no retries, exclusions, or regularized fallback.\n")
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"Replication error: {exc}\n")


if __name__ == "__main__":
    main()
