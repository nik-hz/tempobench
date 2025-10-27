import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from .benchmark import TempobenchBenchmarker


def parse_args():
    p = argparse.ArgumentParser(description="tempobench CLI")
    p.add_argument("--dataset_path", type=str, help="Path to JSONL dataset")
    p.add_argument(
        "--task",
        type=str,
        default="causal",
        help="Task name (e.g., causality, trace_acceptance, trace_acceptance_hard)",
    )
    p.add_argument("--batch_size", type=int, default=20)
    p.add_argument("--max_workers", type=int, default=20)
    p.add_argument(
        "--log_root",
        type=str,
        default="runs/tempo_bench",
        help="Root dir for TensorBoard logs",
    )
    p.add_argument(
        "--results_dir",
        type=str,
        default="benchmark_results",
        help="Root dir to save CSV/JSONL",
    )
    p.add_argument(
        "--prebuilt",
        action="store_true",
        default=True,
        help="Dataset entries are prebuilt (prompt/label/stats)",
    )
    p.add_argument(
        "--mode", type=str, default="normal", help="Dataset subset mode (if applicable)"
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Single model id to run (overrides --models)",
    )
    p.add_argument(
        "--models", type=str, default=None, help="Comma-separated list of models to run"
    )
    p.add_argument(
        "--use_leaderboard", action="store_true", help="Run the leaderboard preset"
    )
    p.add_argument(
        "--run_all",
        action="store_true",
        help="Run all experiments: preset models ∪ leaderboard",
    )
    p.add_argument(
        "--run_all_tasks",
        action="store_true",
        help="Run all benchmark datasets (causal/trace hard+normal) in parallel",
    )
    p.add_argument(
        "--sample",
        action="store_true",
        help="Use sample datasets instead of full datasets",
    )
    return p.parse_args()


def run_single_dataset(dataset_path, task, args, models, tqdm_position=0):
    """Helper to run one dataset/task with the existing TempobenchBenchmarker loop."""
    all_results = []
    for model in models:
        # tqdm.write(f"\n=== Running benchmark for {model} on {task} ===")
        runner = TempobenchBenchmarker(
            dataset_path=dataset_path,
            task=task,
            model_id=model,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            log_dir=os.path.join(args.log_root, model.replace("/", "_")),
            mode=args.mode,
            prebuilt=args.prebuilt,
            results_dir=args.results_dir,
            tqdm_position=tqdm_position,
            console_prints=True,
        )
        df = runner.evaluate()
        df["model"] = model
        all_results.append(df)

    # save combined results per dataset
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        os.makedirs(args.results_dir, exist_ok=True)

        suffix = "_sample" if args.sample else ""
        final_csv = os.path.join(
            args.results_dir, f"final_results_all_models_{task}{suffix}.csv"
        )
        final_jsonl = os.path.join(
            args.results_dir, f"final_results_all_models_{task}{suffix}.jsonl"
        )
        # --- If file exists, append instead of overwrite ---
        if os.path.exists(final_csv):
            old_csv = pd.read_csv(final_csv)
            final_df = pd.concat([old_csv, final_df], ignore_index=True)

        if os.path.exists(final_jsonl):
            old_json = pd.read_json(final_jsonl, lines=True)
            final_df = pd.concat([old_json, final_df], ignore_index=True)

        # Save back to disk (appended, no deduplication)
        final_df.to_csv(final_csv, index=False)
        final_df.to_json(final_jsonl, orient="records", lines=True)


# tensorboard --logdir runs/tempo_bench --host 0.0.0.0 --port 6006
"""
--dataset_path data/causal-100-hard.jsonl --task causal_hard
--dataset_path data/causal-200-normak.jsonl --task causal_normal
--dataset_path data/trace-100-hard.jsonl --task trace_hard
--dataset_path data/trace-200-normal.jsonl --task trace_normal
"""


def main():
    preset_models = [
        "openai/gpt-4o-mini",
        # "openai/gpt-4o",
        # "anthropic/claude-3.5-sonnet",
        # "anthropic/claude-sonnet-4.5",
        # "qwen/qwen3-coder-plus",
        # "google/gemini-2.5-pro",
        # "deepseek/deepseek-v3.2-exp",
    ]

    leaderboard = [
        "openai/gpt-5",
        "anthropic/claude-opus-4.1",
        "deepseek/deepseek-r1-0528",
        "anthropic/claude-sonnet-4.5",
        "qwen/qwen3-coder-plus",
        "openai/gpt-4o-mini",
        "openai/gpt-4o",
    ]

    args = parse_args()

    # Build model list from flags
    selected = []
    if args.model:
        selected = [args.model]
    elif args.models:
        selected = [m.strip() for m in args.models.split(",") if m.strip()]
    elif args.run_all:
        selected = sorted(set(preset_models) | set(leaderboard))
    elif args.use_leaderboard:
        selected = leaderboard
    else:
        selected = preset_models

    print(f"Models to run: {selected}")

    if getattr(args, "run_all_tasks", False):
        dataset_tasks = [
            ("data/causal-100-hard.jsonl", "causal_hard"),
            ("data/causal-200-normal.jsonl", "causal_normal"),
            ("data/trace-100-hard.jsonl", "trace_hard"),
            ("data/trace-200-normal.jsonl", "trace_normal"),
        ]
        if getattr(args, "sample", False):
            dataset_tasks = [
                ("data/causal-hard-sample.jsonl", "causal_hard_sample"),
                ("data/causal-normal-sample.jsonl", "causal_normal_sample"),
                ("data/trace-hard-sample.jsonl", "trace_hard_sample"),
                ("data/trace-normal-sample.jsonl", "trace_normal_sample"),
            ]
        with ProcessPoolExecutor(max_workers=len(dataset_tasks)) as ex:
            futures = [
                ex.submit(run_single_dataset, d, t, args, selected, i)
                for i, (d, t) in enumerate(dataset_tasks)
            ]
            for f in futures:
                f.result()  # re-raise exceptions if any
    else:
        # Single dataset run (your existing behavior)
        run_single_dataset(args.dataset_path, args.task, args, selected)


if __name__ == "__main__":
    main()
# TODO: Wrap in try etc blocks
# TODO add sample aggregation to the dataset
