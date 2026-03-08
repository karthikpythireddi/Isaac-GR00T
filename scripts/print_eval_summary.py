#!/usr/bin/env python3
"""
print_eval_summary.py

Print a comparison table of evaluation results across model checkpoints.

Usage:
  python scripts/print_eval_summary.py --eval_dir outputs/eval
  python scripts/print_eval_summary.py --eval_dir outputs/eval --model_order base sft dpo rwr ppo
"""

import argparse
import json
import os


def load_results(eval_dir: str) -> dict:
    """Load all eval_results.json files from eval_dir/{model}/{task}/."""
    results = {}  # {model_name: {task_name: summary}}

    if not os.path.isdir(eval_dir):
        print(f"Eval directory not found: {eval_dir}")
        return results

    for model_name in sorted(os.listdir(eval_dir)):
        model_dir = os.path.join(eval_dir, model_name)
        if not os.path.isdir(model_dir):
            continue
        results[model_name] = {}

        for task_name in sorted(os.listdir(model_dir)):
            task_dir = os.path.join(model_dir, task_name)
            results_file = os.path.join(task_dir, "eval_results.json")
            if os.path.isfile(results_file):
                with open(results_file) as f:
                    data = json.load(f)
                results[model_name][task_name] = data.get("summary", {})

    return results


def print_table(results: dict, model_order: list = None):
    """Print a formatted comparison table."""
    if not results:
        print("No evaluation results found.")
        return

    # Determine model and task ordering
    if model_order:
        models = [m for m in model_order if m in results]
    else:
        models = sorted(results.keys())

    all_tasks = set()
    for m in models:
        all_tasks.update(results[m].keys())
    tasks = sorted(all_tasks)

    if not tasks:
        print("No task results found.")
        return

    # Abbreviate long task names
    def short_name(t):
        t = t.replace("_GR1ArmsAndWaistFourierHands_Env", "")
        if len(t) > 30:
            t = t[:27] + "..."
        return t

    # Print header
    col_width = max(12, max(len(m) for m in models) + 2)
    task_width = 32
    header = f"{'Task':<{task_width}}"
    for m in models:
        header += f" | {m:^{col_width}}"
    print("=" * len(header))
    print("GR00T RLHF Evaluation Summary")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    # Print rows
    model_avgs = {m: [] for m in models}
    for task in tasks:
        row = f"{short_name(task):<{task_width}}"
        for m in models:
            if task in results.get(m, {}):
                s = results[m][task]
                rate = s.get("success_rate", 0) * 100
                n_s = s.get("n_success", 0)
                n_t = s.get("n_episodes", 0)
                row += f" | {n_s:>2}/{n_t:<2} ({rate:5.1f}%)"
                model_avgs[m].append(rate)
            else:
                row += f" | {'N/A':^{col_width}}"
        print(row)

    # Print averages
    print("-" * len(header))
    avg_row = f"{'AVERAGE':<{task_width}}"
    for m in models:
        if model_avgs[m]:
            avg = sum(model_avgs[m]) / len(model_avgs[m])
            avg_row += f" | {'':>5}{avg:5.1f}%{'':>3}"
        else:
            avg_row += f" | {'N/A':^{col_width}}"
    print(avg_row)
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", default="outputs/eval",
                        help="Directory containing eval results")
    parser.add_argument("--model_order", nargs="*", default=None,
                        help="Order of models in table (e.g., base sft dpo rwr ppo)")
    args = parser.parse_args()

    results = load_results(args.eval_dir)
    print_table(results, args.model_order)


if __name__ == "__main__":
    main()
