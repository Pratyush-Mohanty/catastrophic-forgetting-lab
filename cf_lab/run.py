"""Command-line entrypoint: run experiments from the terminal.

Examples:
    python -m cf_lab.run                          # baseline default run
    python -m cf_lab.run --mitigation ewc
    python -m cf_lab.run --mitigation replay --epochs 2 --samples 2000
"""

from __future__ import annotations

import argparse

from .config import default_config
from .experiment import run_experiment
from .plotting import task_accuracy_curves, confusion_heatmap
from .results import save_result

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Catastrophic forgetting experiment")
    parser.add_argument("--mitigation", choices=["baseline", "ewc", "replay"], default="baseline")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None, help="train samples per task")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--plot", action="store_true", help="save plots to experiments/")
    parser.add_argument("--no-save", action="store_true", help="skip saving the JSON result")
    args = parser.parse_args()

    cfg = default_config()
    cfg.mitigation.kind = args.mitigation
    if args.epochs:
        cfg.training.epochs = args.epochs
    if args.samples:
        for task in cfg.tasks:
            task.max_train_samples = args.samples
    if args.model:
        cfg.model_name = args.model
    if args.seed:
        cfg.training.seed = args.seed

    print(f"Device: {cfg.resolve_device()} | model: {cfg.model_name}")
    print(f"Tasks: {[t.name for t in cfg.tasks]} | mitigation: {args.mitigation}")

    result = run_experiment(cfg)
    print(result.summary().to_string(index=False))

    if args.plot:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        task_accuracy_curves(result, ax=axes[0])
        confusion_heatmap(result, ax=axes[1])
        out_png = f"{cfg.output_dir}/cf_{args.mitigation}.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"Saved plot -> {out_png}")

    if not args.no_save:
        path = save_result(result)
        print(f"Saved result -> {path}")
    print(f"Runtime: {result.runtime_seconds:.1f}s")


if __name__ == "__main__":
    main()
