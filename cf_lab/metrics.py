"""Metrics for measuring catastrophic forgetting across sequential tasks."""

from __future__ import annotations

import pandas as pd


def retention(acc_after: float, acc_peak: float) -> float:
    """Fraction of the task's best accuracy still retained after later training."""
    if acc_peak <= 0:
        return 0.0
    return acc_after / acc_peak


def forgetting(acc_peak: float, acc_after: float) -> float:
    """Absolute drop in accuracy from a task's peak to its final state."""
    return acc_peak - acc_after


def summarize_runs(
    run_name: str,
    task_names: list[str],
    peak_accs: list[list[float]],
    final_accs: list[list[float]],
) -> pd.DataFrame:
    """Build a per-task summary of peak/final accuracy, retention and forgetting.

    peak_accs[i][j] is the best accuracy seen on task j during phase i.
    final_accs[i][j] is the accuracy on task j at the end of phase i.
    """
    rows = []
    for i, phase_task in enumerate(task_names):
        for j, seen_task in enumerate(task_names):
            acc = final_accs[i][j]
            best = max(p[j] for p in peak_accs[: i + 1]) if i > 0 else final_accs[0][j]
            rows.append(
                {
                    "run": run_name,
                    "phase": phase_task,
                    "task": seen_task,
                    "acc_peak": best,
                    "acc_after_phase": acc,
                    "retention": retention(acc, best),
                    "forgetting": forgetting(best, acc),
                }
            )
    return pd.DataFrame(rows)


def build_confusion(acc_matrix: list[list[float]], task_names: list[str]) -> pd.DataFrame:
    """Accuracy matrix where rows are training phases and cols are evaluated tasks."""
    return pd.DataFrame(acc_matrix, index=task_names, columns=task_names)
