"""Plotting helpers for catastrophic-forgetting results."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .experiment import ExperimentResult


def task_accuracy_curves(result: ExperimentResult, ax=None):
    """Per-task validation accuracy vs training step (the core CF plot)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    palette = sns.color_palette("husl", len(result.task_names))
    for j, task in enumerate(result.task_names):
        steps = []
        accs = []
        for log in result.step_logs:
            if task in log.task_accs:
                steps.append(log.step)
                accs.append(log.task_accs[task])
        ax.plot(steps, accs, marker="o", markersize=4, label=f"{task} accuracy", color=palette[j])
    for i in range(len(result.task_names) - 1):
        xs = [l.step for l in result.step_logs if l.phase == result.task_names[i]]
        if xs:
            ax.axvline(x=xs[-1], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Catastrophic forgetting — {result.config.mitigation.kind}")
    ax.legend(loc="lower left", fontsize=8)
    return ax


def confusion_heatmap(result: ExperimentResult, ax=None):
    """Heatmap of accuracy: rows = training phase, cols = evaluated task."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    df = result.confusion()
    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "accuracy"},
        ax=ax,
    )
    ax.set_xlabel("Evaluated task")
    ax.set_ylabel("Training phase")
    ax.set_title("Task accuracy matrix (row = phase trained on)")
    return ax


def summary_table(result: ExperimentResult) -> pd.DataFrame:
    return result.summary()


def compare_runs(results: list[ExperimentResult], task_index: int = 0) -> pd.DataFrame:
    """Retention on one task across mitigation strategies."""
    rows = []
    for r in results:
        s = r.summary()
        seen = s[(s["phase"] == s["phase"].iloc[-1]) & (s["task"] == r.task_names[task_index])]
        if not seen.empty:
            rows.append(
                {
                    "mitigation": r.config.mitigation.kind,
                    "final_acc": seen.iloc[0]["acc_after_phase"],
                    "retention": seen.iloc[0]["retention"],
                    "forgetting": seen.iloc[0]["forgetting"],
                }
            )
    return pd.DataFrame(rows)
