"""Generate all figures for the README from the saved experiment results."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from cf_lab.results import list_results, load_result
from cf_lab.plotting import task_accuracy_curves, confusion_heatmap

OUT = Path("docs/images")
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

results = {r.config.mitigation.kind: r for r in map(load_result, list_results())}
names = ["baseline", "ewc", "replay"]
for n in names:
    assert n in results, f"missing result for {n}"

# 1) Per-strategy learning curves
for n in names:
    fig, ax = plt.subplots(figsize=(9, 5))
    task_accuracy_curves(results[n], ax=ax)
    fig.tight_layout()
    fig.savefig(OUT / f"curves_{n}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# 2) Confusion matrices side by side
fig, axes = plt.subplots(1, 3, figsize=(21, 5.2))
for ax, n in zip(axes, names):
    confusion_heatmap(results[n], ax=ax)
    ax.set_title(f"{n.upper()} — accuracy matrix")
fig.tight_layout()
fig.savefig(OUT / "confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 3) Retention / forgetting comparison
summaries = {n: results[n].summary() for n in names}
tasks = results["baseline"].task_names
x = np.arange(len(tasks))
width = 0.25
fig, ax = plt.subplots(figsize=(10, 5.5))
colors = {"baseline": "#d64541", "ewc": "#e67e22", "replay": "#27ae60"}
for i, n in enumerate(names):
    s = summaries[n]
    last = s[s["phase"] == s["phase"].iloc[-1]]
    retention = [last[last["task"] == t]["retention"].iloc[0] for t in tasks]
    bars = ax.bar(x + (i - 1) * width, retention, width, label=f"{n}", color=colors[n])
    ax.bar_label(bars, fmt="%.2f", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(tasks)
ax.set_ylabel("Retention (final acc / peak acc)")
ax.set_ylim(0, 1.1)
ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="no forgetting")
ax.set_title("Retention on each task after all training (higher = better)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "retention_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 4) Task-1 accuracy over steps across strategies
fig, ax = plt.subplots(figsize=(10, 5))
task = results["baseline"].task_names[0]
for n in names:
    r = results[n]
    steps = [l.step for l in r.step_logs if task in l.task_accs]
    accs = [l.task_accs[task] for l in r.step_logs if task in l.task_accs]
    ax.plot(steps, accs, marker="o", markersize=4, label=f"{n}", color=colors[n])
ax.set_xlabel("Training step")
ax.set_ylabel(f"Accuracy on {task}")
ax.set_ylim(0, 1.05)
ax.axvline(x=results["baseline"].step_logs[-1].step // 3, color="gray", ls="--", lw=1)
ax.axvline(x=results["baseline"].step_logs[-1].step * 2 // 3, color="gray", ls="--", lw=1)
ax.legend()
ax.set_title(f"{task} accuracy across training as later tasks are learned")
fig.tight_layout()
fig.savefig(OUT / "task1_across_strategies.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("figures written to", OUT)
for p in sorted(OUT.glob("*.png")):
    print(" -", p.name)