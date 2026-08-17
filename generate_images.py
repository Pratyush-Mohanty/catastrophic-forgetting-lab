"""Generate polished README + Medium figures from the saved experiment results."""

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

plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.frameon": True,
        "axes.grid": True,
        "grid.alpha": 0.35,
    }
)
sns.set_theme(style="whitegrid")

results = {r.config.mitigation.kind: r for r in map(load_result, list_results())}
names = ["baseline", "ewc", "replay"]
assert all(n in results for n in names), f"missing results, have {list(results)}"
tasks = results["baseline"].task_names
colors = {"baseline": "#d64541", "ewc": "#e67e22", "replay": "#27ae60"}
phase_colors = sns.color_palette("husl", len(tasks))

# ------------------------------------------------------------------ 1) curves
for n in names:
    fig, ax = plt.subplots(figsize=(10, 6.2))
    r = results[n]
    for j, t in enumerate(tasks):
        steps = [l.step for l in r.step_logs if t in l.task_accs]
        accs = [l.task_accs[t] for l in r.step_logs if t in l.task_accs]
        ax.plot(
            steps, accs, marker="o", markersize=5, linewidth=2.2,
            label=f"{t} accuracy", color=phase_colors[j],
        )
    for i in range(len(tasks) - 1):
        xs = [l.step for l in r.step_logs if l.phase == tasks[i]]
        if xs:
            ax.axvline(x=xs[-1], color="gray", ls="--", lw=1.2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0, 1.06)
    title = {"baseline": "Baseline — sequential fine-tuning",
             "ewc": "EWC — Fisher-anchored fine-tuning",
             "replay": "Experience replay — rehearsal"}[n]
    ax.set_title(f"Task accuracy over training · {title}")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(OUT / f"curves_{n}.png", bbox_inches="tight")
    plt.close(fig)

# ------------------------------------------------------------------ 2) matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
for ax, n in zip(axes, names):
    df = results[n].confusion()
    sns.heatmap(
        df, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
        cbar=False, annot_kws={"size": 12}, linewidths=1, ax=ax,
    )
    ax.set_xlabel("Evaluated task")
    ax.set_ylabel("Training phase")
    ax.set_title(f"{n.upper()}", fontsize=14, weight="bold")
fig.suptitle("Accuracy matrix · row = phase trained on, column = task evaluated",
             fontsize=15, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "confusion_matrices.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ 3) retention
x = np.arange(len(tasks))
width = 0.26
fig, ax = plt.subplots(figsize=(9.5, 5.8))
for i, n in enumerate(names):
    s = results[n].summary()
    last = s[s["phase"] == s["phase"].iloc[-1]]
    ret = [last[last["task"] == t]["retention"].iloc[0] for t in tasks]
    bars = ax.bar(x + (i - 1) * width, ret, width, label=n, color=colors[n])
    ax.bar_label(bars, fmt="%.2f", fontsize=10, padding=2)
ax.set_xticks(x)
ax.set_xticklabels(tasks)
ax.set_ylabel("Retention  (final acc ÷ peak acc)")
ax.set_ylim(0, 1.15)
ax.axhline(1.0, color="black", ls="--", lw=1.2, label="no forgetting")
ax.set_title("Retention on each task after all three training phases")
ax.legend(loc="lower right", ncol=2)
fig.tight_layout()
fig.savefig(OUT / "retention_comparison.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ 4) task-1 across
fig, ax = plt.subplots(figsize=(10, 6.2))
task = tasks[0]
for n in names:
    r = results[n]
    steps = [l.step for l in r.step_logs if task in l.task_accs]
    accs = [l.task_accs[task] for l in r.step_logs if task in l.task_accs]
    ax.plot(steps, accs, marker="o", markersize=5, linewidth=2.2, label=n, color=colors[n])
ax.set_xlabel("Training step")
ax.set_ylabel(f"Accuracy on {task}")
ax.set_ylim(0, 1.06)
ax.set_title(f"{task} accuracy while the model learns later tasks")
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
fig.tight_layout()
fig.savefig(OUT / "task1_across_strategies.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ 5) final summary table image
s = results["baseline"].summary()
last = s[s["phase"] == s["phase"].iloc[-1]]
rows = []
for n in names:
    sm = results[n].summary()
    lm = sm[sm["phase"] == sm["phase"].iloc[-1]]
    rows.append([n] + [f"{lm[lm['task']==t]['retention'].iloc[0]:.2f}" for t in tasks])
fig, ax = plt.subplots(figsize=(9, 1.6))
ax.axis("off")
tbl = ax.table(cellText=rows, colLabels=["Strategy"] + tasks, loc="center",
               cellLoc="center", colWidths=[0.25] + [0.18] * len(tasks))
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1, 1.8)
for (rr, cc), cell in tbl.get_celld().items():
    if rr == 0:
        cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white", weight="bold")
    else:
        cell.set_facecolor(["#fdecea", "#fef5e7", "#e8f8f5"][rr - 1])
ax.set_title("Retention summary (final ÷ peak)", fontsize=13, weight="bold")
fig.tight_layout()
fig.savefig(OUT / "retention_table.png", bbox_inches="tight")
plt.close(fig)

# clean up font-test scratch if present
scratch = OUT / "_fonttest.png"
if scratch.exists():
    scratch.unlink()

print("regenerated figures in", OUT)
for p in sorted(OUT.glob("*.png")):
    print(" -", p.name, p.stat().st_size, "bytes")