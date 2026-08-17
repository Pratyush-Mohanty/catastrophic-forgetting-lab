"""Draw the architecture diagram used in README + Medium article."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.size": 12, "axes.titlesize": 14})
fig, ax = plt.subplots(figsize=(13, 6.5))
ax.axis("off")
ax.set_xlim(0, 13)
ax.set_ylim(0, 6.5)

def box(x, y, w, h, text, fc, ec="#2c3e50", fs=12, weight="normal", tc="black"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
                       fc=fc, ec=ec, lw=1.6)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            weight=weight, color=tc)
    return (x, y, w, h)

def arrow(x1, y1, x2, y2, color="#34495e", lw=2.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=22, lw=lw, color=color))

# --- tasks (inputs) ---
box(0.4, 4.6, 2.6, 1.1, "IMDb reviews\n(sentiment, 2 labels)", "#d6eaf8")
box(0.4, 2.75, 2.6, 1.1, "AG News articles\n(topic, 4 labels)", "#d6eaf8")
box(0.4, 0.9, 2.6, 1.1, "DBpedia entities\n(ontology, 14 labels)", "#d6eaf8")

# --- shared encoder ---
box(4.2, 2.15, 3.6, 3.4, "Shared BERT encoder\n(L-4 H-256 transformer)\n\n\n\n"
                          "one set of weights\nshared by all tasks", "#fef9e7", fs=12.5)
ax.text(6.0, 4.9, "catastrophic\nforgetting lives here", ha="center", va="center",
        fontsize=10.5, color="#b7950b", style="italic")

# --- heads ---
box(9.1, 4.7, 2.6, 1.0, "Head A\n2 classes", "#e8f8f5")
box(9.1, 2.85, 2.6, 1.0, "Head B\n4 classes", "#e8f8f5")
box(9.1, 1.0, 2.6, 1.0, "Head C\n14 classes", "#e8f8f5")

# --- outputs ---
box(12.05, 4.7, 0.75, 1.0, "pos/neg", "#d5f5e3", fs=10)
box(12.05, 2.85, 0.75, 1.0, "topic", "#d5f5e3", fs=10)
box(12.05, 1.0, 0.75, 1.0, "type", "#d5f5e3", fs=10)

# --- arrows: tasks -> encoder ---
arrow(3.0, 5.15, 4.2, 4.6)
arrow(3.0, 3.3, 4.2, 3.3)
arrow(3.0, 1.45, 4.2, 2.0)
# --- encoder -> heads ---
arrow(7.8, 4.4, 9.1, 5.2)
arrow(7.8, 3.3, 9.1, 3.35)
arrow(7.8, 2.2, 9.1, 1.5)
# --- heads -> outputs ---
arrow(11.7, 5.2, 12.05, 5.2)
arrow(11.7, 3.35, 12.05, 3.35)
arrow(11.7, 1.5, 12.05, 1.5)

# --- training timeline ---
box(0.4, 0.05, 12.4, 0.5, "Phase 1: fine-tune encoder + Head A on IMDb   ·   "
                          "Phase 2: fine-tune encoder + Head B on AG News   ·   "
                          "Phase 3: fine-tune encoder + Head C on DBpedia",
    "#eaf2f8", fs=10.5, weight="bold")

ax.set_title("Catastrophic Forgetting Lab — task-incremental architecture",
             fontsize=16, weight="bold", pad=12)
fig.tight_layout()
fig.savefig("docs/images/architecture.png", dpi=200, bbox_inches="tight")
print("architecture.png written")