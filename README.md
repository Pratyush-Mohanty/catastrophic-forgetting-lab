# Catastrophic Forgetting Lab

> An end-to-end experiment platform to **see catastrophic forgetting happen in a
> real language model** — and to measure how strategies like *Elastic Weight
> Consolidation (EWC)* and *Experience Replay* slow it down.

![baseline curves](https://raw.githubusercontent.com/Pratyush-Mohanty/catastrophic-forgetting-lab/master/docs/images/curves_baseline.png)

---

## Table of Contents

1. [What this project is](#what-this-project-is)
2. [The theory: catastrophic forgetting](#the-theory-catastrophic-forgetting)
3. [How we simulated a real-world scenario](#how-we-simulated-a-real-world-scenario)
4. [What we built (architecture)](#what-we-built-architecture)
5. [Results & outputs](#results--outputs)
6. [Getting started](#getting-started)
7. [Project layout](#project-layout)
8. [Extending the lab](#extending-the-lab)
9. [Full experiment report](docs/EXPERIMENT_REPORT.md) — what we did, the results, and the learning points

---

## What this project is

We built a complete application that fine-tunes a transformer language model
(BERT) sequentially on **three different NLP tasks** and measures what happens to
earlier tasks as the model learns later ones. This is the *task-incremental
continual learning* setup used in the research literature, and it is the cleanest
way to observe **catastrophic forgetting**:

- **Task 1 — IMDb** · binary sentiment classification (positive / negative)
- **Task 2 — AG News** · news topic classification (4 classes)
- **Task 3 — DBpedia** · ontology classification (14 classes)

Each task has its own label space and its own small classification head, but all
three share one encoder (the LLM backbone). When the shared weights are fine-tuned
for task 3, the information needed for task 1 decays.

The app ships with an **interactive Streamlit dashboard** (run experiments, watch
the accuracy curves and heatmaps update live), a **CLI**, and pre-computed
**experiment results** so you can explore the phenomenon immediately.

---

## The theory: catastrophic forgetting

Catastrophic forgetting (McCloskey & Cohen, 1989) is the tendency of a neural
network to **destroy previously learned knowledge when trained on new data**.
Unlike graceful degradation, the drop in performance on old tasks is sudden and
severe.

**Why it happens.** Gradient descent updates *all* weights to fit the current
batch. The features that solved task A are not marked as "precious" — the
optimizer happily rewrites them to solve task B. With shared, distributed
representations, fixing B *conflicts* with A. French (1999) showed that the
problem grows with the degree of overlap between tasks.

**Why this matters in the real world.** LLMs are increasingly adapted to new
domains (legal, medical, customer service) and new capabilities (chat, coding,
tool-use). Naively fine-tuning a general assistant on a narrow domain can erase
its general knowledge — the "fine-tuning collapse" reported by practitioners.

### The three strategies we compare

| Strategy | Idea | How it works in code |
|---|---|---|
| **Baseline** | Do nothing special | Plain sequential fine-tuning. The control. |
| **EWC** (Kirkpatrick et al. 2017) | Protect important weights | After each task, compute the *diagonal Fisher information* `F` — a per-parameter estimate of which weights mattered for that task. During later training, add the penalty `½λ·F·(θ − θ_A)²` to the loss, anchoring the important parameters. |
| **Experience replay** (rehearsal) | Revisit the past | Keep a fixed-size memory buffer of examples from earlier tasks and mix a `replay_ratio` fraction of them into each new-task batch, with the correct task head for each example. |

---

## How we simulated a real-world scenario

A realistic deployment story: **a company fine-tunes a general-purpose language
model for three successive specialized products.**

1. **Product A — sentiment analytics.** The model is fine-tuned on IMDb reviews
   to judge product/movie sentiment. It reaches ~65% on the held-out sentiment set.
2. **Product B — news categorization.** The same model is fine-tuned on AG News
   to tag articles into topics. Because the shared weights now move toward
   "topic-detection", the sentiment head starts to degrade.
3. **Product C — knowledge-base entity classification.** The model is fine-tuned
   on DBpedia (14 ontology classes). By this point the sentiment and topic
   features have been substantially overwritten.

After every training phase we evaluate **all three tasks**. The "forgetting"
numbers quantify how much of each earlier product's capability was lost — exactly
the scenario a real MLOps team faces when models are updated in sequence.

### What the numbers say

Final accuracy matrix (rows = training phase, columns = evaluated task):

| Trained on ↓ | IMDb | AG News | DBpedia |
|---|---|---|---|
| IMDb | **0.648** | 0.270 | 0.012 |
| AG News | 0.628 | **0.805** | 0.050 |
| DBpedia | 0.600 | 0.810 | **0.905** |

- **IMDb** fell from a peak of **0.648 → 0.600** after the other two tasks were
  learned — a **retention of 0.93**. The model genuinely lost sentiment ability.
- **AG News** held up better (retention 0.98); its features are closer to those
  learned last.
- **DBpedia**, being newest, is untouched (retention 1.00).

This is *exactly* the shape of catastrophic forgetting: older tasks suffer most.

---

## Results & outputs

All runs share identical settings (model `bert_uncased_L-4_H-256_A-4`, 1000 train
samples / 400 eval per task, 2 epochs, batch 32, LR 1e-4, CPU).

### Learning curves — the "aha" plot

Accuracy on **every task** as training progresses. Watch each line drop the moment
the *next* task starts training.

**Baseline** — the oldest task's accuracy erodes with each new phase:

![baseline](https://raw.githubusercontent.com/Pratyush-Mohanty/catastrophic-forgetting-lab/master/docs/images/curves_baseline.png)

**EWC** — the Fisher penalty keeps the shared weights anchored; old tasks decay
much less:

![ewc](https://raw.githubusercontent.com/Pratyush-Mohanty/catastrophic-forgetting-lab/master/docs/images/curves_ewc.png)

**Replay** — re-training on remembered examples holds every task essentially flat:

![replay](https://raw.githubusercontent.com/Pratyush-Mohanty/catastrophic-forgetting-lab/master/docs/images/curves_replay.png)

### Accuracy matrices

Rows = what we just trained on; columns = what we evaluated. The off-diagonal
drops in the baseline matrix are the catastrophe.

![confusion matrices](https://raw.githubusercontent.com/Pratyush-Mohanty/catastrophic-forgetting-lab/master/docs/images/confusion_matrices.png)

### Retention comparison

Retention = final accuracy ÷ peak accuracy. **1.00 = nothing forgotten.**

![retention comparison](https://raw.githubusercontent.com/Pratyush-Mohanty/catastrophic-forgetting-lab/master/docs/images/retention_comparison.png)

![retention table](https://raw.githubusercontent.com/Pratyush-Mohanty/catastrophic-forgetting-lab/master/docs/images/retention_table.png)

| Strategy | IMDb retention | AG News retention | DBpedia retention |
|---|---|---|---|
| Baseline | 0.93 | 0.98 | 1.00 |
| EWC | 0.97 | 0.98 | 1.00 |
| Replay | 1.00 | 1.00 | 1.00 |

> Takeaway: replay is the strongest defense (it literally revisits old data),
> EWC provides solid protection at zero extra data cost, and doing nothing leaves
> you losing ~7% of your first task's capability — more on longer task sequences
> or with longer training.

### Task-1 accuracy over training, all strategies

The first task's accuracy as later tasks are trained — the baseline curve visibly
slopes downward where EWC and replay hold it up:

![task1 across strategies](https://raw.githubusercontent.com/Pratyush-Mohanty/catastrophic-forgetting-lab/master/docs/images/task1_across_strategies.png)

---

## What we built (architecture)

![architecture](https://raw.githubusercontent.com/Pratyush-Mohanty/catastrophic-forgetting-lab/master/docs/images/architecture.png)

The model is a shared BERT encoder with **one linear classification head per
task**. Training phase N updates the shared weights *and* head N. Because every
task shares the same backbone, the gradient updates for later tasks overwrite the
features that earlier heads depend on — that interference is catastrophic
forgetting.

```
catastrophic-forgetting-lab/
├── cf_lab/                      # core Python package
│   ├── config.py                # hardware detection, tasks, hyperparameters
│   ├── data.py                  # dataset loading + tokenization (label-sorted shuffling)
│   ├── model.py                 # shared encoder + per-task heads, Fisher computation
│   ├── training.py              # fine-tuning loop, EWC penalty, replay buffer
│   ├── metrics.py               # retention / forgetting / accuracy matrices
│   ├── experiment.py            # the IMDb → AG News → DBpedia orchestration
│   ├── plotting.py              # learning-curve & heatmap plotting
│   ├── results.py               # JSON persistence + reloading
│   └── run.py                   # CLI entry point
├── app/dashboard.py             # Streamlit interactive dashboard
├── experiments/                 # saved JSON results (already populated)
├── docs/images/                 # README figures
└── generate_images.py           # regenerates the README figures
└── architecture_diagram.py      # regenerates the architecture diagram
```

**Key design decisions (and the bugs we hit while validating):**

- **Per-task heads, shared encoder.** Distinct label spaces require a head per task.
  During phase B training, batches are routed to head B via a `head_index`; replay
  batches can contain examples from several tasks, so each row carries its own
  `head_index` and the loss uses `-inf`-padded logits so smaller tasks ignore the
  extra columns.
- **Label-sorted datasets.** `stanfordnlp/imdb` is *sorted by label* on HuggingFace;
  blindly taking the first N examples yields a single-class subset (our first runs
  "learned" to predict 0 forever). We shuffle before subsampling.
- **Correct Fisher.** EWC's Fisher needs *per-sample* gradients (the square of a sum
  is not the sum of squares) computed *outside* `torch.no_grad()` — both easy to get
  wrong, both caught during validation.
- **Same-domain tasks hide the effect.** Our first design used three sentiment
  tasks; learning task B *helped* task A (positive transfer, no forgetting).
  The distinct-task multi-head design is what actually exposes catastrophic
  forgetting.

---

## Getting started

### Prerequisites

- Python 3.10+ (installed) · Git (installed)
- ~3 GB of free disk for the venv + model + dataset caches

### 1. Set up (Windows)

```powershell
cd catastrophic-forgetting-lab
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> We install the **CPU build** of PyTorch on this machine because the MX150 GPU
> carries an old CUDA 11.1 driver that modern wheels reject. Everything runs on
> CPU and the app auto-detects hardware. On a CUDA-capable GPU you can instead
> `pip install torch` normally and pick `distilbert-base-uncased` in the dashboard.

### 2. Interactive dashboard (recommended)

```powershell
streamlit run app/dashboard.py
```

- **Experiment tab** — configure model / strategy / epochs / samples in the sidebar,
  click **Run experiment**, watch live accuracy curves and the confusion heatmap.
- **Compare strategies tab** — overlays retention and task-1 accuracy for baseline,
  EWC, and replay (pre-computed runs load automatically from `experiments/`).
- **Theory tab** — the background reading.
- **Saved results tab** — inspect the raw JSON of any run.

### 3. Command line

```powershell
python -m cf_lab.run --mitigation baseline --plot
python -m cf_lab.run --mitigation ewc --epochs 2 --samples 1000
python -m cf_lab.run --mitigation replay --epochs 2 --samples 1000 --plot
```

Results are saved under `experiments/` as JSON, and `--plot` writes the curve +
heatmap to PNG.

### 4. Regenerate the README figures

```powershell
python generate_images.py      # results figures (requires saved experiments)
python architecture_diagram.py # architecture diagram
```

---

## Project layout

```
cf_lab/
├── config.py      # hardware detection, task & hyperparameter config
├── data.py        # dataset loading + tokenization (shuffled subsampling)
├── model.py       # model loading, Fisher information computation
├── training.py    # fine-tuning loop, EWC penalty, replay buffer
├── metrics.py     # retention / forgetting / accuracy matrices
├── experiment.py  # the A → B → C orchestration
├── plotting.py    # learning-curve and heatmap plotting
├── results.py     # JSON persistence
└── run.py         # CLI entry point
app/dashboard.py   # Streamlit UI
experiments/       # saved JSON results
docs/images/       # README figures
```

---

## Extending the lab

- **Add a task** — append a `TaskSpec` (name, dataset, config, columns, `num_labels`)
  to `ExperimentConfig.tasks` in `cf_lab/config.py`.
- **Add a mitigation** — implement it in `cf_lab/training.py` (e.g. `kind="lwf"` for
  Learning Without Forgetting, or `kind="lora"` for low-rank adapters) and wire it
  into `fine_tune_phase`.
- **Load results** — `cf_lab.results.load_result(path)` returns an
  `ExperimentResult` with `.summary()`, `.confusion()`, and `.step_logs`.

---

## References

- McCloskey, M. & Cohen, N. J. (1989). *Catastrophic interference in connectionist
  networks: The sequential learning problem.* Psychology of Learning and Motivation.
- French, R. M. (1999). *Catastrophic forgetting in connectionist networks.* Trends
  in Cognitive Sciences.
- Kirkpatrick, J. et al. (2017). *Overcoming catastrophic forgetting in neural
  networks.* PNAS / Nature.

---

## License

MIT — use it, learn from it, extend it.
