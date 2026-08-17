# Catastrophic Forgetting Lab — Experiment Report

**An end-to-end account of what we did, what the experiments showed, and what we
learned** — from raw curiosity about why LLMs forget, to a working, reproducible
experiment platform with measurable results.

- Repo: https://github.com/Pratyush-Mohanty/catastrophic-forgetting-lab
- Code: `cf_lab/` package · Dashboard: `app/dashboard.py` · Results: `experiments/*.json`
- Date: August 2026 · Hardware: laptop, **CPU-only** (NVIDIA MX150, unsupported CUDA)

---

## 1. What we set out to do

We kept reading that large language models "forget" old knowledge after
fine-tuning. Every explanation was abstract — curves in papers, claims without a
demo. So we decided to **build it from scratch** to watch it happen, measure it,
and test the standard defenses.

### The question

> Fine-tune one transformer on Task A, then Task B, then Task C — how much of
> Task A's accuracy is destroyed? And can EWC or Experience Replay prevent it?

### The experiment design

Task-incremental continual learning with a **shared encoder + one head per task**:

| Phase | Task | Dataset | Classes | Skill |
|---|---|---|---|---|
| 1 | IMDb | `stanfordnlp/imdb` | 2 | Sentiment analysis |
| 2 | AG News | `fancyzhx/ag_news` | 4 | News topic classification |
| 3 | DBpedia | `fancyzhx/dbpedia_14` | 14 | Entity ontology typing |

- **Model:** `google/bert_uncased_L-4_H-256_A-4` (~11M params — chosen so a full
  run finishes in ~11–13 min on CPU).
- **Settings:** 1000 train / 400 eval samples per task, 2 epochs, batch 32, LR 1e-4.
- **Protocol:** after every phase, evaluate **all** tasks with their own heads.
- **Real-world framing:** one company model sequentially fine-tuned for three
  products (sentiment analytics → news categorization → knowledge-base typing).

### The three strategies compared

1. **Baseline** — plain sequential fine-tuning. No memory, no protection.
2. **EWC** (Elastic Weight Consolidation, Kirkpatrick et al. 2017) — after each
   task, compute the diagonal **Fisher information** per weight, then add a penalty
   `½·λ·Σᵢ Fᵢ·(θᵢ−θ_Aᵢ)²` during later training to anchor the important weights.
   λ = 500.
3. **Experience Replay** — keep a memory buffer (300 examples) of old tasks and
   mix them into every batch (50% old, 50% new), each example routed to its own
   task's head.

---

## 2. What we built (end to end)

```
catastrophic-forgetting-lab/
├── cf_lab/                # core package
│   ├── config.py          # tasks, hyperparameters, hardware detection
│   ├── data.py            # load / shuffle / subsample / tokenize datasets
│   ├── model.py           # ContinualClassifier (shared encoder + per-task heads), compute_fisher
│   ├── training.py        # fine_tune_phase, EWC penalty, ReplayBuffer, _mix_replay
│   ├── experiment.py      # phase orchestration + step logging
│   ├── metrics.py         # retention, forgetting, confusion matrices
│   ├── plotting.py        # learning-curve & heatmap plots
│   ├── results.py         # JSON persistence + reloading
│   └── run.py             # CLI entry point
├── app/dashboard.py       # Streamlit dashboard (run / compare / theory / saved results)
├── experiments/           # 3 saved JSON results
├── docs/images/           # report figures
├── generate_images.py     # regenerate figures from results
└── run_dashboard.bat      # one-click launcher (venv-safe)
```

We deliberately wrote the training loop ourselves (no continual-learning
framework) so every moving part — Fisher estimation, replay mixing, multi-head
routing — is visible and debuggable.

---

## 3. What the experiments showed

All runs used identical settings (see §1). Accuracy on each task is measured
after every training phase.

### 3.1 Accuracy matrix (row = trained on, column = evaluated)

**Baseline** — the older the task, the more it decays:

| Trained on \ Evaluated | IMDb | AG News | DBpedia |
|---|---|---|---|
| IMDb | **0.647** | 0.270 | 0.013 |
| AG News | 0.627 | **0.805** | 0.050 |
| DBpedia | 0.600 | 0.810 | **0.905** |

**EWC** — the decay is largely suppressed:

| Trained on \ Evaluated | IMDb | AG News | DBpedia |
|---|---|---|---|
| IMDb | **0.647** | 0.273 | 0.013 |
| AG News | 0.637 | **0.820** | 0.048 |
| DBpedia | 0.630 | 0.818 | **0.902** |

**Replay** — nothing is lost, and task 1 even improves:

| Trained on \ Evaluated | IMDb | AG News | DBpedia |
|---|---|---|---|
| IMDb | **0.647** | 0.273 | 0.013 |
| AG News | 0.703 | **0.790** | 0.083 |
| DBpedia | 0.725 | 0.802 | **0.812** |

![confusion matrices](https://raw.githubusercontent.com/Pratyush-Mohanty/catastrophic-forgetting-lab/master/docs/images/confusion_matrices.png)

### 3.2 Retention summary (final accuracy ÷ peak accuracy; 1.00 = nothing forgotten)

| Strategy | IMDb | AG News | DBpedia | Runtime |
|---|---|---|---|---|
| Baseline | **0.93** | 0.98 | 1.00 | 678 s |
| EWC | **0.97** | 0.98 | 1.00 | 802 s |
| Replay | **1.00** | 1.00 | 1.00 | 790 s |

![retention comparison](https://raw.githubusercontent.com/Pratyush-Mohanty/catastrophic-forgetting-lab/master/docs/images/retention_comparison.png)

### 3.3 The numbers that matter

- **Baseline** lost **4.8 accuracy points** on IMDb (0.647 → 0.600), a ~7%
  retention drop — measurable, reproducible catastrophic forgetting.
- **EWC** cut that loss to **1.8 points** (0.647 → 0.630) using **zero extra data**,
  at a ~18% runtime cost.
- **Replay** not only preserved IMDb (0.647 → **0.725**) but *improved* it, at a
  ~16% runtime cost — with one trade-off: the newest task scored slightly lower
  (DBpedia 0.812 vs baseline 0.905) because half of each batch is replayed old data.

### 3.4 Learning curves

![baseline curves](https://raw.githubusercontent.com/Pratyush-Mohanty/catastrophic-forgetting-lab/master/docs/images/curves_baseline.png)
![ewc curves](https://raw.githubusercontent.com/Pratyush-Mohanty/catastrophic-forgetting-lab/master/docs/images/curves_ewc.png)
![replay curves](https://raw.githubusercontent.com/Pratyush-Mohanty/catastrophic-forgetting-lab/master/docs/images/curves_replay.png)

Watch the older-task lines slope downward in the baseline; EWC holds them flatter;
replay holds them essentially flat.

![task1 across strategies](https://raw.githubusercontent.com/Pratyush-Mohanty/catastrophic-forgetting-lab/master/docs/images/task1_across_strategies.png)

---

## 4. What we achieved

1. **Made catastrophic forgetting visible and measurable** in a real transformer,
   on a laptop CPU, in ~12 minutes per run.
2. **Demonstrated both classic defenses work:** EWC recovers most of the loss with
   no extra data; replay eliminates it entirely.
3. **Showed a practical trade-off:** replay spends "training budget" on old data,
   so the *newest* task gains less — a real engineering consideration.
4. **Delivered a reusable platform:** CLI, Streamlit dashboard (with pre-loaded
   results), JSON persistence, plotting, and an article-grade figure pipeline.
5. **Documented the journey** end to end — README, this report, and a Medium
   article (`MEDIUM.md`) with reproducible results.

---

## 5. Learning points

### On catastrophic forgetting itself

- **Forgetting needs conflicting tasks.** Our first attempt (IMDb → SST-2 →
  Amazon, all sentiment) showed *no* forgetting — the tasks were the same skill
  with different data, so training B *helped* A. Catastrophic forgetting is an
  interference phenomenon, not an inevitability.
- **It compounds with task order.** The older the task, the more it decays —
  every subsequent phase overwrites the shared weights that earlier heads depend on.
- **Peak ≠ final.** Evaluating "the model after the last phase" is the honest
  metric; evaluating right after training a task flatters it.

### On the mitigation strategies

- **EWC is data-free but hyperparameter-sensitive.** The penalty strength λ is a
  real dial: too small → forgetting returns; too large → the model stops learning
  the new task. (We used λ = 500.)
- **Replay is simple and strong but costs capacity.** It preserved *and improved*
  old tasks, but the newest task scored lower because half of every batch is old
  data. In a real product rollout, that is a direct business trade-off.
- **Frameworks ≠ magic.** Replay and EWC are ~60 lines of honest PyTorch; you do
  not need a continual-learning library to run serious experiments.

### On engineering pitfalls (the bugs that taught us the most)

- **HuggingFace `imdb` is sorted by label.** Taking the first N examples gives a
  degenerate single-class subset — our first "model" learned to always predict 0.
  Fix: **shuffle before subsampling** (now in `data.py`).
- **Correct Fisher estimation needs per-sample gradients computed *outside*
  `torch.no_grad()`** — the square of a sum is not the sum of squares. Doing it
  inside `no_grad` silently produces a wrong (near-zero) penalty.
- **`datasets` v5 changed its API.** IDs are namespaced (`fancyzhx/dbpedia_14`,
  not `dbpedia_14`), `trust_remote_code` no longer exists, and Amazon/DBpedia use a
  `content` column, not `text`. Old tutorials quietly break.
- **Multi-head replay breaks without head routing.** Replay batches contain
  examples from several tasks; each row needs its own `head_index`, and other
  tasks' logit columns must be padded with `-inf` so the loss ignores them.
- **Match the model to the machine.** `distilbert-base-uncased` is ~5 s/step on
  CPU; the L-4 H-256 BERT is ~0.85 s/step — a 6× difference that decides whether
  you can iterate at all. Right-sizing the model *is* an experiment design choice.
- **Environment discipline matters.** The dashboard crashed with a cryptic
  Streamlit error when launched with the wrong (system) Python that lacked the ML
  stack. Fix: a venv launcher (`run_dashboard.bat`) and a friendly preflight check
  in the app.

### On process

- **Reproducible runs beat big runs.** Fixed seeds, saved JSON per run, and a
  results loader let us compare strategies hours apart and re-plot without rerunning.
- **Start tiny, then scale.** Validating the whole pipeline at 200 samples (one
  fast run) before the full 1000-sample runs caught most bugs in ~3 minutes instead
  of ~1 hour.

---

## 6. Reproduce it yourself

```bash
git clone https://github.com/Pratyush-Mohanty/catastrophic-forgetting-lab.git
cd catastrophic-forgetting-lab
py -3.10 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt

# interactive dashboard (pre-loaded results)
run_dashboard.bat          # or: .venv\Scripts\python -m streamlit run app/dashboard.py

# CLI
.venv\Scripts\python -m cf_lab.run --mitigation baseline --plot
.venv\Scripts\python -m cf_lab.run --mitigation ewc --samples 1000 --epochs 2
.venv\Scripts\python -m cf_lab.run --mitigation replay --samples 1000 --epochs 2
```

Saved results live in `experiments/` and the dashboard's *Compare strategies* tab
loads them automatically.

## 7. Next steps

- Longer task chains (5–8 tasks) — forgetting should compound further.
- More mitigations: Learning Without Forgetting (LWF), LoRA adapters, GEM/AGEM.
- Larger encoders on GPU (`distilbert`, `bert-base`).
- A deployment-style regression test: evaluate the *final* model against all
  earlier product benchmarks, as a real MLOps platform would.
