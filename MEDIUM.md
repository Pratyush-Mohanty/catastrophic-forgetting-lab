# Catastrophic Forgetting in LLMs: I Built an Experiment Lab to Watch It Happen

*An end-to-end journey: theory → real-world simulation → code → visual proof of why
your fine-tuned model "forgets" — and how EWC and Experience Replay fight back.*

---

> **How to publish this on Medium:** open `MEDIUM.md`, copy the whole text, and
> paste it into the Medium editor. Medium ignores local image paths, so **after
> pasting, upload the images** from `docs/images/` in this order:
> `architecture.png` (sec 3) → `confusion_matrices.png` (6.1) →
> `curves_baseline.png` (6.2) → `curves_ewc.png` (6.2) → `curves_replay.png`
> (6.2) → `retention_comparison.png` + `retention_table.png` (6.3) →
> `task1_across_strategies.png` (6.4). Click each `![...]` placeholder's image
> icon and replace it with the uploaded image. (The GitHub raw URLs below are
> there so previews render everywhere; Medium itself needs the uploaded files.)

---

> **TL;DR** I built a complete experiment platform that fine-tunes a transformer
> language model on three tasks in sequence (IMDb → AG News → DBpedia) and
> measures what happens to earlier tasks. The result? The model genuinely loses
> up to 7% of its first task's accuracy — and the classic defenses (EWC,
> Experience Replay) visibly slow that loss. All code, results, and an interactive
> dashboard are in the open-source repo:
> **github.com/Pratyush-Mohanty/catastrophic-forgetting-lab**.

---

## 1. The problem nobody talks about until it bites

Every ML practitioner knows this pain: you fine-tune a strong foundation model on
*your* domain data, it performs beautifully, and then — weeks later — you fine-tune
it again for a second product. The second model is great. But quietly, the first
product's accuracy has collapsed. The model forgot.

### Why I built this

Honestly? I was just **curious**. I kept reading that large language models
"forget" older knowledge after fine-tuning, and every explanation was abstract —
curves in papers, equations in reviews, claims without a demo I could poke at. So I
decided to stop reading about it and **build it from scratch** to see it with my own
eyes. I wrote the continual-learning trainer, the three mitigation strategies, the
dashboard, and the plotting pipeline myself — no framework for the core loop, just
PyTorch, HuggingFace transformers, and a lot of debugging. The graphs in this
article are the output of my own experiments, not screenshots borrowed from a paper.
If you're the kind of person who has to *break something yourself* to believe it —
this article is for you.

### What catastrophic forgetting actually is

This is **catastrophic forgetting (CF)**, first formalized by McCloskey & Cohen in
1989:

> When a neural network is trained on new information, gradient descent overwrites
> the shared weights. Knowledge required for an earlier task is destroyed — often
> rapidly and completely — even though nothing about that task changed.

It matters more than ever for LLMs, because a single pretrained model is
increasingly adapted to many tasks, domains, and customers. Naive sequential
fine-tuning silently erases general knowledge.

**The goal of this project:** build an end-to-end lab to *see* forgetting happen
in a real transformer, quantify it, and test the strategies researchers use to stop
it.

---

## 2. The experiment design (and the trap I fell into)

To study CF you need a setup where learning task B genuinely *conflicts* with
task A. The standard research formulation is **task-incremental continual
learning**:

```
Train on Task A (IMDb)    → evaluate on {IMDb, AG News, DBpedia}
Train on Task B (AG News) → evaluate on {IMDb, AG News, DBpedia}
Train on Task C (DBpedia) → evaluate on {IMDb, AG News, DBpedia}
```

- **Task 1 — IMDb**: binary sentiment (positive / negative)
- **Task 2 — AG News**: news topic (4 classes)
- **Task 3 — DBpedia**: ontology (14 classes)

Each task has a *different label space* and its **own small classification head**,
but all share **one BERT encoder**. When training on DBpedia moves the shared
weights, the features IMDb's head relies on get rewritten. That interference is
the catastrophe.

### The design trap (worth knowing)

My first attempt used three *sentiment* tasks (IMDb → SST-2 → Amazon). The result:
**no forgetting at all.** Learning task B actually *helped* task A — positive
transfer, because all three tasks were the same skill with different data.

> **Lesson:** catastrophic forgetting needs tasks that *conflict*, not tasks that
> overlap. Distinct label spaces with a shared backbone is the clean way to expose it.

---

## 3. The architecture

![architecture](https://raw.githubusercontent.com/Pratyush-Mohanty/catastrophic-forgetting-lab/master/docs/images/architecture.png)

- A shared **BERT encoder** (L-4 H-256, ~11M params — small enough to fine-tune on a CPU in minutes).
- **One linear head per task** (`2 / 4 / 14` classes).
- A `head_index` per example routes it to the correct head, even inside a single
  replay batch.
- After each training phase we evaluate **all** tasks with their own heads.

---

## 4. The three strategies: baseline, EWC, replay

This project compares three ways to handle sequential fine-tuning. Here's what
each one is.

### 4.1 Baseline — do nothing special

Plain sequential fine-tuning: optimize the whole network on each task one after
another. No memory, no protection. This is the control group, and it is where we
expect the forgetting.

**Why it forgets:** the optimizer has no reason to preserve features from task A.
Every gradient step is "what's best for the current batch," and the current batch
doesn't care about sentiment anymore.

### 4.2 EWC — Elastic Weight Consolidation (Kirkpatrick et al., 2017)

EWC protects old knowledge by *telling the optimizer which weights mattered*.

After finishing a task, we compute the **diagonal Fisher information matrix `F`** —
a per-parameter estimate of how sensitive the model's predictions were to each
weight. Weights with high Fisher values were crucial for that task.

During later training, we add a penalty to the loss:

```
L = L_new  +  ½ · λ · Σ_i  F_i · (θ_i − θ_A,i)²
```

This "anchors" the important parameters: they can still move, but only if the new
task *really* needs them to. The Fisher is computed with **per-sample gradients**
(a subtle detail — the square of a sum is not the sum of squares — and one of the
bugs we caught during validation).

**Cost:** no extra data needed, only a small computational overhead. But choosing
λ (the penalty strength) matters — too high and the model can't learn the new task.

### 4.3 Experience Replay (rehearsal)

The simplest, most human idea: **don't forget the past — revisit it.**

Keep a fixed-size memory buffer of examples from earlier tasks. While training on
the new task, mix a `replay_ratio` fraction of remembered examples into each
batch (each example tagged with the head it belongs to). The model re-trains on
old data *while* learning new data, so its weights never drift far from what task
A needed.

**Cost:** needs to store old data (privacy/storage concerns in the real world) and
adds a bit of training time.

| Strategy | Core idea | Needs extra data? | Code |
|---|---|---|---|
| Baseline | Just fine-tune | No | `fine_tune_phase` + plain loss |
| EWC | Penalize moving important weights | No | Fisher penalty `½λ·F·(θ−θ_A)²` |
| Replay | Mix old examples into new batches | Yes (buffer) | `ReplayBuffer` + `_mix_replay` |

---

## 5. Simulating a real-world scenario

Let's make it concrete. Imagine a company that fine-tunes **one general LLM** for
three successive products:

1. **Product A — sentiment analytics** (IMDb reviews). The model learns to judge
   positive/negative sentiment.
2. **Product B — news categorization** (AG News). The same model is re-fine-tuned
   to tag articles by topic.
3. **Product C — knowledge-base typing** (DBpedia). Again, the same model, now
   classifying entities into 14 ontology classes.

After every product's training we run regression tests on *all three* products —
exactly what a real MLOps team would do. This is the "real-world use case
simulation": one shared model, sequentially adapted, with measurable
capability loss on older products.

---

## 6. What happened — the results

All runs share identical settings: BERT L-4 H-256, 1000 train / 400 eval samples
per task, 2 epochs, batch 32, LR 1e-4, CPU.

### 6.1 The accuracy matrix (baseline)

Rows are what we trained on; columns are what we evaluated.

![confusion matrices](https://raw.githubusercontent.com/Pratyush-Mohanty/catastrophic-forgetting-lab/master/docs/images/confusion_matrices.png)

The off-diagonal story: **IMDb peaked at 0.648 and finished at 0.600** after the
other tasks were learned — a real, measurable loss. AG News held better (its
features overlap with what's learned last), and DBpedia, being newest, is intact.

### 6.2 Learning curves — watch the lines fall

Accuracy on every task as training progresses. The vertical dashed lines mark
when each new task starts. Watch the older tasks' lines slope downward.

**Baseline** — the oldest task's accuracy erodes with every new phase:

![baseline curves](https://raw.githubusercontent.com/Pratyush-Mohanty/catastrophic-forgetting-lab/master/docs/images/curves_baseline.png)

**EWC** — the Fisher anchor keeps old-task accuracy much flatter:

![ewc curves](https://raw.githubusercontent.com/Pratyush-Mohanty/catastrophic-forgetting-lab/master/docs/images/curves_ewc.png)

**Replay** — re-training on remembered examples holds every task essentially flat:

![replay curves](https://raw.githubusercontent.com/Pratyush-Mohanty/catastrophic-forgetting-lab/master/docs/images/curves_replay.png)

### 6.3 Retention summary

Retention = final accuracy ÷ peak accuracy. **1.00 = nothing forgotten.**

![retention comparison](https://raw.githubusercontent.com/Pratyush-Mohanty/catastrophic-forgetting-lab/master/docs/images/retention_comparison.png)

![retention table](https://raw.githubusercontent.com/Pratyush-Mohanty/catastrophic-forgetting-lab/master/docs/images/retention_table.png)

| Strategy | IMDb | AG News | DBpedia |
|---|---|---|---|
| Baseline | 0.93 | 0.98 | 1.00 |
| EWC | 0.97 | 0.98 | 1.00 |
| Replay | 1.00 | 1.00 | 1.00 |

### 6.4 The money plot — task-1 accuracy while everything else is learned

![task1 across strategies](https://raw.githubusercontent.com/Pratyush-Mohanty/catastrophic-forgetting-lab/master/docs/images/task1_across_strategies.png)

The baseline curve visibly slopes down; EWC and replay hold the line.

---

## 7. What we achieved

1. **A working, reproducible experiment** that demonstrates catastrophic
   forgetting in a real transformer, on a laptop CPU, in ~15 minutes per run.
2. **Quantified the phenomenon** — baseline loses ~7% of task-1 retention; the
   older the task, the worse it fares.
3. **Proved the two classic defenses actually work** — EWC recovers most of the
   loss with zero extra data; replay holds the line completely.
4. **An interactive dashboard** (Streamlit) to run experiments, tweak settings,
   and compare strategies live, plus a CLI and pre-computed results.
5. **Hard-won engineering lessons** documented in the code:
   - HuggingFace's `imdb` dataset is **sorted by label** — taking the first N
     examples gives a degenerate single-class subset (our first runs "learned" to
     always predict 0).
   - Correct Fisher estimation needs per-sample gradients *outside*
     `torch.no_grad()`.
   - Multi-head replay batches need `-inf`-padded logits so each task ignores
     other tasks' class columns.

---

## 8. Try it yourself

```bash
git clone https://github.com/Pratyush-Mohanty/catastrophic-forgetting-lab.git
cd catastrophic-forgetting-lab
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# interactive dashboard
streamlit run app/dashboard.py

# or from the CLI
python -m cf_lab.run --mitigation baseline --plot
python -m cf_lab.run --mitigation ewc --epochs 2 --samples 1000
python -m cf_lab.run --mitigation replay --epochs 2 --samples 1000 --plot
```

The dashboard ships with the three pre-computed runs loaded, so you can open the
"Compare strategies" tab and see baseline vs EWC vs replay immediately.

---

## 9. What's next

- **Longer task sequences** — CF compounds; 5–8 tasks should show even steeper
  baseline decay.
- **More mitigations** — Learning Without Forgetting (LWF), LoRA adapters,
  and memory-replay variants (GEM/AGEM).
- **Bigger models** — the same code runs on `distilbert-base-uncased` or
  `bert-base-uncased` if you have a GPU.
- **Real deployment analog** — regression-test the *final* model against all
  earlier benchmarks, the way a production ML platform would.

---

## 10. References

- McCloskey, M. & Cohen, N. J. (1989). *Catastrophic interference in connectionist
  networks.* Psychology of Learning and Motivation.
- French, R. M. (1999). *Catastrophic forgetting in connectionist networks.* Trends
  in Cognitive Sciences.
- Kirkpatrick, J. et al. (2017). *Overcoming catastrophic forgetting in neural
  networks.* PNAS.
- Repo: [Pratyush-Mohanty/catastrophic-forgetting-lab](https://github.com/Pratyush-Mohanty/catastrophic-forgetting-lab)

---

*If you found this useful, follow along — the repo is open-source and the next
experiments (LoRA, LWF, longer task chains) are already planned.*