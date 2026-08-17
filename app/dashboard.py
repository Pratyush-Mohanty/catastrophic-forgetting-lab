"""Streamlit dashboard for the Catastrophic Forgetting Lab.

Run from the project root:
    streamlit run app/dashboard.py
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch  # noqa: F401  (probe for a working ML environment)
    import transformers  # noqa: F401
    import datasets  # noqa: F401

    from cf_lab.config import default_config, detect_device
    from cf_lab.experiment import ExperimentResult, run_experiment
    from cf_lab.plotting import (
        task_accuracy_curves,
        confusion_heatmap,
        compare_runs,
    )
    from cf_lab.results import list_results, load_result, save_result
except Exception as exc:  # pragma: no cover - friendly failure path
    st.set_page_config(page_title="Catastrophic Forgetting Lab", layout="wide")
    st.error(
        "**The app could not start: dependencies are missing or broken.**\n\n"
        f"`{type(exc).__name__}: {exc}`\n\n"
        "This usually happens when the app is launched with the **wrong Python** "
        "(e.g. the system Python instead of the project venv). Fix it with:\n\n"
        "```\ncd catastrophic-forgetting-lab\npy -3.10 -m venv .venv\n.venv\\Scripts\\python -m pip install -r requirements.txt\n.venv\\Scripts\\python -m streamlit run app/dashboard.py\n```\n\n"
        "or simply double-click **`run_dashboard.bat`**."
    )
    st.stop()

st.set_page_config(page_title="Catastrophic Forgetting Lab", layout="wide")

THEORY = """
### What is catastrophic forgetting?

A neural network trained sequentially on new tasks quickly **destroys** its
performance on previously learned ones. The weights that solved task A are
overwritten while fitting task B — the model does not "remember" what it knew.

**Key papers**
- McCloskey & Cohen (1989) — the original observation
- French (1999) — why shared weights cause interference
- Kirkpatrick et al. (2017) — *Elastic Weight Consolidation* (EWC), Nature

### How this app demonstrates it
We take a transformer language model and fine-tune it on three text
classification tasks in order. After each phase we measure accuracy on *all*
seen tasks. A big accuracy drop on earlier tasks = catastrophic forgetting.

### Mitigations explored
1. **Baseline** — plain sequential fine-tuning (forgetting expected).
2. **EWC** — penalizes parameter movement that would hurt old tasks, using the
   diagonal Fisher information to say which weights mattered for task A.
3. **Experience replay** — keep a small memory of task-A examples and replay
   them while training on task B (rehearsal).
"""


def _run_in_thread(cfg):
    result_holder = {}

    def worker():
        result_holder["result"] = run_experiment(cfg)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t, result_holder


st.sidebar.title("Catastrophic Forgetting Lab")
st.sidebar.markdown("Experiment configuration")

model_name = st.sidebar.selectbox(
    "Model",
    ["google/bert_uncased_L-4_H-256_A-4", "distilbert-base-uncased", "bert-base-uncased"],
    index=0,
    help="L-4 H-256 is fast on CPU and recommended. distilbert is slower but stronger.",
)

mitigation = st.sidebar.selectbox(
    "Mitigation strategy",
    ["baseline", "ewc", "replay"],
    format_func=lambda m: {
        "baseline": "Baseline (plain fine-tuning)",
        "ewc": "EWC (elastic weight consolidation)",
        "replay": "Experience replay (rehearsal)",
    }[m],
)

epochs = st.sidebar.slider("Epochs per task", 1, 5, 1)
samples = st.sidebar.slider("Train samples per task", 500, 6000, 1500, step=250)
batch_size = st.sidebar.selectbox("Batch size", [4, 8, 16, 32], index=2)
learning_rate = st.sidebar.select_slider(
    "Learning rate", options=[1e-5, 2e-5, 3e-5, 5e-5], value=2e-5
)

ewc_lambda = None
replay_ratio = None
if mitigation == "ewc":
    ewc_lambda = st.sidebar.slider("EWC lambda", 0.0, 2000.0, 500.0, step=50.0)
elif mitigation == "replay":
    replay_ratio = st.sidebar.slider("Replay ratio (fraction of memory per batch)", 0.0, 1.0, 0.5, step=0.1)

device = detect_device()
st.sidebar.markdown(f"**Detected device:** `{device}`")

run_button = st.sidebar.button("Run experiment", type="primary")

st.title("Studying Catastrophic Forgetting in LLMs")
st.markdown(
    f"Fine-tune **{model_name}** sequentially on IMDb → AG News → DBpedia "
    f"(distinct tasks, distinct label spaces, shared encoder + per-task heads). "
    f"Watch what happens to earlier tasks. Strategy: **{mitigation}**."
)

tab_run, tab_compare, tab_theory, tab_saved = st.tabs(
    ["Experiment", "Compare strategies", "Theory", "Saved results"]
)

with tab_theory:
    st.markdown(THEORY)

results_cache = st.session_state.setdefault("results", [])

if run_button:
    cfg = default_config()
    cfg.model_name = model_name
    cfg.mitigation.kind = mitigation
    if ewc_lambda is not None:
        cfg.mitigation.ewc_lambda = ewc_lambda
    if replay_ratio is not None:
        cfg.mitigation.replay_ratio = replay_ratio
    cfg.training.epochs = epochs
    cfg.training.batch_size = batch_size
    cfg.training.learning_rate = learning_rate
    for task in cfg.tasks:
        task.max_train_samples = samples

    with tab_run:
        status = st.status(f"Running {mitigation} experiment…", expanded=True)
        progress = st.progress(0.0)
        status.write(f"Device: {device} | model: {model_name}")

        thread, holder = _run_in_thread(cfg)
        while thread.is_alive():
            time.sleep(0.5)
        result = holder["result"]
        progress.progress(1.0)
        status.write(f"Done in {result.runtime_seconds:.1f}s")

        results_cache.append(result)
        save_result(result)

with tab_run:
    if results_cache:
        result = results_cache[-1]
        st.subheader(f"Result: {result.config.mitigation.kind}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Runtime", f"{result.runtime_seconds:.1f}s")
        col2.metric("Tasks", " → ".join(result.task_names))
        col3.metric(
            "Task-1 retention",
            f"{result.summary().iloc[0]['retention']:.0%}",
        )

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(9, 5))
            task_accuracy_curves(result, ax=ax)
            st.pyplot(fig)
            plt.close(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(8, 5))
            confusion_heatmap(result, ax=ax)
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("#### Per-task summary")
        st.dataframe(result.summary().round(3))

with tab_compare:
    if not results_cache:
        saved = list_results()
        for p in saved:
            results_cache.append(load_result(p))
    if results_cache:
        st.subheader("Retention on the first task vs mitigation")
        df = compare_runs(results_cache, task_index=0)
        if not df.empty:
            st.dataframe(df.round(3))
            st.bar_chart(df.set_index("mitigation")[["retention", "forgetting"]])

            st.markdown("#### All task accuracy curves (overlaid)")
            fig, ax = plt.subplots(figsize=(10, 5))
            for r in results_cache:
                task = r.task_names[0]
                steps = [l.step for l in r.step_logs if task in l.task_accs]
                accs = [l.task_accs[task] for l in r.step_logs if task in l.task_accs]
                ax.plot(steps, accs, marker="o", markersize=3, label=r.config.mitigation.kind)
            ax.set_xlabel("Training step")
            ax.set_ylabel(f"Accuracy on {results_cache[-1].task_names[0]}")
            ax.set_ylim(0, 1.05)
            ax.legend()
            ax.set_title("Task-1 accuracy across mitigation strategies")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Run at least one experiment first.")
    else:
        st.info("Run at least one experiment first.")

with tab_saved:
    paths = list_results()
    if not paths:
        st.info("No saved results yet.")
    else:
        choice = st.selectbox("Saved experiment files", paths, format_func=str)
        if choice:
            loaded = load_result(choice)
            st.json(loaded.to_dict(), expanded=False)
