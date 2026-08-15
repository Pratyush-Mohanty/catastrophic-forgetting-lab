"""Tiny end-to-end smoke test of the experiment engine."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cf_lab.config import ExperimentConfig, MitigationConfig, TaskSpec, TrainingHyperparams
from cf_lab.experiment import run_experiment

for mit in ["baseline", "ewc", "replay"]:
    cfg = ExperimentConfig(
        tasks=[
            TaskSpec(name="IMDb", dataset="stanfordnlp/imdb", max_train_samples=400, max_eval_samples=200),
            TaskSpec(name="SST-2", dataset="nyu-mll/glue", config_name="sst2",
                     text_column="sentence", max_train_samples=400, max_eval_samples=200),
        ],
        model_name="google/bert_uncased_L-2_H-128_A-2",
        training=TrainingHyperparams(epochs=1, batch_size=8, eval_every_steps=5, warmup_steps=5),
        mitigation=MitigationConfig(kind=mit, ewc_fisher_samples=100, replay_buffer_size=200),
        device="cpu",
    )
    result = run_experiment(cfg)
    print(f"\n=== {mit} ===")
    print(result.confusion().round(3).to_string())
    print(f"retention on IMDb: {result.summary().iloc[0]['retention']:.3f}")
    print(f"runtime: {result.runtime_seconds:.1f}s")
