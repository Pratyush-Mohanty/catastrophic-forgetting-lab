"""Validate the real pipeline: distilbert, 2 tasks, baseline, expect forgetting."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cf_lab.config import ExperimentConfig, TaskSpec, TrainingHyperparams, MitigationConfig
from cf_lab.experiment import run_experiment

cfg = ExperimentConfig(
    tasks=[
        TaskSpec(name="IMDb", dataset="stanfordnlp/imdb", max_train_samples=500, max_eval_samples=300),
        TaskSpec(name="SST-2", dataset="nyu-mll/glue", config_name="sst2",
                 text_column="sentence", max_train_samples=500, max_eval_samples=300),
    ],
    model_name="distilbert-base-uncased",
    training=TrainingHyperparams(epochs=1, batch_size=8, eval_every_steps=10, warmup_steps=5),
    mitigation=MitigationConfig(kind="baseline"),
    device="cpu",
)
result = run_experiment(cfg)
for log in result.step_logs:
    print(f"step={log.step:>4} phase={log.phase:>5} loss={log.loss:6.3f} train_acc={log.train_acc:5.3f} accs={ {k: round(v,3) for k,v in log.task_accs.items()} }")
print("\nconfusion:")
print(result.confusion().round(3).to_string())
print("summary:")
print(result.summary().round(3).to_string(index=False))
