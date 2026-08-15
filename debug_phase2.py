"""Debug phase 2: does the model actually learn SST-2? Dump step logs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cf_lab.config import ExperimentConfig, TaskSpec, TrainingHyperparams, MitigationConfig
from cf_lab.experiment import run_experiment

cfg = ExperimentConfig(
    tasks=[
        TaskSpec(name="IMDb", dataset="stanfordnlp/imdb", max_train_samples=300, max_eval_samples=150),
        TaskSpec(name="SST-2", dataset="nyu-mll/glue", config_name="sst2",
                 text_column="sentence", max_train_samples=300, max_eval_samples=150),
    ],
    model_name="google/bert_uncased_L-2_H-128_A-2",
    training=TrainingHyperparams(epochs=1, batch_size=8, eval_every_steps=10, warmup_steps=5),
    mitigation=MitigationConfig(kind="baseline"),
    device="cpu",
)
result = run_experiment(cfg)
for log in result.step_logs:
    print(f"step={log.step:>4} phase={log.phase:>5} loss={log.loss:6.3f} train_acc={log.train_acc:5.3f} accs={ {k: round(v,3) for k,v in log.task_accs.items()} }")
print()
print("confusion:")
print(result.confusion().round(3).to_string())
