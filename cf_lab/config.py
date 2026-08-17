"""Configuration, hardware detection and sensible CPU/GPU-adaptive defaults."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import torch


def detect_device() -> str:
    """Return the best available torch device, preferring CUDA when usable."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _gpu_memory_gb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        return None


@dataclass
class TaskSpec:
    """A single continual-learning task built from a HuggingFace dataset."""

    name: str
    dataset: str            # HF dataset id, e.g. "stanfordnlp/imdb", "nyu-mll/glue"
    config_name: str = ""   # HF subset name, e.g. "sst2" for glue
    text_column: str = "text"
    label_column: str = "label"
    num_labels: int = 2
    max_train_samples: int = 4000
    max_eval_samples: int = 800
    max_length: int = 128


@dataclass
class TrainingHyperparams:
    """Shared hyperparameters for every fine-tuning phase."""

    epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 1e-4
    warmup_steps: int = 5
    eval_every_steps: int = 50
    seed: int = 42


@dataclass
class MitigationConfig:
    """Settings for the anti-forgetting mitigation strategies."""

    kind: str = "baseline"      # "baseline" | "ewc" | "replay"
    ewc_lambda: float = 500.0   # EWC: weight of the Fisher penalty
    ewc_fisher_samples: int = 200
    replay_buffer_size: int = 300
    replay_ratio: float = 0.5   # fraction of each replay-batch coming from memory


@dataclass
class ExperimentConfig:
    """Everything needed to run a catastrophic-forgetting experiment."""

    tasks: list = field(
        default_factory=lambda: [
            TaskSpec(name="IMDb", dataset="stanfordnlp/imdb", num_labels=2,
                     max_train_samples=1000, max_eval_samples=400),
            TaskSpec(name="AG News", dataset="fancyzhx/ag_news", num_labels=4,
                     max_train_samples=1000, max_eval_samples=400),
            TaskSpec(name="DBpedia", dataset="fancyzhx/dbpedia_14", num_labels=14,
                     text_column="content", max_train_samples=1000, max_eval_samples=400),
        ]
    )
    model_name: str = "google/bert_uncased_L-4_H-256_A-4"
    training: TrainingHyperparams = field(default_factory=TrainingHyperparams)
    mitigation: MitigationConfig = field(default_factory=MitigationConfig)
    output_dir: str = "experiments"
    device: str = "auto"

    def resolve_device(self) -> str:
        if self.device == "auto":
            return detect_device()
        return self.device

    def resolved_training(self) -> TrainingHyperparams:
        """Return hyperparams with CPU-adaptive sizes for slow machines."""
        return self.training

    def to_dict(self) -> dict:
        return asdict(self)


def default_config() -> ExperimentConfig:
    """Build a config tuned to the current hardware."""
    cfg = ExperimentConfig()
    mem = _gpu_memory_gb()
    if mem is not None and mem < 3.0:
        # Tiny GPU: shrink model and data so experiments still fit.
        cfg.model_name = "google/bert_uncased_L-4_H-256_A-4"
        for task in cfg.tasks:
            task.max_train_samples = min(task.max_train_samples, 1500)
    return cfg
