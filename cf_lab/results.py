"""Persist and reload experiment results (JSON on disk)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .config import ExperimentConfig, TaskSpec
from .experiment import ExperimentResult
from .training import StepLog


def run_id(cfg: ExperimentConfig) -> str:
    """Stable, human-readable run id derived from the config."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tasks = "+".join(t.name.replace(" ", "").lower() for t in cfg.tasks)
    return f"{cfg.mitigation.kind}_{cfg.model_name.split('/')[-1]}_{tasks}_{stamp}"


def save_result(result: ExperimentResult) -> Path:
    rid = run_id(result.config)
    return result.save(result.config.output_dir, rid)


def load_result(path: str | Path) -> ExperimentResult:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    cfg_dict = data["config"]
    tasks = [TaskSpec(**t) for t in cfg_dict["tasks"]]
    mit = cfg_dict["mitigation"]
    train = cfg_dict["training"]
    from .config import MitigationConfig, TrainingHyperparams

    cfg = ExperimentConfig(
        tasks=tasks,
        model_name=cfg_dict["model_name"],
        training=TrainingHyperparams(**train),
        mitigation=MitigationConfig(**mit),
        output_dir=cfg_dict["output_dir"],
        device=cfg_dict["device"],
    )
    step_logs = [StepLog(**s) for s in data["steps"]]
    return ExperimentResult(
        config=cfg,
        task_names=data["task_names"],
        step_logs=step_logs,
        acc_matrix=data["acc_matrix"],
        peak_matrix=data["peak_matrix"],
        runtime_seconds=data["runtime_seconds"],
    )


def list_results(output_dir: str | Path = "experiments") -> list[Path]:
    out = Path(output_dir)
    if not out.exists():
        return []
    return sorted(out.glob("*.json"))
