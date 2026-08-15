"""Experiment orchestration: sequential fine-tuning across tasks with logging."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from . import metrics
from .config import ExperimentConfig
from .data import TaskData, load_task
from .model import load_model_and_tokenizer
from .training import (
    ReplayBuffer,
    fine_tune_phase,
    make_optimizer_and_scheduler,
    EWCState,
    StepLog,
)


@dataclass
class ExperimentResult:
    """Everything produced by one experiment run."""

    config: ExperimentConfig
    task_names: list[str]
    step_logs: list[StepLog] = field(default_factory=list)
    acc_matrix: list = field(default_factory=list)   # [phase][task] final accs
    peak_matrix: list = field(default_factory=list)  # [phase][task] best accs seen
    runtime_seconds: float = 0.0

    def summary(self):
        return metrics.summarize_runs(
            run_name=self.config.mitigation.kind,
            task_names=self.task_names,
            peak_accs=self.peak_matrix,
            final_accs=self.acc_matrix,
        )

    def confusion(self):
        return metrics.build_confusion(self.acc_matrix, self.task_names)

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "task_names": self.task_names,
            "steps": [s.__dict__ for s in self.step_logs],
            "acc_matrix": self.acc_matrix,
            "peak_matrix": self.peak_matrix,
            "runtime_seconds": self.runtime_seconds,
        }

    def save(self, output_dir: str | Path, run_id: str) -> Path:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{run_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def run_experiment(cfg: ExperimentConfig, progress_cb=None) -> ExperimentResult:
    """Sequentially fine-tune a model on each task and log accuracies.

    Mirrors the classic catastrophic-forgetting experiment: train on task A,
    then task B, then task C, measuring accuracy on *every* seen task after
    every phase. Mitigations (EWC / replay) modify training after the first
    task to reduce forgetting.

    progress_cb: optional callable(phase_index, phase_name, step, total_steps).
    """
    t0 = time.time()
    device = cfg.resolve_device()
    hp = cfg.resolved_training()
    mit = cfg.mitigation

    torch.manual_seed(hp.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(hp.seed)

    num_labels = cfg.tasks[0].num_labels
    model, tokenizer = load_model_and_tokenizer(cfg.model_name, num_labels, device)

    task_data: list[TaskData] = [load_task(spec, tokenizer) for spec in cfg.tasks]
    names = [td.spec.name for td in task_data]

    step_logs: list[StepLog] = []
    acc_matrix: list[list[float]] = []
    peak_matrix: list[list[float]] = []
    ewc_list: list[EWCState] = []
    replay_buffer = ReplayBuffer(mit.replay_buffer_size)
    step = 0

    def evaluate_all() -> dict:
        accs = {}
        model.eval()
        with torch.no_grad():
            for td in task_data:
                dataloader = DataLoader(td.eval, batch_size=hp.batch_size)
                correct = total = 0
                for batch in dataloader:
                    logits = model(
                        batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                    ).logits
                    correct += (logits.argmax(-1) == batch["labels"].to(device)).sum().item()
                    total += logits.size(0)
                accs[td.spec.name] = correct / max(total, 1)
        model.train()
        return accs

    accs = evaluate_all()
    step_logs.append(
        StepLog(step=0, phase="init", loss=float("nan"), lr=0.0, train_acc=0.0, task_accs=accs)
    )

    for phase_idx, td in enumerate(task_data):
        phase_name = td.spec.name
        dataloader = DataLoader(td.train, batch_size=hp.batch_size, shuffle=True)
        optimizer, scheduler = make_optimizer_and_scheduler(model, hp, len(dataloader))

        logs, step, ewc_list = fine_tune_phase(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=dataloader,
            phase_name=phase_name,
            task_index=phase_idx,
            num_epochs=hp.epochs,
            device=device,
            eval_fn=evaluate_all,
            eval_every_steps=hp.eval_every_steps,
            step_start=step,
            ewc_list=ewc_list,
            mitigation=mit,
            replay_buffer=replay_buffer,
            progress_cb=progress_cb,
        )
        step_logs.extend(logs)

        if mit.kind == "replay" and phase_idx == 0:
            for batch in dataloader:
                replay_buffer.add(batch, mit.replay_buffer_size)
                if len(replay_buffer.items) >= mit.replay_buffer_size:
                    break

        phase_accs = evaluate_all()
        acc_matrix.append([phase_accs[n] for n in names])

        best_this_phase = {n: 0.0 for n in names}
        for log in logs:
            for n in names:
                best_this_phase[n] = max(best_this_phase[n], log.task_accs.get(n, 0.0))
        for n in names:
            best_this_phase[n] = max(best_this_phase[n], phase_accs[n])

        prev_peak = peak_matrix[-1] if peak_matrix else [0.0] * len(names)
        peak_matrix.append(
            [max(prev_peak[j], best_this_phase[names[j]]) for j in range(len(names))]
        )

    return ExperimentResult(
        config=cfg,
        task_names=names,
        step_logs=step_logs,
        acc_matrix=acc_matrix,
        peak_matrix=peak_matrix,
        runtime_seconds=time.time() - t0,
    )
