"""Fine-tuning loops with support for EWC penalties and experience replay."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .config import MitigationConfig, TrainingHyperparams
from .model import compute_fisher, model_params_state


@dataclass
class StepLog:
    step: int
    phase: str
    loss: float
    lr: float
    train_acc: float
    task_accs: dict = field(default_factory=dict)


@dataclass
class EWCState:
    """Fisher information and anchor parameters from a previously trained task."""

    fisher: dict[str, torch.Tensor]
    anchor: dict[str, torch.Tensor]


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def _build_ewc_penalty(
    model: nn.Module, ewc: list[EWCState], lam: float, device: str
) -> torch.Tensor:
    """Weighted squared-distance penalty anchoring parameters to old tasks."""
    if not ewc:
        return torch.zeros((), device=device)
    total = torch.zeros((), device=device)
    for state in ewc:
        for name, p in model.named_parameters():
            if p.requires_grad and name in state.fisher:
                diff = p - state.anchor[name]
                total = total + (state.fisher[name] * diff.square()).sum()
    return 0.5 * lam * total


class ReplayBuffer:
    """Fixed-size exemplar memory of (input, label, head) from earlier tasks."""

    def __init__(self, size: int):
        self.size = size
        self.items: list[dict] = []

    def add(self, batch: dict, head_index: int, max_add: int):
        for i in range(len(batch["input_ids"])):
            item = {
                "input_ids": batch["input_ids"][i],
                "attention_mask": batch["attention_mask"][i],
                "labels": batch["labels"][i],
                "head_index": torch.tensor(head_index, dtype=torch.long),
            }
            if len(self.items) < self.size:
                self.items.append(item)
            else:
                self.items[random.randrange(self.size)] = item
        self.items = self.items[:max_add]

    def sample(self, n: int) -> dict:
        chosen = random.sample(self.items, min(n, len(self.items)))
        return {
            "input_ids": torch.stack([c["input_ids"] for c in chosen]),
            "attention_mask": torch.stack([c["attention_mask"] for c in chosen]),
            "labels": torch.stack([c["labels"] for c in chosen]),
            "head_index": torch.stack([c["head_index"] for c in chosen]),
        }


def _mix_replay(
    batch: dict,
    buffer: ReplayBuffer,
    ratio: float,
) -> dict:
    """Replace a `ratio` fraction of a current-task batch with memory samples."""
    if not buffer.items or ratio <= 0:
        return batch
    n = len(batch["input_ids"])
    k = max(1, int(n * ratio))
    mem = buffer.sample(k)
    idx = random.sample(range(n), k)
    merged = dict(batch)
    for key in ("input_ids", "attention_mask", "labels", "head_index"):
        src = mem[key]
        merged[key] = batch[key].clone()
        for j, i in enumerate(idx):
            merged[key][i] = src[j]
    return merged


def fine_tune_phase(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    dataloader: DataLoader,
    phase_name: str,
    task_index: int,
    num_epochs: int,
    device: str,
    eval_fn: Callable[[], dict],
    eval_every_steps: int,
    step_start: int,
    ewc_list: list[EWCState],
    mitigation: MitigationConfig,
    replay_buffer: ReplayBuffer | None,
    progress_cb=None,
) -> tuple[list[StepLog], int, list[EWCState]]:
    """Train one phase on one task. Returns (step_logs, final_step, ewc_list)."""
    logs: list[StepLog] = []
    step = step_start
    total_steps = len(dataloader) * num_epochs
    head_index = torch.tensor([task_index], device=device, dtype=torch.long)
    optimizer.zero_grad(set_to_none=True)
    model.train()

    for epoch in range(num_epochs):
        running_loss, running_correct, running_total = 0.0, 0, 0
        for batch in dataloader:
            batch["head_index"] = head_index.expand(batch["input_ids"].size(0))
            if mitigation.kind == "replay" and replay_buffer is not None:
                batch = _mix_replay(batch, replay_buffer, mitigation.replay_ratio)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            heads = batch["head_index"].to(device)

            logits = model(input_ids, attention_mask, heads)
            loss = F.cross_entropy(logits, labels)
            if mitigation.kind == "ewc":
                loss = loss + _build_ewc_penalty(
                    model, ewc_list, mitigation.ewc_lambda, device
                )

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if progress_cb is not None:
                progress_cb(task_index, phase_name, step - step_start, total_steps)

            running_loss += loss.item() * input_ids.size(0)
            running_correct += (logits.argmax(-1) == labels).sum().item()
            running_total += input_ids.size(0)
            step += 1

            if step % eval_every_steps == 0 or step - step_start >= total_steps:
                model.eval()
                task_accs = eval_fn()
                model.train()
                logs.append(
                    StepLog(
                        step=step,
                        phase=phase_name,
                        loss=running_loss / max(running_total, 1),
                        lr=scheduler.get_last_lr()[0],
                        train_acc=running_correct / max(running_total, 1),
                        task_accs=task_accs,
                    )
                )
                running_loss, running_correct, running_total = 0.0, 0, 0

    if not logs:
        model.eval()
        logs.append(
            StepLog(
                step=step,
                phase=phase_name,
                loss=running_loss / max(running_total, 1),
                lr=scheduler.get_last_lr()[0],
                train_acc=running_correct / max(running_total, 1),
                task_accs=eval_fn(),
            )
        )
        model.train()

    if mitigation.kind == "ewc":
        fisher = compute_fisher(
            model, dataloader, mitigation.ewc_fisher_samples, device, task_index
        )
        ewc_list = list(ewc_list)
        ewc_list.append(
            EWCState(fisher=fisher, anchor=model_params_state(model))
        )
    return logs, step, ewc_list


def make_optimizer_and_scheduler(
    model: nn.Module,
    hp: TrainingHyperparams,
    steps_per_epoch: int,
):
    """AdamW optimizer + linear warmup/linear decay scheduler."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hp.learning_rate, weight_decay=0.01
    )
    total_steps = steps_per_epoch * hp.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=hp.warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler