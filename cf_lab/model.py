"""Model loading and helpers (Fisher information, per-sample gradients)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model_and_tokenizer(model_name: str, num_labels: int, device: str):
    """Load a transformer LM plus tokenizer, moved to `device`."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    ).to(device)
    model.train()
    return model, tokenizer


def _module_grads(model: nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot of current parameter gradients (filtered to trainable, non-bias)."""
    grads = {}
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads[name] = p.grad.detach().clone()
    return grads


def compute_fisher(
    model: nn.Module,
    dataloader: DataLoader,
    num_samples: int,
    device: str,
) -> dict[str, torch.Tensor]:
    """Diagonal Fisher information of the output log-probabilities on task data.

    F_i = E_x[ ( d log p(y | x, theta) / d theta_i )^2 ]

    Used by EWC to identify which parameters matter most for a previously
    learned task. Per-sample gradients are needed (the square of a sum is not
    the sum of squares), so we loop over individual examples.
    """
    fisher = {}
    count = 0
    model.eval()
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        for i in range(input_ids.size(0)):
            if count >= num_samples:
                break
            logits = model(input_ids[i : i + 1], attention_mask=attention_mask[i : i + 1]).logits
            probs = torch.softmax(logits, dim=-1)[0]
            sample_idx = torch.multinomial(probs, 1)
            log_prob = torch.log_softmax(logits, dim=-1)[0, sample_idx]

            model.zero_grad()
            log_prob.backward()

            for name, g in _module_grads(model).items():
                fisher.setdefault(name, torch.zeros_like(g))
                fisher[name] += g.square()
            count += 1

    for name in fisher:
        fisher[name] /= max(count, 1)
    model.train()
    return fisher


def model_params_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot of the current trainable parameters."""
    return {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }
