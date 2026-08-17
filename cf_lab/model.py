"""Model loading: a shared encoder with task-specific classification heads.

This is the standard *task-incremental* continual-learning setup: one shared
backbone plus one linear head per task. Sequential fine-tuning on task B shifts
the shared weights, which degrades task A's head — the classic catastrophic
forgetting scenario (distinct tasks, distinct label spaces).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer


class ContinualClassifier(nn.Module):
    """Shared transformer encoder + one linear head per task."""

    def __init__(self, base: nn.Module, hidden_size: int, head_sizes: list[int]):
        super().__init__()
        self.base = base
        self.heads = nn.ModuleDict(
            {str(i): nn.Linear(hidden_size, n) for i, n in enumerate(head_sizes)}
        )
        self.max_out = max(head_sizes)

    def forward(self, input_ids, attention_mask, head_index) -> torch.Tensor:
        """Logits for each row, using the head selected by head_index.

        Output is padded to max_out columns with -inf so that cross-entropy
        ignores the padding for smaller tasks (useful for replay-mixed batches).
        """
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output
        logits = torch.full(
            (pooled.size(0), self.max_out), float("-inf"), device=pooled.device
        )
        for i, head in self.heads.items():
            hidx = int(i)
            sel = head_index == hidx
            if sel.any():
                logits[sel, : head.out_features] = head(pooled[sel])
        return logits


def load_continual_model(
    model_name: str, head_sizes: list[int], device: str
) -> tuple[ContinualClassifier, AutoTokenizer]:
    """Load a base LM and attach one classification head per task size."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base = AutoModel.from_pretrained(model_name)
    hidden = base.config.hidden_size
    model = ContinualClassifier(base, hidden, head_sizes).to(device)
    model.train()
    return model, tokenizer


def _module_grads(model: nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot of current parameter gradients (filtered to trainable)."""
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
    head_index: int = 0,
) -> dict[str, torch.Tensor]:
    """Diagonal Fisher information of the output log-probabilities on task data.

    F_i = E_x[ ( d log p(y | x, theta) / d theta_i )^2 ]

    Used by EWC to identify which parameters matter most for a previously
    learned task. Per-sample gradients are needed (the square of a sum is not
    the sum of squares), so we loop over individual examples.
    """
    fisher = {}
    count = 0
    hidx = torch.tensor([head_index], device=device, dtype=torch.long)
    model.eval()
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        for i in range(input_ids.size(0)):
            if count >= num_samples:
                break
            logits = model(input_ids[i : i + 1], attention_mask[i : i + 1], hidx)
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