"""Task loading: fetch, subsample and tokenize HuggingFace datasets."""

from __future__ import annotations

from dataclasses import dataclass

from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from .config import TaskSpec


@dataclass
class TaskData:
    """Tokenized train/validation split for one task."""

    spec: TaskSpec
    train: "TorchDataset"
    eval: "TorchDataset"
    num_labels: int


class TorchDataset(Dataset):
    """Pytorch wrapper over a HuggingFace dataset of tokenized examples."""

    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        return {
            "input_ids": row["input_ids"],
            "attention_mask": row["attention_mask"],
            "labels": row["label"],
        }


def load_task(spec: TaskSpec, tokenizer: PreTrainedTokenizerBase) -> TaskData:
    """Load, subsample and tokenize one task's train/validation splits."""
    if spec.config_name:
        ds = load_dataset(spec.dataset, spec.config_name)
    else:
        ds = load_dataset(spec.dataset)

    train = ds["train"].select(range(min(spec.max_train_samples, len(ds["train"]))))
    split_key = "validation" if "validation" in ds else "test"
    eval_ds = ds[split_key].select(range(min(spec.max_eval_samples, len(ds[split_key]))))

    def tokenize(batch):
        texts = batch[spec.text_column]
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=spec.max_length,
            padding="max_length",
        )
        return {**enc, "label": batch[spec.label_column]}

    train_tok = train.map(tokenize, batched=True, remove_columns=set(train.column_names))
    eval_tok = eval_ds.map(tokenize, batched=True, remove_columns=set(eval_ds.column_names))

    train_tok = train_tok.with_format("torch")
    eval_tok = eval_tok.with_format("torch")

    return TaskData(
        spec=spec,
        train=TorchDataset(train_tok),
        eval=TorchDataset(eval_tok),
        num_labels=spec.num_labels,
    )
