"""Debug: check eval sanity before/after training on IMDb only."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from cf_lab.config import ExperimentConfig, TaskSpec, TrainingHyperparams, MitigationConfig
from cf_lab.data import load_task
from cf_lab.model import load_model_and_tokenizer
from cf_lab.training import fine_tune_phase, make_optimizer_and_scheduler
from torch.utils.data import DataLoader

cfg = ExperimentConfig(
    tasks=[
        TaskSpec(name="IMDb", dataset="stanfordnlp/imdb", max_train_samples=400, max_eval_samples=200),
        TaskSpec(name="SST-2", dataset="nyu-mll/glue", config_name="sst2",
                 text_column="sentence", max_train_samples=400, max_eval_samples=200),
    ],
    model_name="google/bert_uncased_L-2_H-128_A-2",
    training=TrainingHyperparams(epochs=1, batch_size=8, eval_every_steps=5, warmup_steps=5),
    mitigation=MitigationConfig(kind="baseline"),
    device="cpu",
)

model, tokenizer = load_model_and_tokenizer(cfg.model_name, 2, "cpu")
imdb = load_task(cfg.tasks[0], tokenizer)
sst2 = load_task(cfg.tasks[1], tokenizer)

dl_imdb = DataLoader(imdb.train, batch_size=8, shuffle=True)

def eval_ds(td, name):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for b in DataLoader(td.eval, batch_size=8):
            logits = model(b["input_ids"], attention_mask=b["attention_mask"]).logits
            correct += (logits.argmax(-1) == b["labels"]).sum().item()
            total += b["labels"].size(0)
    print(f"{name} acc (untrained): {correct/total:.3f}")
    return correct / total

eval_ds(imdb, "IMDb")
eval_ds(sst2, "SST-2")

optimizer, sched = make_optimizer_and_scheduler(model, cfg.resolved_training(), len(dl_imdb))
logs, step, _ = fine_tune_phase(
    model=model, optimizer=optimizer, scheduler=sched, dataloader=dl_imdb,
    phase_name="IMDb", task_index=0, num_epochs=1, device="cpu",
    eval_fn=lambda: {
        "IMDb": eval_ds(imdb, "IMDb"), "SST-2": eval_ds(sst2, "SST-2"),
    },
    eval_every_steps=5, step_start=0, ewc_list=[], mitigation=cfg.mitigation,
    replay_buffer=None,
)
print("last train loss:", logs[-1].loss, "train acc:", logs[-1].train_acc)
print("last eval accs:", logs[-1].task_accs)

# inspect raw samples
sample = imdb.train[0]
print("IMDb sample keys:", list(sample.keys()))
print("label:", sample["labels"], "input len:", len(sample["input_ids"]))
print("decoded:", tokenizer.decode(sample["input_ids"][:40]))
