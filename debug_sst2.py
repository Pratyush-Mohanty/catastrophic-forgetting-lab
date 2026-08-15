"""Debug: inspect SST-2 tokenized samples and test training on SST-2 from scratch."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from cf_lab.config import ExperimentConfig, TaskSpec, TrainingHyperparams, MitigationConfig
from cf_lab.data import load_task
from cf_lab.model import load_model_and_tokenizer
from cf_lab.training import fine_tune_phase, make_optimizer_and_scheduler

cfg = ExperimentConfig(
    tasks=[
        TaskSpec(name="SST-2", dataset="nyu-mll/glue", config_name="sst2",
                 text_column="sentence", max_train_samples=300, max_eval_samples=150),
    ],
    model_name="google/bert_uncased_L-2_H-128_A-2",
    training=TrainingHyperparams(epochs=1, batch_size=8, eval_every_steps=10, warmup_steps=5),
    mitigation=MitigationConfig(kind="baseline"),
    device="cpu",
)

model, tokenizer = load_model_and_tokenizer(cfg.model_name, 2, "cpu")
sst2 = load_task(cfg.tasks[0], tokenizer)

print("columns:", sst2.train[0].keys())
for i in range(3):
    s = sst2.train[i]
    print(f"label={s['labels']} text={tokenizer.decode(s['input_ids'][:40])!r}")

dl = DataLoader(sst2.train, batch_size=8, shuffle=True)

def ev():
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for b in DataLoader(sst2.eval, batch_size=8):
            logits = model(b["input_ids"], attention_mask=b["attention_mask"]).logits
            correct += (logits.argmax(-1) == b["labels"]).sum().item()
            total += b["labels"].size(0)
    model.train()
    return {"SST-2": correct / max(total, 1)}

print("eval before:", ev())
optimizer, sched = make_optimizer_and_scheduler(model, cfg.resolved_training(), len(dl))
logs, step, _ = fine_tune_phase(
    model=model, optimizer=optimizer, scheduler=sched, dataloader=dl,
    phase_name="SST-2", task_index=0, num_epochs=1, device="cpu",
    eval_fn=ev, eval_every_steps=10, step_start=0, ewc_list=[],
    mitigation=cfg.mitigation, replay_buffer=None,
)
for l in logs:
    print(f"step={l.step:>3} loss={l.loss:6.3f} train_acc={l.train_acc:5.3f} accs={ {k: round(v,3) for k,v in l.task_accs.items()} }")
