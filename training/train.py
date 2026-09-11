#!/usr/bin/env python3
"""From-scratch LoRA fine-tune of Qwen3.5-9B for the ORena SAVE FOCUS FRAME track.

The LoRA is initialised fresh (Gaussian init, ``init_lora_weights=True``).  It
covers every linear layer: the LLM attention (q/k/v/o), the LLM linear-attention
(in_proj_a/b/qkv/z, out_proj), the LLM MLP (gate/up/down), the vision tower
(qkv/proj/linear_fc1/fc2) and the merger (linear_fc1/fc2).  That is 716 tensors
= 205.06M trainable parameters.

Recipe:

    base model     Qwen/Qwen3.5-9B @ c202236235762e1c871ad0ccb60c8ee5ba337b9a
    LoRA           r=64, alpha=128, dropout=0.05, bias=none, all-linear,
                   task_type=CAUSAL_LM, init_lora_weights=True, rslora/DoRA off
    optimiser      adamw_torch_fused  lr=1e-4  wd=0.1  b1=0.9  b2=0.95  eps=1e-8
    schedule       cosine  warmup_ratio=0.03
    max_grad_norm  1.0
    dtype          bf16, gradient_checkpointing on, attn_impl=sdpa
    seed           43
    batch          effective 64   (--bs * --grad-accum)
    epochs         15
    max_length     4096, truncation, padding_side=right
    max_pixels     None = native resolution (no downscale)
    prompt         system + <image> + question, enable_thinking=False

Data: ``frames_train_true.parquet`` (built by data_prep/build_training_data.py),
one row per (frame, question) pair, 17,748 rows.  Loss is computed on the answer
tokens only (tail match).
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (AutoProcessor, AutoModelForImageTextToText, Trainer,
                          TrainingArguments, TrainerCallback)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from focus_common import build_frame_index, resolve_frame, make_prompt, SYSTEM_S1

# --- frozen snapshot of the base model --------------------------------------
BASE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
TARGET_MODULES = (
    r"^(model\.language_model(?=\.).*\.(in_proj_z|k_proj|up_proj|out_proj|"
    r"gate_proj|in_proj_a|down_proj|v_proj|o_proj|in_proj_b|q_proj|in_proj_qkv)|"
    r"(?!(model.visual.merger))model\.visual(?=\.).*\.(linear_fc2|attn.proj|qkv|"
    r"linear_fc1)|model\.visual\.merger(?=\.).*\.(linear_fc2|linear_fc1))$"
)
EXPECTED_TRAINABLE = 205_060_096   # 205.06M = 716 tensors (all-linear LoRA)

TRAIN_PARQUET = "/home/ubuntu/orena/metadata/frames_train_true.parquet"
BASE = "/home/ubuntu/orena/models/qwen3.5-9b"


class LiveLog(TrainerCallback):
    def __init__(self, path):
        self.path, self.t0 = path, None
        os.makedirs(os.path.dirname(path), exist_ok=True)
    def _w(self, m):
        el = 0 if self.t0 is None else time.time() - self.t0
        with open(self.path, "a") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] [{el/3600:5.2f}h] {m}\n"); f.flush()
    def on_train_begin(self, args, state, control, **kw):
        self.t0 = time.time(); self._w(f"START max_steps={state.max_steps}")
    def on_log(self, args, state, control, logs=None, **kw):
        if not logs: return
        el = time.time() - self.t0 if self.t0 else 0
        st, tot = state.global_step, max(1, state.max_steps)
        eta = el / st * (tot - st) / 3600 if st else 0
        self._w(f"step {st}/{tot} ({100*st/tot:5.1f}%) eta={eta:5.2f}h  " +
                " ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in logs.items() if k != "epoch"))
    def on_save(self, args, state, control, **kw):
        self._w(f"SAVED checkpoint-{state.global_step}")
    def on_train_end(self, args, state, control, **kw):
        self._w("TRAIN END")


def video_level_split(df: pd.DataFrame, frac: float, seed: int = 1234):
    """Hold out whole videos, stratified by procedure_type."""
    rng = np.random.RandomState(seed)
    val_vids = []
    for proc_t, g in df.groupby("procedure_type"):
        vids = sorted(g["video"].unique())
        k = max(1, int(round(frac * len(vids))))
        val_vids += list(rng.choice(vids, size=k, replace=False))
    val_vids = set(val_vids)
    return df[~df["video"].isin(val_vids)].reset_index(drop=True), \
           df[df["video"].isin(val_vids)].reset_index(drop=True), sorted(val_vids)


class FocusFrames(Dataset):
    def __init__(self, df, idx):
        recs = []
        miss = 0
        for r in df.itertuples(index=False):
            try:
                p, _, ex = resolve_frame(idx, r.video, r.timestamp_start)
            except KeyError:
                miss += 1; continue
            recs.append({"q": r.question, "a": str(r.answer), "img": p,
                         "fmt": r.answer_format})
        self.recs = recs
        self.missing = miss
    def __len__(self): return len(self.recs)
    def __getitem__(self, i): return self.recs[i]


class Collator:
    """Right-padded batch; loss on the ANSWER tokens only (tail match)."""
    def __init__(self, proc, max_len, verify_first=True):
        self.proc, self.max_len = proc, max_len
        self.verify = verify_first
        self.im_end = proc.tokenizer.convert_tokens_to_ids("<|im_end|>")
    def __call__(self, feats):
        imgs = [Image.open(f["img"]).convert("RGB") for f in feats]
        prompts = [make_prompt(self.proc, f["q"], SYSTEM_S1) for f in feats]
        targets = [f["a"] + "<|im_end|>" for f in feats]
        texts = [p + t for p, t in zip(prompts, targets)]
        batch = self.proc(text=texts, images=imgs, return_tensors="pt",
                          padding=True, truncation=True, max_length=self.max_len)
        tgt_ids = [self.proc.tokenizer(t, add_special_tokens=False)["input_ids"]
                   for t in targets]
        labels = batch["input_ids"].clone()
        am = batch["attention_mask"]
        for i in range(len(feats)):
            L = int(am[i].sum())
            k = len(tgt_ids[i])
            labels[i, :max(0, L - k)] = -100
            labels[i, L:] = -100
            if self.verify:
                got = batch["input_ids"][i, L - k:L].tolist()
                assert got == tgt_ids[i], (
                    f"target tokenisation mismatch\n got={got}\n want={tgt_ids[i]}\n"
                    f" text={texts[i][-120:]!r}")
        self.verify = False
        batch["labels"] = labels
        return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=BASE, help="local Qwen3.5-9B (must be the "
                    f"exact snapshot {BASE_REVISION})")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=15.0)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=16, help="eff batch = bs*accum (64)")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--max-pixels", type=int, default=None,
                    help="None = native (matches max_pixels=null); pass int to downscale")
    ap.add_argument("--val-frac", type=float, default=0.0,
                    help="0 = train on all 17,748 (faithful). >0 carves video-level val for eval_loss")
    ap.add_argument("--grad-ckpt", type=int, default=1)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--log", default=None)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--ignore-data-skip", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = a.log or os.path.join(a.out, "live.log")
    json.dump(vars(a), open(os.path.join(a.out, "run_args.json"), "w"), indent=2)

    df = pd.read_parquet(TRAIN_PARQUET)
    if a.val_frac > 0:
        tr_df, va_df, val_vids = video_level_split(df, a.val_frac, a.seed)
        json.dump(val_vids, open(os.path.join(a.out, "val_videos.json"), "w"), indent=2)
        print(f"train {len(tr_df)} rows | val {len(va_df)} rows", flush=True)
    else:
        tr_df, va_df = df, None
        print(f"train {len(tr_df)} rows | no val (faithful: train on all)", flush=True)

    idx = build_frame_index()
    tr = FocusFrames(tr_df, idx)
    va = FocusFrames(va_df, idx) if va_df is not None else None
    print(f"dataset: train {len(tr)} (missing {tr.missing})", flush=True)

    proc = AutoProcessor.from_pretrained(a.base, padding_side="right")
    if a.max_pixels is not None:
        proc.image_processor.size = {"longest_edge": a.max_pixels,
                                     "shortest_edge": 65536}

    model = AutoModelForImageTextToText.from_pretrained(
        a.base, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa")
    model.config.use_cache = False

    # Fresh LoRA, all linear layers, Gaussian init.
    from peft import LoraConfig, get_peft_model
    lc = LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0.05, bias="none",
        target_modules=TARGET_MODULES, task_type="CAUSAL_LM",
        init_lora_weights=True, use_rslora=False, use_dora=False,
        modules_to_save=[],
    )
    model = get_peft_model(model, lc)
    tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    print(f"fresh LoRA: trainable {tp/1e6:.2f}M / {tot/1e6:.0f}M "
          f"({100*tp/tot:.2f}%)", flush=True)
    if abs(tp - EXPECTED_TRAINABLE) > 1:
        print(f"WARNING: trainable {tp} != expected {EXPECTED_TRAINABLE}; "
              f"target_modules may not reproduce the 716-tensor layout", flush=True)

    if a.grad_ckpt:
        model.enable_input_require_grads()

    eff = a.bs * a.grad_accum
    spe = math.ceil(len(tr) / eff)
    total = a.max_steps if a.max_steps > 0 else int(spe * a.epochs)
    warmup = int(0.03 * total)   # warmup_ratio=0.03 (transformers 5.x dropped warmup_ratio)
    print(f"eff batch {eff} | steps/epoch {spe} | total {total} | warmup {warmup}",
          flush=True)

    targs = TrainingArguments(
        output_dir=a.out,
        max_steps=a.max_steps if a.max_steps > 0 else -1,
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.bs,
        gradient_accumulation_steps=a.grad_accum,
        per_device_eval_batch_size=a.bs,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        optim="adamw_torch_fused",
        bf16=True,
        logging_steps=5,
        save_strategy="steps", save_steps=a.save_steps, save_total_limit=8,
        eval_strategy="epoch" if va is not None else "no",
        gradient_checkpointing=bool(a.grad_ckpt),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=a.workers,
        dataloader_pin_memory=True,
        dataloader_drop_last=False,
        remove_unused_columns=False,
        report_to=[],
        seed=a.seed,
        label_names=["labels"],
        ignore_data_skip=bool(a.ignore_data_skip),
    )

    coll = Collator(proc, a.max_len)
    trainer = Trainer(model=model, args=targs, train_dataset=tr, eval_dataset=va,
                      data_collator=coll, callbacks=[LiveLog(log)])

    trainer.train(resume_from_checkpoint=a.resume_from or None)
    trainer.save_model(os.path.join(a.out, "final"))
    trainer.state.save_to_json(os.path.join(a.out, "trainer_state.json"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
