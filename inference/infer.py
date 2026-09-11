#!/usr/bin/env python3
"""Batched greedy inference for Qwen3.5-9B (+ optional LoRA) on a FOCUS frame parquet.

Emits a CSV with a positional `row` column aligned to the ground-truth parquet,
so score.py can key by row (no id-duplication ambiguity, no remap step).
"""
from __future__ import annotations
import argparse, os, sys, time
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from focus_common import build_frame_index, resolve_frame, make_prompt, eos_ids, SYSTEM_S1

CLEAN_GT = "/home/ubuntu/orena/metadata/frames_test_true.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/home/ubuntu/orena/models/qwen3.5-9b")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--gt", default=CLEAN_GT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--max-pixels", type=int, default=1048576)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    gt = pd.read_parquet(a.gt)
    gt["row"] = range(len(gt))
    if a.limit:
        gt = gt.head(a.limit).reset_index(drop=True)
    print(f"rows: {len(gt)}  gt={a.gt}  adapter={a.adapter}", flush=True)

    idx = build_frame_index()
    paths, exacts = [], []
    for r in gt.itertuples(index=False):
        p, d, e = resolve_frame(idx, r.video, r.timestamp_start)
        paths.append(p); exacts.append(e)
    gt["frame_path"] = paths; gt["frame_exact"] = exacts
    print(f"frames resolved exactly: {sum(exacts)}/{len(gt)}", flush=True)

    proc = AutoProcessor.from_pretrained(
        a.base, padding_side="left",
        size={"longest_edge": a.max_pixels, "shortest_edge": 65536})
    model = AutoModelForImageTextToText.from_pretrained(
        a.base, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa")
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
        model = model.merge_and_unload()
        print(f"LoRA merged: {a.adapter}", flush=True)
    model.eval(); model.config.use_cache = True

    EOS = eos_ids(proc)
    print("eos_token_id ->", EOS, flush=True)

    prompts = [make_prompt(proc, r.question, SYSTEM_S1) for r in gt.itertuples(index=False)]
    print("=== EXAMPLE PROMPT ===\n" + repr(prompts[0]) + "\n=== END ===", flush=True)

    outs, over = [], 0
    t0 = time.time(); n = len(gt)
    for s in range(0, n, a.batch):
        chunk = gt.iloc[s:s + a.batch]
        imgs = [Image.open(r.frame_path).convert("RGB") for r in chunk.itertuples(index=False)]
        batch = proc(text=prompts[s:s + a.batch], images=imgs,
                     return_tensors="pt", padding=True).to("cuda:0")
        if s == 0:
            print(f"image size {imgs[0].size}  grid_thw {batch.get('image_grid_thw')[0].tolist()}  "
                  f"input_ids {tuple(batch['input_ids'].shape)}", flush=True)
        over += int((batch["input_ids"].shape[1] > 4096))
        with torch.inference_mode():
            gen = model.generate(**batch, max_new_tokens=a.max_new_tokens,
                                 do_sample=False, num_beams=1, eos_token_id=EOS,
                                 pad_token_id=proc.tokenizer.pad_token_id)
        new = gen[:, batch["input_ids"].shape[1]:]
        outs.extend(proc.tokenizer.batch_decode(new, skip_special_tokens=True))
        if (s // a.batch) % 50 == 0:
            el = time.time() - t0; done = min(s + a.batch, n)
            print(f"  {done}/{n}  {el:.0f}s  eta {el/max(1,done)*(n-done)/60:.1f}min", flush=True)
    el = time.time() - t0
    print(f"DONE {n} rows in {el/60:.2f} min ({n/el:.2f} rows/s, {1000*el/n:.0f} ms/q); "
          f"batches over 4096: {over}", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pd.DataFrame({"row": gt["row"].values, "id": gt["id"].astype(str).values,
                  "prediction": outs, "true_format": gt["answer_format"].values,
                  "answer": gt["answer"].values,
                  "frame_exact": gt["frame_exact"].values}).to_csv(a.out, index=False)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
