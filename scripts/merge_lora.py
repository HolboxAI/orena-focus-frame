#!/usr/bin/env python3
"""Merge the LoRA adapter into the base weights and ship a plain transformers model.

The submission container ships *merged* weights in ``resources/model/`` — no
adapter is applied at runtime and nothing is fetched from a hub.  This script
produces that directory:

  1. loads the base Qwen3.5-9B in bf16,
  2. applies the LoRA adapter and merges it back into the base weights,
  3. saves the merged model, then copies the processor/tokenizer files and pins
     the two EOS ids into ``generation_config.json``.

Run on a GPU box (the merge needs the base model in memory):

    python merge_lora.py --base /path/to/qwen3.5-9b \
        --adapter /path/to/adapter --out resources/model
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText

# Qwen3.5-9B ships no generation_config.json and its config.text_config EOS is
# <|endoftext|> (248044) while the chat template closes a turn with <|im_end|>
# (248046).  Generation never stops without both.
EOS_IDS = [248044, 248046]

# Files to mirror from the base snapshot so the merged model is self-contained.
TOKENIZER_FILES = [
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="local Qwen3.5-9B snapshot")
    ap.add_argument("--adapter", required=True, help="LoRA adapter directory")
    ap.add_argument("--out", required=True, help="merged output directory")
    a = ap.parse_args()

    base = Path(a.base)
    adapter = Path(a.adapter)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading base from {base}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        base, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa")

    print(f"merging adapter from {adapter}", flush=True)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()

    print(f"saving merged model to {out}", flush=True)
    model.save_pretrained(out)

    # Mirror the processor/tokenizer files, then fix the EOS ids.
    for f in TOKENIZER_FILES:
        src = base / f
        if src.exists():
            shutil.copy(src, out / f)
    (out / "processor_config.json").unlink(missing_ok=True)

    gen = out / "generation_config.json"
    if gen.exists():
        cfg = json.loads(gen.read_text())
        cfg["eos_token_id"] = EOS_IDS
        gen.write_text(json.dumps(cfg, indent=2))

    print(f"DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
