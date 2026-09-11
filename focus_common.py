"""Shared frame resolution + prompt construction for the ORena FRAME track.

Prompt:
  system "You are a surgical assistant. Be precise and concise."
  user   <image> + question (with any literal '<image>' stripped)
  assistant generation prompt with enable_thinking=False, which the Qwen3.5
  chat template renders as '<think>\\n\\n</think>\\n\\n'.
"""
from __future__ import annotations
import os, re, glob

SYSTEM_S1 = "You are a surgical assistant. Be precise and concise."
FRAME_ROOT = "/home/ubuntu/orena/frames"


def code_to_sec(c: int) -> int:
    return (c // 10000) * 3600 + ((c // 100) % 100) * 60 + (c % 100)


def ts_to_sec(t: str) -> int:
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def build_frame_index(root: str = FRAME_ROOT):
    """video-dir-name -> (codes, secs, dir).  Scans every */<video>/ under root."""
    idx = {}
    for sub in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(sub):
            continue
        for v in os.listdir(sub):
            d = os.path.join(sub, v)
            if not os.path.isdir(d):
                continue
            files = [f for f in os.listdir(d) if f.endswith(".jpg")]
            if not files:
                continue
            codes = sorted(int(re.search(r"(\d+)", f).group(1)) for f in files)
            idx[v] = (codes, [code_to_sec(c) for c in codes], d)
    return idx


def resolve_frame(idx, video: str, ts: str):
    """-> (path, delta_seconds, exact_bool).  Falls back to nearest frame."""
    vid = re.sub(r"\.[A-Za-z0-9]+$", "", video)
    codes, secs, d = idx[vid]
    exact = os.path.join(d, "frame_" + ts.replace(":", "") + ".jpg")
    if os.path.isfile(exact):
        return exact, 0, True
    t = ts_to_sec(ts)
    i = min(range(len(secs)), key=lambda k: abs(secs[k] - t))
    return os.path.join(d, f"frame_{codes[i]:06d}.jpg"), secs[i] - t, False


def make_prompt(proc, question: str, system: str = SYSTEM_S1) -> str:
    q = str(question).replace("<image>", "").strip()
    msgs = []
    if system:
        msgs.append({"role": "system", "content": [{"type": "text", "text": system}]})
    msgs.append({"role": "user",
                 "content": [{"type": "image"}, {"type": "text", "text": q}]})
    return proc.apply_chat_template(msgs, tokenize=False,
                                    add_generation_prompt=True,
                                    enable_thinking=False)


def eos_ids(proc):
    """Qwen/Qwen3.5-9B ships no generation_config.json and config.text_config
    .eos_token_id is 248044 (<|endoftext|>) while the chat template closes turns
    with <|im_end|> (248046).  Generation never stops without both."""
    return sorted({proc.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                   proc.tokenizer.convert_tokens_to_ids("<|endoftext|>")})
