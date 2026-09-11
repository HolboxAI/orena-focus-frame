#!/usr/bin/env python3
"""Build the FRAME-track parquets and extract timestamp-named frames.

Downloads the two public ORena FOCUS FRAME datasets (HeICO + LapChole) and
produces the inputs the training and evaluation scripts consume:

  1. ``frames_train_true.parquet``  — merged train QA (one row per question)
  2. ``frames_test_true.parquet``   — merged test QA
  3. one JPEG per second per referenced video, named ``frame_HHMMSS.jpg``

The frame name encodes the procedure timestamp: a frame shown at 04:47:11 is
written as ``frame_044711.jpg``.  That is the convention ``focus_common.py``
reads, so training and evaluation resolve every question to an exact frame.

Videos and QA annotations come from the public HuggingFace datasets that the
official ``orena-focus`` client exposes:

    orena-dkfz/heico-focus-vqa      (25 fps)
    orena-dkfz/lapchole-focus-vqa   (30 fps)

Run from a box with disk for the videos and, ideally, a GPU is NOT required
(extraction is CPU-only).

Example::

    export FOCUS_ROOT_DIR=/data/focus
    python build_training_data.py --out /data/orena
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# The official challenge client.  `focus` exposes the dataset ids, the frame
# extraction preprocessor, and the config that points at ``FOCUS_ROOT_DIR``.
from focus import download as focus_download
from focus.config import DATASET_BASE_FPS, FOCUS_DATASETS

DATASETS = ["heico", "lapchole"]
TRACK_CONFIG = "frame"          # HuggingFace config name for the FRAME track


# ── timestamps ----------------------------------------------------------------
def ts_to_seconds(ts: str) -> int:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def seconds_to_name(sec: int) -> str:
    """``sec`` (seconds) -> ``frame_HHMMSS.jpg``."""
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"frame_{h:02d}{m:02d}{s:02d}.jpg"


# ── QA parquets ---------------------------------------------------------------
def load_qa(dataset: str, split: str) -> pd.DataFrame:
    """Load the raw FRAME-track QA rows for one dataset/split from HuggingFace."""
    from datasets import load_dataset

    repo_id = FOCUS_DATASETS[dataset]
    hf_split = "train+test" if split == "all" else split
    rows = load_dataset(repo_id, TRACK_CONFIG, split=hf_split)
    df = pd.DataFrame([dict(r) for r in rows])
    df["source"] = dataset
    if "track" not in df.columns:
        df["track"] = "frame"
    if "ood" not in df.columns:
        df["ood"] = False
    return df


def merge_qa(split: str) -> pd.DataFrame:
    """Merge the given split of both datasets into one DataFrame."""
    frames = [load_qa(d, split) for d in DATASETS]
    df = pd.concat(frames, ignore_index=True)
    # Keep the columns our scripts read, in a stable order.
    wanted = ["source", "id", "video", "procedure_type", "question", "answer",
              "answer_format", "track", "generation", "clinical_relevance", "ood",
              "timestamp_start", "timestamp_end", "primary_capability",
              "secondary_capabilities"]
    for c in wanted:
        if c not in df.columns:
            df[c] = "" if c in ("generation",) else ([] if c == "secondary_capabilities" else False)
    return df[wanted]


# ── frame extraction ----------------------------------------------------------
def extract_frames(videos_root: Path, frames_root: Path, videos: set[str],
                   fps_override: dict[str, int] | None = None) -> None:
    """Extract one frame per second per referenced video, named by timestamp.

    Frames land under ``<frames_root>/<dataset>/<video_stem>/frame_HHMMSS.jpg``,
    matching the two-level layout ``focus_common.build_frame_index`` scans.
    ``video_stem`` is the video file name without its extension.
    """
    import cv2
    import decord

    decord.bridge.set_bridge("native")
    fps_map = fps_override or DATASET_BASE_FPS
    wanted_stems = {Path(v).stem for v in videos}

    for dataset in DATASETS:
        videos_dir = videos_root / dataset / "videos"
        if not videos_dir.is_dir():
            continue
        fps = int(fps_map.get(dataset, 25))
        for vpath in sorted(videos_dir.iterdir()):
            if not vpath.is_file():
                continue
            if vpath.stem not in wanted_stems:
                continue
            out_dir = frames_root / dataset / vpath.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            if any(out_dir.iterdir()):
                continue                       # already extracted
            print(f"[frames] {dataset}/{vpath.stem} @ {fps} fps", flush=True)
            vr = decord.VideoReader(str(vpath), ctx=decord.cpu(0), num_threads=1)
            n_sec = len(vr) // fps
            idxs = list(range(0, n_sec * fps, fps))
            batch = 64
            for i in range(0, len(idxs), batch):
                chunk = idxs[i:i + batch]
                imgs = vr.get_batch(chunk).asnumpy()
                for k, img in enumerate(imgs):
                    sec = chunk[k] // fps
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(out_dir / seconds_to_name(sec)), img,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 95])


# ── main ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="where to write the parquets")
    ap.add_argument("--skip-download", action="store_true",
                    help="skip the video download (already on disk)")
    ap.add_argument("--skip-frames", action="store_true",
                    help="skip frame extraction")
    ap.add_argument("--frames-root", default=None,
                    help="where frames land (default: <out>/frames)")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    videos_root = Path(os.environ["FOCUS_ROOT_DIR"])
    frames_root = Path(a.frames_root) if a.frames_root else out / "frames"

    # 1. download videos for both datasets.
    if not a.skip_download:
        for d in DATASETS:
            print(f"[download] {d}", flush=True)
            focus_download(d)

    # 2. build the merged QA parquets.
    for split, name in (("train", "frames_train_true.parquet"),
                        ("test", "frames_test_true.parquet")):
        df = merge_qa(split)
        p = out / name
        df.to_parquet(p, index=False)
        print(f"[parquet] {name}: {len(df)} rows", flush=True)

    # 3. extract frames for every video referenced by train + test.
    if not a.skip_frames:
        train = pd.read_parquet(out / "frames_train_true.parquet")
        test = pd.read_parquet(out / "frames_test_true.parquet")
        videos = set(train["video"]) | set(test["video"])
        print(f"[frames] {len(videos)} unique videos -> {frames_root}", flush=True)
        extract_frames(videos_root, frames_root, videos)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
