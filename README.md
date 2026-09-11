# orena-focus-frame

A from-scratch LoRA fine-tune of Qwen3.5-9B for the **FRAME** track of the
[ORena SAVE FOCUS challenge](https://orena-focus-challenge.org/).

The FRAME track asks one question about one still image from a surgical video.
The model must answer in a strict format. The answer is a class name, a count,
a yes/no, a percentage, or a short phrase.

This repository holds the full pipeline: data preparation, training, inference,
evaluation, and the Docker container for submission.

## Method

We fine-tune **Qwen/Qwen3.5-9B** with LoRA. The adapter covers every linear
layer of the language model, the vision tower, and the merger. That is 716
tensors, 205.06M trainable parameters.

The recipe:

| Setting | Value |
|---|---|
| Base model | Qwen/Qwen3.5-9B |
| LoRA | r=64, alpha=128, dropout=0.05, all-linear |
| Optimiser | adamw_torch_fused, lr 1e-4, wd 0.1 |
| Schedule | cosine, warmup 3% |
| Gradient clip | 1.0 |
| Precision | bf16, SDPA, gradient checkpointing |
| Seed | 43 |
| Effective batch | 64 |
| Epochs | 15 |
| Max length | 4096 |
| Image resolution | native (no downscale) |

Training data: the two public FOCUS FRAME datasets, HeICO and LapChole, merged.
17,748 training rows: Proctocolectomy 6,000, Rectal Resection 6,000, and
Laparoscopic Cholecystectomy 5,748.

The loss covers answer tokens only (tail match). The model sees the question and
the frame. It does not see the foreign-object definitions at inference time.

## Results

Local evaluation on the held-out test set, using the 4-bucket proxy mean
(object_recognition × aggregation × out-of-distribution proxy):

| Metric | Score |
|---|---|
| 4-bucket proxy mean | 65.94% |
| Deterministic micro accuracy | 68.0% |
| fo_class | 75.4% |
| binary | 82.8% |
| number | 52.6% |

These are local numbers. The leaderboard uses a different bucket composition and
a judge for the open-ended formats. The local proxy does not equal the
leaderboard score.

## Repository layout

```
focus_common.py                 shared frame resolution + prompt construction
data_prep/build_training_data.py  download data, build parquets, extract frames
training/train.py               single-GPU LoRA fine-tune
training/train_ddp.py           multi-GPU (DDP) LoRA fine-tune
inference/infer.py              batched inference over a test parquet
evaluation/score.py             format-aware scorer (4-bucket proxy)
evaluation/report.py            per-format and counting report
scripts/merge_lora.py           merge the adapter into base weights
submission/frame-algorithm/     Docker container for submission
```

## Quick start

### 1. Prepare the data

```bash
export FOCUS_ROOT_DIR=/data/focus
pip install orena-focus==0.3.5 pandas pyarrow pillow numpy

python data_prep/build_training_data.py --out /data/orena
```

This downloads the HeICO and LapChole videos, builds
`frames_train_true.parquet` and `frames_test_true.parquet`, and extracts one
frame per second per video, named by timestamp.

Download the base model once:

```bash
hf download Qwen/Qwen3.5-9B --local-dir /data/orena/models/qwen3.5-9b
```

### 2. Train

Single GPU:

```bash
python training/train.py \
    --base /data/orena/models/qwen3.5-9b \
    --out /data/orena/out/frame_15ep \
    --bs 4 --grad-accum 16 --epochs 15
```

Multiple GPUs (effective batch 64 across the world):

```bash
torchrun --nproc_per_node=4 --master_port=29500 training/train_ddp.py \
    --base /data/orena/models/qwen3.5-9b \
    --out /data/orena/out/frame_15ep_ddp \
    --bs 4 --grad-accum 4 --epochs 15
```

The `--base`, parquet, and frame paths default to the EC2 training layout in the
scripts. Point them at your own directories.

### 3. Inference and evaluation

```bash
python inference/infer.py \
    --base /data/orena/models/qwen3.5-9b \
    --adapter /data/orena/out/frame_15ep/final \
    --out /data/orena/preds/frame_15ep.csv

python evaluation/score.py /data/orena/preds/frame_15ep.csv \
    --gt /data/orena/metadata/frames_test_true.parquet --normalise --dump /tmp/dump.csv

python evaluation/report.py /tmp/dump.csv --label frame_15ep
```

### 4. Build the submission container

```bash
# 1. merge the adapter into base weights (on a GPU box)
python scripts/merge_lora.py \
    --base /data/orena/models/qwen3.5-9b \
    --adapter /data/orena/out/frame_15ep/final \
    --out submission/frame-algorithm/resources/model

# 2. build the image
cd submission/frame-algorithm && ./do_build.sh

# 3. test against the stub input
./do_test.sh

# 4. save the image for upload
./do_save.sh
```

The container reads `/input/request.json` and `/input/frames/<qID>.png`, writes
`/output/answer.json`, and runs with `--network none` (no internet at inference).

## License

Apache-2.0. See [LICENSE](LICENSE).

## External resources

See [DISCLOSURE.md](DISCLOSURE.md).
