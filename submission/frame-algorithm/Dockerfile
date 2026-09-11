FROM --platform=linux/amd64 pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime AS frame-algorithm-amd64

# ── WHAT THIS CONTAINER SHIPS ─────────────────────────────────────────────────
# Qwen3.5-9B in bf16 with the from-scratch LoRA already MERGED into the base
# weights, as plain transformers weights in resources/model/. Nothing is loaded
# from a hub at runtime and no adapter is applied — there is no network to
# reach and no peft at inference time.

ENV PYTHONUNBUFFERED=1

# ── System packages (root) ────────────────────────────────────────────────────
# libgl1 / libglib2.0-0 / libxcb1 -> pillow / opencv-python-headless
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libxcb1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/

RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

# ── Guard: did pip swap torch out from under us? ──────────────────────────────
# The FRAME track runs on ml.g6e.xlarge (L40S, sm_89). The base image's
# torch is a CUDA 12.8 build; any requirements.txt change that pulls a different
# torch must fail here, loudly, not at inference.
RUN python -c "import torch; \
flags = torch._C._cuda_getArchFlags() or ''; \
cuda = torch.version.cuda or ''; \
print('torch', torch.__version__, '| cuda', cuda, '|', flags); \
assert torch.version.cuda.startswith('12.8'), 'torch was replaced by a CUDA ' + cuda + ' build'; \
assert 'sm_86' in flags or 'sm_89' in flags, 'no L40S-capable kernels: ' + repr(flags)"

# ── Guard: the exact stack that produced the predictions ──────────────────────
RUN python -c "import transformers, accelerate; \
print('transformers', transformers.__version__, '| accelerate', accelerate.__version__); \
assert transformers.__version__ == '5.15.1', transformers.__version__; \
from transformers import AutoModelForImageTextToText, AutoProcessor"

# ── Weights: the merged model, tokenizer and processor all in one directory ──
COPY --chown=user:user resources/model/ /opt/app/resources/model/

COPY --chown=user:user inference.py /opt/app/

ENTRYPOINT ["python", "inference.py"]
