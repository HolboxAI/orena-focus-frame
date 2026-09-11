# External resources

The challenge requires a list of external resources used in this submission.
All of them were public before the challenge's external-resource cutoff of
2026-07-15.

## Base model

**Qwen/Qwen3.5-9B** — the pre-trained vision-language model we fine-tune.

- Source: https://huggingface.co/Qwen/Qwen3.5-9B
- Revision: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- License: Apache-2.0

## Training data

The two public FOCUS FRAME datasets, accessed through the official
`orena-focus` client:

- **HeICO** — `orena-dkfz/heico-focus-vqa`
- **LapChole** — `orena-dkfz/lapchole-focus-vqa`

Both are distributed by the challenge organisers for this task.

## Software

- **orena-focus** (`orena-focus==0.3.5`) — the official challenge client
  (dataset loading, frame extraction, evaluation primitives). MIT license.
- **transformers** (`5.15.1`), **peft** (`0.20.0`), **accelerate** (`1.14.0`),
  **PyTorch**, and the standard scientific Python stack.

## Method

The LoRA configuration, the training recipe, the prompt, and the evaluation are
our own. The model is trained from scratch on the public datasets above. No
other team's weights or code were used.
