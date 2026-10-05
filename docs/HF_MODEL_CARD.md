---
license: gemma
base_model: google/gemma-3-12b-it
library_name: transformers
tags:
  - crossword
  - gemma
  - lora
  - diffusion
---

# CrossDiffusion Gemma 3 12B

This repository contains the inference weights for CrossDiffusion's crossword
first draft: the Gemma 3 12B IT base checkpoint in `base/` and its matching
CrossDiffusion correction LoRA in `adapter/`. The adapter is a modified Gemma
model derivative. The base files were copied without conversion from the Dell T2
deployment; `SHA256SUMS` records the exact packaged files.

The app reads a crossword image or PDF, checks the detected grid and clues, then
asks this model to fill a first draft. The model predicts masked letters over
the whole grid in repeated passes and reveals confident cells. An optional,
separate LLM agent can review and revise the draft. This repository supplies
the first-draft weights, not the app or the review LLM.

## Files

| Path | Contents |
| --- | --- |
| `base/` | Gemma 3 12B IT weights, configuration, tokenizer, and processor |
| `adapter/` | CrossDiffusion correction LoRA and matching tokenizer files |
| `SHA256SUMS` | SHA-256 hashes for all 20 packaged model files |
| `Notice.txt`, `GEMMA_TERMS_OF_USE.html`, `GEMMA_PROHIBITED_USE_POLICY.html` | Terms and notices that accompany the weights |

The base checkpoint is about 23 GB and the adapter about 1 GB. Download both;
the adapter alone is insufficient to run this model.

## Download and check

Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli),
then download the tagged package:

```bash
hf download mitchaiet/crossdiffusion-gemma-3-12b --revision v0.1.0 \
  --local-dir models/crossdiffusion-gemma-3-12b
cd models/crossdiffusion-gemma-3-12b
sha256sum -c SHA256SUMS
```

On macOS, use `shasum -a 256 -c SHA256SUMS`. The model repository is public; a
Hugging Face login is not required for this download.

## Serve with the CrossDiffusion app

The model server is part of the separate CrossDiffusion application source.
The [GitHub source repository](https://github.com/mitchaiet/crossdiffusion) is public. On a CUDA
workstation, install a compatible PyTorch build, Transformers with Gemma 3
support, PEFT, Accelerate, and SentencePiece. Then, from the app repository:

```bash
python -m training.serve \
  --base /path/to/models/crossdiffusion-gemma-3-12b/base \
  --adapter /path/to/models/crossdiffusion-gemma-3-12b/adapter \
  --host 127.0.0.1 --port 8011
crossword web --diffusion-url http://127.0.0.1:8011
```

The verified Dell deployment uses PyTorch 2.11.0+cu130, Transformers 5.1.0,
PEFT 0.18.1, and Accelerate 1.13.0 on an RTX PRO 6000 with 96 GB VRAM. Other
GPU setups need their own compatibility check. The 12B checkpoint does not run
in the project's current Apple MPS loader on a 16 GB Mac.

Local OCR and PDF text extraction in the app do not need Nebius. Vision
fallback, open-ended chat, and final LLM review need a separate Nebius API key;
no key is included here.

## Evaluation and limitations

This package preserves the inference checkpoint, not its original training
corpus, optimizer state, or a held-out evaluation report for this 12B adapter.
File-hash verification establishes that the published files match the Dell
copy; it does not establish crossword accuracy. The app's
[evaluation methodology](https://github.com/mitchaiet/crossdiffusion/blob/master/docs/EVALUATION.md)
defines letter and word accuracy, full-puzzle solve rate, fill rate, confidence
calibration, time, and cost. Those current-model results still need a separate
held-out run. OCR mistakes and incorrect clues can also affect the final solve.

## Terms and attribution

The base weights and modified adapter are subject to the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms), including the Section
3.2 use restrictions and incorporated
[Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).
Copies accompany this repository as `GEMMA_TERMS_OF_USE.html` and
`GEMMA_PROHIBITED_USE_POLICY.html`; the required `Notice.txt` is included.
Recipients must follow those terms when using or redistributing the model.
The MIT license of the application source does not apply to these weights.
CrossDiffusion is not affiliated with or endorsed by Google.
