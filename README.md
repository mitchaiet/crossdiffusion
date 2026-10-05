# CrossDiffusion

CrossDiffusion is a crossword-solving app with an editable puzzle view, an iterative masked-letter first draft, and an optional LLM review. It accepts an image or selected PDF pages, reads the grid and clues, checks their numbering, and lets you correct the transcription before solving.

This public source release contains **no bundled puzzles, clues, answer keys, example images, user uploads, or training data**. Supply your own puzzle. The inference weights are in the [public model repository](https://huggingface.co/mitchaiet/crossdiffusion-gemma-3-12b).

The [one-minute process video](video/README.md) uses an abstract grid and no puzzle content. Its editable Remotion source and narration are in `video/`.

## Install the app

Use Python 3.11 or later:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Install Tesseract for fast local clue OCR (`brew install tesseract` on macOS or `sudo apt install tesseract-ocr` on Ubuntu). Selectable PDF text is read locally. If the clue numbers fail to match the grid, the app can use a Nebius vision model as fallback. Open-ended chat and the final LLM review also use Nebius; set `NEBIUS_API_KEY` in a local `.env` file if you want those features. Never commit that file.

## Download and serve the first-draft model

The model needs a CUDA workstation with enough memory. The verified deployment uses an RTX PRO 6000 with 96 GB VRAM; other GPU environments need their own compatibility check. Follow [the model handoff](docs/MODEL_HANDOFF.md) to download and verify the Gemma 3 12B base weights and matching correction adapter, install GPU dependencies, and start `training.serve` on port 8011.

Then start the web app:

```bash
crossword web --diffusion-url http://127.0.0.1:8011
```

Open `http://127.0.0.1:8000`. The web app and GPU model server may run on separate machines if the model URL is reachable.

## Solve a puzzle

1. Upload an image or PDF. For a PDF, select every page containing the grid or clues, and leave answer-key pages unselected.
2. Type **read** or choose **Read the puzzle**. Check the extracted grid and clues and correct any mistakes.
3. Choose **Fill first draft**. CrossDiffusion repeatedly predicts hidden letters and fills the most confident cells.
4. If a Nebius key is configured, choose **Recheck with LLM**. The agent compares draft answers with clues and crossings and revises weak entries. The Thinking and Results views remain available in the conversation.

The CLI accepts your own structured crossword JSON or unlocked `.puz` files. The JSON loader and solver entrypoints are in `crossword_agent/`. This release has no included puzzle files or answer keys; grading requires an answer key you supply separately.

## Method and limits

The first-draft model is a Gemma 3 12B IT base checkpoint plus a correction LoRA. The app reads clues with local OCR or selectable PDF text when possible; uncertain extraction can use vision fallback. The grid and clues remain editable. The optional LLM agent proposes answers, propagates crossing constraints, and repairs conflicts.

[Evaluation methodology](docs/EVALUATION.md) defines letter and word accuracy, full-puzzle solve rate, extraction accuracy, latency, and cost for a held-out puzzle set. No held-out result for this exact 12B checkpoint is included in the public release. The original training corpus and optimizer state are not included.

Application code is MIT licensed. The model weights are governed by the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and accompanying notices in the model repository; the MIT license does not cover them.
