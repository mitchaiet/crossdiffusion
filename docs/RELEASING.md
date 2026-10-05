# Public release

This repository is a clean source snapshot of CrossDiffusion. It contains no puzzle examples, answer keys, user uploads, training data, screenshots, or recordings of a puzzle. Its Git history starts with the cleaned export; the original development repository remains private.

## Build

With Python 3.11 or later and the `build` package installed:

```bash
python -m build
```

The wheel contains the app, CLI, and web assets. It contains no sample-puzzle files. The source archive contains the training and evaluation code plus this documentation. The complete inference checkpoint is in the [public model repository](https://huggingface.co/mitchaiet/crossdiffusion-gemma-3-12b/tree/v0.1.0).

The release also includes a 60-second process video made with Remotion. It shows only an abstract grid and contains no puzzle content. Render it from `video/` with `npm ci && npm run render`.

Verify release downloads against the `SHA256SUMS` attached to the GitHub release. Install the wheel in a fresh environment and check `crossword --help`, the web home page, and an empty `/api/demos` response. A real first draft requires the separate GPU model server; optional LLM features require a Nebius API key.

## Evidence boundary

Internal source tests passed before export. The public release excludes their puzzle fixtures. No held-out accuracy result for the current 12B model is claimed; see [the evaluation protocol](EVALUATION.md).
