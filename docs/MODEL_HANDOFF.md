# Run the inference model

The [public Hugging Face model repository](https://huggingface.co/mitchaiet/crossdiffusion-gemma-3-12b/tree/v0.1.0) includes the Gemma 3 12B IT base checkpoint, matching correction LoRA, SHA-256 manifest, model card, Gemma terms, and required notices. It includes no training corpus or optimizer state. The app source is separate from the model weights.

On a CUDA workstation, install a compatible PyTorch build, Transformers with Gemma 3 support, PEFT, Accelerate, and SentencePiece. The verified Dell runtime uses PyTorch 2.11.0+cu130, Transformers 5.1.0, PEFT 0.18.1, and Accelerate 1.13.0 on an RTX PRO 6000 with 96 GB VRAM. This is a tested configuration, not a portability guarantee.

```bash
hf download mitchaiet/crossdiffusion-gemma-3-12b --revision v0.1.0 \
  --local-dir models/crossdiffusion-gemma-3-12b
cd models/crossdiffusion-gemma-3-12b
sha256sum -c SHA256SUMS
cd ../..
python -m training.serve \
  --base models/crossdiffusion-gemma-3-12b/base \
  --adapter models/crossdiffusion-gemma-3-12b/adapter \
  --host 127.0.0.1 --port 8011
```

On macOS, use `shasum -a 256 -c SHA256SUMS`. The model server and web app can run on different machines when the model URL is reachable. Start the app with `crossword web --diffusion-url http://127.0.0.1:8011`.

The Gemma base and adapter are subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms), including its incorporated Prohibited Use Policy. The source code's MIT license does not cover the model weights.
