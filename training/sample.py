"""Iterative unmasking sampler for the CrossDiffusion crossword filler.

Start from a canvas where every unknown white cell is masked. Each step, predict all
masked cells in parallel, reveal the most confident fraction, and repeat. Known letters
(from the puzzle, or committed by the LLM agent in the hybrid) stay fixed throughout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from training.serialize import CanvasVocab, canvas_ids, prefix_lm_mask, prompt_text


@dataclass
class FillResult:
    letters: dict  # cell -> letter
    confidence: dict  # cell -> probability at the time the letter was revealed (1.0 if known)
    history: list[dict]
    snapshots: list[dict] | None = None  # canvas (cell -> letter | None) after each step


class DiffusionFiller:
    def __init__(self, model, tok, vocab: CanvasVocab, device: str) -> None:
        self.model, self.tok, self.vocab, self.device = model, tok, vocab, device
        self.letter_ids = torch.tensor(vocab.letter_ids, device=device)
        self.model.eval()

    @classmethod
    def load(cls, base: str, adapter: str | None = None, device: str | None = None, attn: str = "eager",
             init_adapter: str | None = None) -> DiffusionFiller:
        """`init_adapter` is merged into the base first: a run trained with --init-adapter X produced
        an adapter that only makes sense on top of X's merged weights, so serve it the same way."""
        from transformers import AutoTokenizer

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(adapter or base)
        from training.train_diffusion import load_base_model

        model = load_base_model(base, torch.bfloat16 if device == "cuda" else torch.float32, attn, device)
        for path in (init_adapter, adapter):
            if path:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, path).merge_and_unload()
        return cls(model, tok, CanvasVocab.from_tokenizer(tok), device)

    @torch.no_grad()
    def _letter_probs(self, prompt_ids: list[int], puzzle, letters: dict, temperature: float):
        cids, cells = canvas_ids(puzzle, self.vocab, letters)
        ids = torch.tensor([prompt_ids + cids], device=self.device)
        am = prefix_lm_mask(ids.shape[1], len(prompt_ids), device=self.device,
                            dtype=next(self.model.parameters()).dtype)
        logits = self.model(input_ids=ids, attention_mask=am).logits[0, len(prompt_ids):, :]
        logits = logits[:, self.letter_ids].float()
        if temperature > 0:
            logits = logits / temperature
        return torch.softmax(logits, dim=-1), cells

    @torch.no_grad()
    def fill(self, puzzle, known: dict | None = None, steps: int = 12, temperature: float = 0.0,
             record: bool = False, remask_frac: float = 0.0, samples: int = 1) -> FillResult:
        """Iterative unmasking. Two optional refinements:

        remask_frac  after each step (except the last two), send this fraction of the
                     lowest-confidence revealed cells back to masked so they are re-predicted
                     with more context. Early wrong letters can then be corrected.
        samples      run this many stochastic fills (temperature > 0) and vote per cell;
                     confidence becomes the vote share. Snapshots/history come from the first run.
        """
        if samples > 1:
            temp = temperature if temperature > 0 else 0.7
            runs = [self.fill(puzzle, known, steps, temp, record and i == 0, remask_frac, 1) for i in range(samples)]
            votes: dict = {}
            for r in runs:
                for c, ch in r.letters.items():
                    if ch:
                        votes.setdefault(c, {}).setdefault(ch, 0)
                        votes[c][ch] += 1
            letters = {c: (max(v.items(), key=lambda kv: kv[1])[0] if v else None) for c, v in votes.items()}
            for c in puzzle.white_cells:
                letters.setdefault(c, None)
            confidence = {c: (max(v.values()) / samples if v else 0.0) for c, v in votes.items()}
            for c, ch in (known or {}).items():
                if ch:
                    letters[c], confidence[c] = ch, 1.0
            hist = runs[0].history + [{"step": "vote", "samples": samples, "temperature": temp}]
            return FillResult(letters, confidence, hist, runs[0].snapshots)

        prompt_ids = self.tok.encode(prompt_text(puzzle), add_special_tokens=True)
        fixed = {c for c, ch in (known or {}).items() if ch}
        letters = {c: None for c in puzzle.white_cells}
        confidence: dict = {}
        for c in fixed:
            letters[c] = known[c]
            confidence[c] = 1.0
        history: list[dict] = []
        snapshots: list[dict] = [] if record else None
        for step in range(steps):
            masked = [c for c, v in letters.items() if v is None]
            if not masked:
                break
            probs, cells = self._letter_probs(prompt_ids, puzzle, letters, temperature)
            pos_of = {cell: i for i, cell in enumerate(cells) if cell is not None}
            preds = []
            for cell in masked:
                p = probs[pos_of[cell]]
                if temperature > 0:
                    idx = torch.multinomial(p, 1).item()
                else:
                    idx = int(p.argmax())
                preds.append((float(p[idx]), cell, chr(65 + idx)))
            preds.sort(reverse=True)
            n_reveal = math.ceil(len(masked) / (steps - step))
            revealed = preds[:n_reveal]
            for conf, cell, ch in revealed:
                letters[cell] = ch
                confidence[cell] = conf
            n_remasked = 0
            if remask_frac > 0 and step < steps - 2:
                revealed_cells = [c for c, v in letters.items() if v is not None and c not in fixed]
                n_remasked = int(remask_frac * len(revealed_cells))
                for c in sorted(revealed_cells, key=lambda c: confidence.get(c, 0.0))[:n_remasked]:
                    letters[c] = None
            history.append({
                "step": step + 1, "masked_before": len(masked), "revealed": len(revealed), "remasked": n_remasked,
                "min_conf": round(revealed[-1][0], 3), "mean_conf": round(sum(r[0] for r in revealed) / len(revealed), 3),
            })
            if snapshots is not None:
                snapshots.append(dict(letters))
        return FillResult(letters, confidence, history, snapshots)
