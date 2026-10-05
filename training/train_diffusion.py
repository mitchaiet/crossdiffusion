"""Fine-tune Gemma 3 as a masked-diffusion crossword filler (prefix-LM, LoRA).

    python -m training.train_diffusion --model google/gemma-3-4b-it \
        --train data/corpus/train.jsonl --val data/corpus/val.jsonl --out runs/gemma4b-r1

Objective (LLaDA-style): sample a mask ratio t per example, mask each white cell with
probability t, predict the letters at masked cells with bidirectional attention over the
canvas; loss = cross-entropy over the 26 letters at masked positions, weighted by 1/t.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from crossword_agent.grid import Puzzle
from training.serialize import CanvasVocab, build_example, prefix_lm_mask


def load_base_model(path: str, dtype, attn: str, device: str):
    """Return a text-only Gemma3ForCausalLM.

    Gemma 3 1B ships as a plain causal LM. The 4B/12B/27B checkpoints are multimodal wrappers
    (Gemma3ForConditionalGeneration) whose training-time mask builder demands token_type_ids and
    image bookkeeping we do not want. So for those we lift the language model out of the wrapper
    into a Gemma3ForCausalLM: same layers, same weights, no vision tower, and the plain text
    forward that accepts our 4D prefix-LM mask (the path the 1B run validated)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(path)
    if getattr(cfg, "model_type", "") == "gemma3_text" or not hasattr(cfg, "text_config"):
        return AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation=attn, device_map={"": device})

    import gc

    from transformers import Gemma3ForCausalLM, Gemma3ForConditionalGeneration

    print("multimodal Gemma 3 checkpoint: extracting the text model", flush=True)
    mm = Gemma3ForConditionalGeneration.from_pretrained(path, dtype=dtype, attn_implementation=attn, device_map={"": "cpu"})
    text_cfg = cfg.text_config
    text_cfg._attn_implementation = attn
    model = Gemma3ForCausalLM(text_cfg).to(dtype)
    lm = mm.model.language_model if hasattr(mm, "model") else mm.language_model
    missing, unexpected = model.model.load_state_dict(lm.state_dict(), strict=False)
    if missing or unexpected:
        print(f"  language model keys: missing={missing[:5]} unexpected={unexpected[:5]}", flush=True)
    model.lm_head.load_state_dict(mm.lm_head.state_dict())
    del mm, lm
    gc.collect()
    model = model.to(device)
    model.config._attn_implementation = attn
    return model


def load_puzzles(path: Path, limit: int | None = None) -> list[Puzzle]:
    out = []
    skipped = 0
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            p = Puzzle.from_rows(d["grid"], d.get("clues", {}), title=d.get("title", ""), source=d.get("source", ""))
            # the canvas has exactly 26 letter tokens: anything outside A-Z cannot be encoded
            if p.solution is None or any(not ("A" <= ch <= "Z") for ch in p.solution.values()):
                skipped += 1
                continue
            out.append(p)
            if limit and len(out) >= limit:
                break
    if skipped:
        print(f"skipped {skipped} puzzles with non A-Z letters in {path}", flush=True)
    return out


class PuzzleDataset(Dataset):
    """Fresh random masks every time an item is fetched (free augmentation)."""

    def __init__(self, puzzles: list[Puzzle], tok, vocab: CanvasVocab, seed: int = 0, fixed_t: float | None = None,
                 t1_frac: float = 0.15, noise: float = 0.0):
        self.puzzles, self.tok, self.vocab, self.seed, self.fixed_t = puzzles, tok, vocab, seed, fixed_t
        self.t1_frac, self.noise = t1_frac, noise
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.puzzles)

    def __getitem__(self, i: int):
        rng = random.Random(hash((self.seed, self.epoch, i)) & 0xFFFFFFFF)
        return build_example(self.puzzles[i], self.tok, self.vocab, rng, t=self.fixed_t, t1_frac=self.t1_frac, noise=self.noise)


class MixedBatchSampler(torch.utils.data.Sampler):
    """Batches of one kind at a time: full grids (small batches, long sequences) and one-row
    clue-pair puzzles (big batches, short sequences), interleaved in random order."""

    def __init__(self, puzzles: list[Puzzle], grid_batch: int, pair_batch: int, seed: int = 0):
        self.grid_idx = [i for i, p in enumerate(puzzles) if p.height > 1]
        self.pair_idx = [i for i, p in enumerate(puzzles) if p.height == 1]
        self.grid_batch, self.pair_batch, self.seed, self.epoch = grid_batch, pair_batch, seed, 0

    def _batches(self):
        rng = random.Random(self.seed + self.epoch)
        g, q = list(self.grid_idx), list(self.pair_idx)
        rng.shuffle(g)
        rng.shuffle(q)
        batches = [g[i : i + self.grid_batch] for i in range(0, len(g) - self.grid_batch + 1, self.grid_batch)]
        batches += [q[i : i + self.pair_batch] for i in range(0, len(q) - self.pair_batch + 1, self.pair_batch)]
        rng.shuffle(batches)
        return batches

    def __iter__(self):
        yield from self._batches()

    def __len__(self) -> int:
        return len(self.grid_idx) // self.grid_batch + len(self.pair_idx) // self.pair_batch


def collate(batch, pad_id: int):
    L = max(len(ex.input_ids) for ex in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), L), -100, dtype=torch.long)
    weights = torch.zeros(len(batch))
    lengths, prompt_lens = [], []
    for b, ex in enumerate(batch):
        n = len(ex.input_ids)
        ids[b, :n] = torch.tensor(ex.input_ids)
        labels[b, :n] = torch.tensor(ex.labels)
        weights[b] = 1.0 / max(ex.t, 0.05) / ex.n_white
        lengths.append(n)
        prompt_lens.append(ex.prompt_len)
    return ids, labels, weights, lengths, prompt_lens


def batch_attention_mask(lengths, prompt_lens, L, device, dtype):
    masks = []
    for n, pl in zip(lengths, prompt_lens, strict=True):
        m = prefix_lm_mask(L, pl, device=device, dtype=dtype)[0, 0]
        if n < L:  # padding: nobody attends to pads
            m[:, n:] = torch.finfo(dtype).min
        masks.append(m)
    return torch.stack(masks)[:, None]  # (B, 1, L, L)


def letter_loss(logits, labels, letter_ids, weights):
    """CE over the 26 letter logits at masked positions, per-example 1/t weighting."""
    sel = logits[..., letter_ids]  # (B, L, 26)
    mask = labels != -100
    if not mask.any():
        return logits.sum() * 0.0, 0
    ce = torch.nn.functional.cross_entropy(
        sel[mask].float(), labels[mask], reduction="none"
    )
    # map each masked token back to its batch row for weighting
    rows = mask.nonzero()[:, 0]
    loss = (ce * weights.to(ce.device)[rows]).sum() / logits.shape[0]
    acc = (sel[mask].argmax(-1) == labels[mask]).float().mean().item()
    return loss, acc


@torch.no_grad()
def evaluate(model, loader, letter_ids, device, max_batches: int = 50):
    model.eval()
    tot_loss, tot_acc, n = 0.0, 0.0, 0
    for i, (ids, labels, weights, lengths, prompt_lens) in enumerate(loader):
        if i >= max_batches:
            break
        ids, labels = ids.to(device), labels.to(device)
        am = batch_attention_mask(lengths, prompt_lens, ids.shape[1], device, torch.bfloat16)
        out = model(input_ids=ids, attention_mask=am)
        loss, acc = letter_loss(out.logits, labels, letter_ids, weights)
        tot_loss += loss.item()
        tot_acc += acc
        n += 1
    model.train()
    return tot_loss / max(n, 1), tot_acc / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--train", type=Path, nargs="+", required=True, help="one or more JSONL puzzle files")
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--full", action="store_true", help="full fine-tune instead of LoRA")
    ap.add_argument("--init-adapter", type=Path, default=None, help="merge this LoRA adapter into the base first, then train a fresh adapter on top")
    ap.add_argument("--resume-adapter", type=Path, default=None, help="keep training this LoRA adapter's own weights (gentler continuation)")
    ap.add_argument("--t1-frac", type=float, default=0.15, help="share of fully-masked training examples")
    ap.add_argument("--noise", type=float, default=0.0, help="fraction of visible letters corrupted during training")
    ap.add_argument("--pair-batch", type=int, default=64, help="batch size for one-row clue-pair puzzles")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N training puzzles")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attn", default="eager", help="attention implementation (eager|sdpa)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    vocab = CanvasVocab.from_tokenizer(tok)
    letter_ids = torch.tensor(vocab.letter_ids, device=device)

    model = load_base_model(args.model, torch.bfloat16, args.attn, device)
    model.config.use_cache = False
    if args.init_adapter:
        from peft import PeftModel

        print(f"merging previous adapter {args.init_adapter} into the base", flush=True)
        model = PeftModel.from_pretrained(model, str(args.init_adapter)).merge_and_unload()
    if args.resume_adapter:
        from peft import PeftModel

        print(f"resuming adapter {args.resume_adapter} (its weights stay trainable)", flush=True)
        model = PeftModel.from_pretrained(model, str(args.resume_adapter), is_trainable=True)
        model.print_trainable_parameters()
    elif not args.full:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                task_type="CAUSAL_LM",
            ),
        )
        model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    if not args.full:
        model.enable_input_require_grads()

    train_p = []
    for path in args.train:
        train_p += load_puzzles(path, args.limit)
    if args.limit:
        train_p = train_p[: args.limit]
    val_p = load_puzzles(args.val, 400)
    print(f"train puzzles {len(train_p)}  val puzzles {len(val_p)}")
    train_ds = PuzzleDataset(train_p, tok, vocab, seed=args.seed, t1_frac=args.t1_frac, noise=args.noise)
    val_ds = PuzzleDataset(val_p, tok, vocab, seed=1, fixed_t=None)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    coll = lambda b: collate(b, pad)  # noqa: E731
    sampler = MixedBatchSampler(train_p, args.batch, args.pair_batch, seed=args.seed)
    print(f"batches per epoch: {len(sampler)} ({len(sampler.grid_idx)} grids in batches of {args.batch}, "
          f"{len(sampler.pair_idx)} clue pairs in batches of {args.pair_batch})", flush=True)
    train_dl = DataLoader(train_ds, batch_sampler=sampler, collate_fn=coll, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=coll)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    steps_per_epoch = len(train_dl) // args.grad_accum
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    warmup = max(10, total_steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    log = (args.out / "log.jsonl").open("a")

    model.train()
    step, micro, t0 = 0, 0, time.time()
    run_loss, run_acc = 0.0, 0.0
    epoch = 0
    while step < total_steps:
        train_ds.epoch = epoch
        sampler.epoch = epoch
        for ids, labels, weights, lengths, prompt_lens in train_dl:
            ids, labels = ids.to(device), labels.to(device)
            am = batch_attention_mask(lengths, prompt_lens, ids.shape[1], device, torch.bfloat16)
            out = model(input_ids=ids, attention_mask=am)
            loss, acc = letter_loss(out.logits, labels, letter_ids, weights)
            (loss / args.grad_accum).backward()
            run_loss += loss.item() / args.grad_accum
            run_acc += acc / args.grad_accum
            micro += 1
            if micro % args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                rec = {"step": step, "loss": round(run_loss / 10, 4), "masked_acc": round(run_acc / 10, 4),
                       "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0, 1)}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                run_loss, run_acc = 0.0, 0.0
            if step % args.eval_every == 0 or step == total_steps:
                vl, va = evaluate(model, val_dl, letter_ids, device)
                rec = {"step": step, "val_loss": round(vl, 4), "val_masked_acc": round(va, 4)}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
            if step % args.save_every == 0 or step == total_steps:
                model.save_pretrained(args.out / "adapter")
                tok.save_pretrained(args.out / "adapter")
            if step >= total_steps:
                break
        epoch += 1
    print("done")


if __name__ == "__main__":
    main()
