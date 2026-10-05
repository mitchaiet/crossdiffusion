# CrossDiffusion model

CrossDiffusion predicts letters in a crossword grid from its clues and any known
letters. It starts with masked cells and fills the most confident predictions over
several passes. The grid gives each crossing a shared cell, so across and down
entries always use the same letter there. This does not guarantee correct words.

## Input and attention

`training/serialize.py` builds a clue prompt followed by a grid:

```text
<bos> Crossword 5x5.                    causal attention
      Title: ...
      Across: 1. Hatfield-McCoy conflict (4) ...
      Down: 1. Woodwind played sideways (5) ...
      Grid:
      # F E U D \n                     bidirectional attention within the grid;
      C L A R A \n                     full attention to the clue prompt
      ...
```

The grid uses one token per cell, in row order, plus a newline after each row.
The column of a grid token is `position mod (width + 1)`. The tokenizer must encode
`" A"` through `" Z"`, `" #"` (black), `" _"` (masked), and `"\n"` as single tokens;
`CanvasVocab` checks this when loaded. The loss and sampler consider only the 26
letter logits at white cells. The prompt cannot attend to the grid.

## Training

For each example, sample `t ~ U(0, 1)`, with a separate probability of choosing
`t = 1`, then mask each white cell with probability `t`. The default share of fully
masked examples is 15%. The implementation clamps `t` to at least 0.02 and ensures
at least one masked cell.

`training/train_diffusion.py` sums cross-entropy over masked cells, weights each
example by `1 / (max(t, 0.05) * number_of_white_cells)`, and averages over the batch.
Masks vary by example and epoch. The default trains a LoRA adapter with rank 64 on
the attention and MLP projections; `--full` trains all model weights.

The loader uses Gemma's text model. For multimodal checkpoints, it extracts the
language model and output head before training with the custom attention mask.

`training/build_dataset.py` splits published xd grids into training, validation,
and test files; defaults reserve 200 validation and 300 test puzzles. It can also
refill layouts from the training split and attach clues from the xd clue bank.
Synthetic examples are written separately to `train_synth.jsonl` and must be
included explicitly in `--train`.

The real grid split does not establish that clues or answers are absent from all
training sources. The synthetic clue bank uses the full supplied clue file. When
building one-row clue/answer examples with `training/build_clue_pairs.py`, use
`--exclude` to remove pairs from validation and test files. Training skips solutions
containing characters outside A–Z.

## Sampling and agent integration

`training/sample.py` starts with unknown white cells masked and keeps supplied
letters fixed. At each step it predicts masked cells and reveals the most confident
`ceil(remaining / steps_left)`. Cell confidence is the predicted probability when
the letter was revealed. Entry confidence is the geometric mean of its cells.
Optional settings remask weak predictions or vote across multiple sampled fills.

`training/serve.py` exposes the filler over HTTP. `crossword_agent/diffusion.py`
provides three integrations:

- `DiffusionSolver` fills the grid in one denoising run.
- `DiffusionOracle` proposes answers inside the agent loop, using committed letters
  as fixed input.
- `CompositeOracle` combines sources. In `inject` mode their candidates compete;
  in `fallback` mode the secondary source supplies only entries left unanswered
  by the primary source.

See [the training runbook](../training/README.md) for commands and
[the evaluation notes](EVALUATION.md) for historical comparisons.

## Recorded experiments

The September 2026 notes describe a 4B, rank-64 run on about 74,000 grids. They
report 63.5% validation accuracy on masked letters and 56.3% letter accuracy when
filling blank mixed-publisher 15×15 grids. These are different tasks and metrics.
The notes also report 57.5% on 100 random test grids and 35.1% on 11 NYT Saturdays.
Raw result files are not included in this checkout; these figures have not been
recomputed for the current deployment.

On the recorded 20-puzzle mixed set, the LLM agent scored 91.8% of letters,
competing diffusion candidates scored 85.0%, and fallback candidates scored 93.1%.
These runs suggest that how candidates are combined matters. They do not establish
an improvement across other puzzle sets or model versions.

A September 20 continuation with a fresh rank-128 adapter on merged v1 weights
fell from 63.0% to 61.8% masked-letter validation accuracy over 1,000 steps, then
failed on a non-ASCII clue answer. Its reported blank-grid accuracy was 54.4%,
compared with 56.3% for v1. A later plan resumed v1's adapter at learning rate
`6e-5`, with 120,000 clue pairs, 30% fully masked examples, and 3% corrupted visible
letters. No completion result for that plan is recorded here.

When training with `--init-adapter`, serve with the same `--init-adapter`: the new
adapter depends on those merged weights. `--resume-adapter` instead continues
training the existing adapter. The separate 12B correction adapter is described
in [the archive notes](RELEASE_ARTIFACTS.md).
