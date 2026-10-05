# Evaluation methodology

The public release contains no puzzle examples or answer keys. To evaluate it, assemble a separate licensed test set with known solutions and keep those solutions out of the solver input.

## Measures

- **Letter accuracy:** correct white-cell letters divided by all white cells; blanks count as wrong.
- **Word accuracy:** completely correct Across and Down entries divided by all entries.
- **Full-puzzle solve rate:** puzzles with every white cell correct divided by all puzzles.
- **Fill rate and precision:** filled cells divided by white cells, and correct letters divided by filled cells.
- **Confidence calibration:** ten-bin expected calibration error for entry confidence.
- **Resource use:** elapsed time, rounds, model calls, tokens, and estimated API cost.
- **Reading quality:** grid dimensions, black-cell accuracy, clue-number matching, and exact clue transcription before solving.

The scorer in `crossword_agent/evaluate.py` reports per-puzzle scores and aggregates. Run it on your own licensed JSON, JSONL, or unlocked `.puz` inputs with answer keys. Keep the answer keys available only to the scorer.

## Proposed held-out protocol

1. Freeze the source revision, model and adapter revision, prompts, settings, test-set IDs and hashes, and cache policy. Keep tuning puzzles separate from the final set.
2. Select at least 100 licensed puzzles across sizes, publishers, and difficulty levels. Check for duplicate grids and clue-answer pairs. Holding out a puzzle from local training cannot rule out foundation-model exposure.
3. Run the same puzzles with a one-shot LLM, a per-clue top-1 baseline, CrossDiffusion alone, and the complete LLM agent. Compare single-round and no-repair ablations to measure the contribution of iteration.
4. Repeat each live configuration at least three times. Report paired differences, 95% bootstrap intervals resampled by puzzle, failure counts, latency, and cost. Independent runs should disable response caches.
5. Test image and PDF reading separately from solving using verified transcriptions. Then measure the complete upload-to-solution path. Report human corrections separately.

The current public release does not include a held-out benchmark report for its Gemma 3 12B adapter. Hash checks establish file integrity, not solving accuracy.
