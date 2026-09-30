# Training Chinese Chess Jev: the full story

This is the deep dive: how the AI side of the game actually decides on a move, how the
fine-tuned checkpoints were produced, what broke along the way, and what the real
bottleneck turned out to be. Everything here is backed by files in this repo — the code
in `scripts/`, and the raw training logs in `results/`.

## Part 1 — How one AI turn actually works

1. The human clicks a destination → the browser `POST`s `{from, to}` to `serve_xiangqi.py`.
2. The server validates the move against `xiangqi_board.legal_moves(...)`, applies it, and
   flips whose turn it is.
3. It's the AI's turn. The server generates every legal move for that side and runs it
   through three code-side filters (Part 2) before ever asking the model.
4. The server builds one NanoJev request: `state` is the whole board rendered as text,
   and one `choice` question whose `criteria` are the surviving candidate moves.
5. It `POST`s that to `serve_decisions.py`'s `/api/evaluate` — the model, already loaded
   in a separate process — and gets back a probability for every candidate.
6. The server re-weights those probabilities (Part 2), applies the winner, appends it to
   the move log, and checks for checkmate/stalemate.
7. The new state (plus what the AI played and how confident it was) goes back to the
   browser in the same response that started at step 1.

The model is never asked to produce a move — only to score a list the rules engine
already knows is 100% legal.

## Part 2 — Why the model needs help, and what that help looks like

A `choice` question is answered in a single forward pass. The model cannot see its own
reply two moves later. Three failure modes showed up in real games, in this order, and
each got a small, unit-tested piece of code rather than more training data (the first
two are genuinely fixable with better training; the third turned out to be a bug in how
candidates were being offered, not a modeling problem at all):

### 1. Infinite piece-shuffling

Observed directly: a Knight moved back and forth between the same two squares for
several turns. Undoing your own last move can look perfectly reasonable to a model with
no memory of what it just did. Fix: `serve_xiangqi.py` tracks the last 8 board positions
(`board_key`, a hashable snapshot) and excludes any candidate that would recreate one of
them — unless that would leave zero candidates, in which case the filter backs off
entirely rather than stalemate the AI by accident.

### 2. Giving away material in "trades" the model doesn't evaluate

Observed: after a standard central-cannon opening, the AI traded its own Cannon for the
human's Horse — a mildly unfavorable trade in real Xiangqi theory (a Cannon is generally
worth a little more than a Horse early in the game, when there are plenty of screening
pieces on the board). The model has no explicit notion of piece value; it just picked the
capture that looked appealing in isolation.

Fix: `material_adjustment()` does one ply of lookahead purely for material. For every
candidate that captures something, it checks whether the destination square would be
immediately recapturable, and by what. If the trade nets a loss — *any* loss, mild or
severe — that candidate's model-given probability gets multiplied by `0.02` before the
argmax. It's never driven to exactly zero, so a bad trade can still be played if it's
genuinely the only option, but in practice a better alternative almost always exists and
wins instead.

### 3. Leaving a piece hanging for multiple turns — the real bug

Observed: a Cannon sat on an undefended square for two full turns, ignored, before the
opponent's Rook simply took it for free. The first fix attempt added a `hanging_pieces()`
scan (does the opponent threaten one of my own pieces right now, for a profit?) and a
`threat_adjustment()` penalty mirroring the material-trade logic above — any candidate
that leaves an already-hanging piece exactly as exposed gets the same harsh discount.

That fix didn't work the first time, and the reason was more interesting than the bug it
was chasing: **the defending move was never in the model's candidate list to begin with.**
The opening position has 40+ legal moves; `prune_moves()` keeps every capture and fills
the rest of a 12-move cap with a random sample of quiet moves, for GPU-memory reasons (see
Part 3). A reweighting scheme can only rescue a candidate that's actually offered — if the
one move that saves a piece loses the coin flip during random sampling, no amount of
downstream probability adjustment can bring it back. The real fix had to go one layer
earlier: `prune_moves()` now also guarantees inclusion of any move that resolves a
currently-hanging piece, with the same priority as a capture. Verified by replaying the
exact opening position that exposed the bug: before the fix, the central pawn was
provably hanging and nothing addressed it; after, the model played 马8进7 — the textbook
response — and the pawn was provably safe (see `test_serve_xiangqi.py` for the
regression tests, including one that runs the "does the fix survive random sampling"
check 20 times in a row, since the original bug was a matter of odds, not a hard failure).

## Part 3 — Training a checkpoint on almost nothing

The base Qwen3-0.6B-backed checkpoint this project started from had never seen a single
Xiangqi position. It answers a `choice` question fluently (valid probabilities, valid
format) with zero actual chess judgment. Making it better meant generating labeled data
and fine-tuning — on a single 4GB consumer GPU.

### Data generation — Pikafish as the teacher, no human labeling

[Pikafish](https://github.com/official-pikafish/Pikafish) is a free, GPLv3, UCI-protocol
Xiangqi engine descended from Stockfish's codebase. It is **not** included in this repo —
download and extract a release yourself, then point `pikafish_engine.py` at the binary.
Conveniently, UCI Xiangqi squares (`<file a-i><rank 0-9>`) map directly onto this repo's
`(row, col)` coordinates, so the only translation needed is a letter-to-index lookup.

`build_xiangqi_data.py` self-plays games that pick a uniformly random legal move each ply
(seeded, for reproducibility), samples positions along the way, and asks Pikafish for its
best move at each one. That move becomes the `gold` label; the (pruned) legal move list at
that position becomes the `choice` criteria. Positions with only one legal move are
skipped — a `choice` question needs at least two options, and a forced move teaches
nothing anyway.

```bash
python scripts/build_xiangqi_data.py --output-dir data/xiangqi_v3 \
  --train-count 6000 --dev-count 15 --calibration-count 15 --test-count 20 --ood-count 20
```

Always validate before spending GPU time:
```bash
python scripts/train_pipeline_decisions.py --input data/xiangqi_v3 --validate-only
```

### `--freeze-backbone`: fitting fine-tuning into 4GB

Full-parameter fine-tuning of a 0.6B-parameter backbone needs roughly 9.6GB of AdamW
state (parameters + gradients + two optimizer moments, all in fp32) — confirmed
experimentally by watching it OOM on a 4GB card. `--freeze-backbone` (added to
`train_pipeline_decisions.py` for this project) keeps the ~600M backbone parameters out
of the optimizer entirely and never unfreezes them after head warmup, so only the
~200K-parameter decision head is actually trained. Measured peak usage: ~2.8GB, stable
across every run regardless of dataset size.

```bash
python scripts/train_pipeline_decisions.py --input data/xiangqi_v3 --output-dir runs/xiangqi_v3 \
  --model Qwen/Qwen3-0.6B --objective gold_distribution --loss ce --set-head attention \
  --head-steps 60 --steps 1980 --batch-questions 4 --microbatch-questions 1 \
  --max-microbatch-tokens 4000 --eval-every 200 --precision bf16 --freeze-backbone
```

The obvious tradeoff: the backbone's understanding of text is never adapted to what
Xiangqi board descriptions actually look like. Only the small head learns anything, which
caps how much it can learn from any amount of data — but it's the only option that fits
this hardware.

### Three rounds of training, and what they show

| Run | Labeled positions | Steps | Best step | Dev CE | Wall time | Peak GPU |
|---|---:|---:|---:|---:|---:|---:|
| v1 | 1,200 | 150 | 150/150 | 2.374 | ~2.0h | 2.77GB |
| v2 | 1,200 | 430 | 370/430 | 2.355 | ~1.7h | 2.77GB |
| v3 | 6,000 | 2,040 | 1,460/2,040 | **2.053** | ~7.6h | 2.77GB |

Full `summary.json` for each run is in [`results/`](../results/). Cross-entropy is
against Pikafish's chosen move; lower is better. Two things stand out:

- **The best checkpoint was never the last step**, in every single run. Dev
  cross-entropy bottoms out and then drifts back up — the standard overfitting signature
  of a fixed-size pool being cycled repeatedly. `train_pipeline_decisions.py` already
  selects on dev loss automatically, so this isn't a failure, just a visible ceiling.
- **v3's 5x larger dataset produced by far the largest improvement**, despite using
  proportionally *fewer* passes over its data than v2 used over its smaller pool. More
  steps on the same 1,200 examples (v1 → v2) helped a little; more *unique* examples
  helped much more. The highest-leverage next step for this project is generating a
  substantially larger dataset, not training longer on what's already here.

### Two operational lessons from running this on a small Windows GPU

- **Check `nvidia-smi` for competing processes before trusting any timing estimate.**
  Per-question training cost swung between ~10.5s and ~3.5s across otherwise-identical
  runs, purely depending on whether an idle inference server was left resident on the
  same 4GB card. Always look at wall-clock evidence, not assumptions.
- **`nvidia-smi`'s reported memory isn't the same as real risk.** Reserved memory crept
  toward ~3.8GB mid-run at one point and looked alarming, but
  `torch.cuda.max_memory_allocated()` — the actual peak tensor allocation — stayed at
  2.77GB throughout every run. The gap is the CUDA caching allocator holding
  reserved-but-unused blocks, which fluctuates with whatever candidate-count/text-length
  shapes a given random batch happens to sample. `expandable_segments:True` (PyTorch's
  usual fix for this kind of fragmentation) is **not supported on Windows**; a periodic
  `torch.cuda.empty_cache()` every 20 steps was added to `train_pipeline_decisions.py`
  as a cross-platform mitigation instead.

Keep evaluation splits small (15-30 examples was enough here): the full-split evaluation
that runs once after training scales directly with their combined size, and at this
GPU's per-question cost, an oversized eval split alone can eat hours of wall time before
a single real training step happens.

## Reproducing this

```bash
# 1. Download and extract Pikafish (github.com/official-pikafish/Pikafish) to engines/
# 2. Generate labeled data
python scripts/build_xiangqi_data.py --output-dir data/xiangqi_v3 \
  --engine-path engines/<pikafish-binary> \
  --train-count 6000 --dev-count 15 --calibration-count 15 --test-count 20 --ood-count 20

# 3. Validate, then train
python scripts/train_pipeline_decisions.py --input data/xiangqi_v3 --validate-only
python scripts/train_pipeline_decisions.py --input data/xiangqi_v3 --output-dir runs/xiangqi_v3 \
  --model Qwen/Qwen3-0.6B --objective gold_distribution --loss ce --set-head attention \
  --head-steps 60 --steps 1980 --batch-questions 4 --microbatch-questions 1 \
  --max-microbatch-tokens 4000 --eval-every 200 --precision bf16 --freeze-backbone

# 4. Serve the result
python scripts/serve_decisions.py --checkpoint-dir runs/xiangqi_v3 --web-root web --port 8765
python scripts/serve_xiangqi.py --web-root web --port 8766 --nanojev-url http://127.0.0.1:8765
```

Trained checkpoints aren't included in this repo (the weights file alone is ~2.3GB); the
JSON summaries in `results/` are the real, unedited output of the three runs described
above.
