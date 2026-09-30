# Chinese Chess Jev

**Teaching a 0.6B "System-1" decision model to play Chinese chess (Xiangqi) — and watching, in detail, what it takes to make that actually work on a 4GB GPU.**

Chinese Chess Jev is a human-vs-AI Xiangqi app built on top of [NanoJev](https://github.com/TianyuCodings/NanoJev), an open-source replica of [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) — a "System-1" decision model that doesn't generate text at all. Give it a situation and a set of typed questions (yes/no, multiple choice, or a rating), and it returns a calibrated probability for every option in a single forward pass. This project asks a narrow question: **can that kind of model, fine-tuned on almost no data and no GPU budget to speak of, learn to make sane Xiangqi moves?**

The honest answer is "somewhat, and here's exactly what it took." This repo is both the game and the paper trail: real training logs, real GPU-memory war stories, and three separate rounds of "the model looks confused, here's the actual bug."

## Play it

```bash
pip install -r requirements.txt

# Terminal 1: load the model
python scripts/serve_decisions.py --checkpoint-dir <checkpoint-dir> --web-root web --port 8765

# Terminal 2: the game server
python scripts/serve_xiangqi.py --web-root web --port 8766 --nanojev-url http://127.0.0.1:8765
```

Open **http://127.0.0.1:8766/xiangqi.html**, pick a side, click a piece, click a destination. No checkpoint yet? See [Training your own checkpoint](docs/TRAINING.md) — or run with `--nanojev-url ""` to play against a uniformly-random legal-move bot while you get everything else wired up.

## What's actually going on here

Two things make this more than "chatbot plays chess":

1. **The model never generates a move.** A pure-Python rules engine ([`xiangqi_board.py`](scripts/xiangqi_board.py)) enumerates every legal move for the side to play. The model is asked exactly one `choice` question — *"pick the best move among these candidates"* — and answers with a probability for each one. No chain-of-thought, no illegal moves are even representable.
2. **The model has no lookahead, and code makes up for it.** A single forward pass can't see two moves ahead, so on its own it will happily shuffle a knight back and forth forever, or trade a Cannon for a Horse for nothing. `serve_xiangqi.py` adds three small, tested layers of code-side judgment before the model ever sees a candidate list:
   - **Repetition avoidance** — moves that would recreate a recent position are filtered out before the model gets to choose.
   - **Material-trade weighting** — a 1-ply "will this piece just get recaptured for less than it's worth" check discounts the model's own probabilities for losing trades.
   - **Hanging-piece defense** — before the model chooses, the server scans for any of its own pieces currently undefended and under attack, and guarantees at least one move that addresses the threat survives candidate pruning (not just gets a better score — actually gets *offered*, which turned out to be the real bug the first two times this was attempted).

This is the same split the original NanoJev maze/Snake demos use: **code plans and enforces constraints; the model only makes the narrow judgment calls.**

## Training pipeline

```
self-play random games  →  Pikafish (open-source engine) labels the best move
        ↓                              ↓
  xiangqi_board.py            "gold" move per position
        ↓                              ↓
              build_xiangqi_data.py → JSONL training data
                        ↓
     train_pipeline_decisions.py --freeze-backbone
                        ↓
         a ~200K-parameter decision head, fine-tuned
```

No human labeling, no paid API: [Pikafish](https://github.com/official-pikafish/Pikafish) (a free, GPL-licensed UCI Xiangqi engine) plays the role of the teacher. Full details, including why `--freeze-backbone` exists and three rounds of real training results, are in **[docs/TRAINING.md](docs/TRAINING.md)**.

### Headline results

| Run | Labeled positions | Training steps | Dev cross-entropy | Wall time (single 4GB GPU) |
|---|---:|---:|---:|---:|
| Untrained baseline | — | 0 | ~2.47–2.50 | — |
| v1 | 1,200 | 150 | 2.374 | ~2.0h |
| v2 | 1,200 | 430 | 2.355 | ~1.7h |
| v3 | 6,000 | 2,040 | **2.053** | ~7.6h |

(Lower is better — cross-entropy against Pikafish's own chosen move.) The clearest finding: **more labeled positions helped far more than more training steps on the same ones** — v3's 5x larger dataset produced the biggest single jump, while v1 and v2 show the classic overfitting signature of a small fixed pool (the best checkpoint always came from *before* the final step). Raw `summary.json` for all three runs is in [`results/`](results/).

## Tech stack

| Layer | What | Where |
|---|---|---|
| Model | Qwen3-0.6B backbone + a small decision head | fine-tuned via `train_pipeline_decisions.py` |
| Inference serving | PyTorch + Transformers + safetensors, CUDA required | [`scripts/serve_decisions.py`](scripts/serve_decisions.py) |
| Rules engine | Pure Python, stdlib only, fully unit-tested | [`scripts/xiangqi_board.py`](scripts/xiangqi_board.py) |
| Model bridge | Board/move ⇄ text for the model's API | [`scripts/xiangqi_notation.py`](scripts/xiangqi_notation.py) |
| Game server | `http.server`, no framework | [`scripts/serve_xiangqi.py`](scripts/serve_xiangqi.py) |
| Data labeling | UCI client talking to Pikafish | [`scripts/pikafish_engine.py`](scripts/pikafish_engine.py), [`scripts/build_xiangqi_data.py`](scripts/build_xiangqi_data.py) |
| Frontend | Vanilla HTML/CSS/JS, zero build step | [`web/xiangqi.html`](web/xiangqi.html) |

Run the test suite (no GPU needed — it's pure rules-engine and data-plumbing logic):
```bash
python -m unittest discover -s scripts -p 'test_*.py' -v
```

## A note on the Chinese text

The model's board state and move descriptions are rendered in Chinese (`state`, `criteria` in [`xiangqi_notation.py`](scripts/xiangqi_notation.py)) — on purpose. That's exactly the text the fine-tuned checkpoint was trained on; translating it would silently feed the model a different input distribution than the one it learned from, on a checkpoint that already has very little training data to spare. The web UI shows an English translation of every move (`description_en`) for readability, and the board itself uses the traditional Chinese piece characters, the same as most English-language Xiangqi resources.

## Acknowledgments

- Built on [NanoJev](https://github.com/TianyuCodings/NanoJev) (MIT), a replica of [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) from [typesafe.ai](https://typesafe.ai). The model architecture, inference server, and training pipeline in `scripts/serve_decisions.py`, `scripts/predict_toy_decisions.py`, `scripts/train_toy_decisions.py`, `scripts/calibrated_objectives.py`, and the base of `scripts/train_pipeline_decisions.py` (plus `--freeze-backbone`, added here) come from that project.
- Training data is labeled by [Pikafish](https://github.com/official-pikafish/Pikafish) (GPLv3), a free and strong open-source Xiangqi engine. Pikafish itself is not redistributed in this repository — download it separately (see [docs/TRAINING.md](docs/TRAINING.md)).
- Base model: [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B).

## License

MIT — see [LICENSE](LICENSE). This repository redistributes and modifies MIT-licensed code from NanoJev; the original notice is preserved.
