#!/usr/bin/env python3
"""Label xiangqi positions with Pikafish's best move, in NanoJev's JSONL training
schema (same shape as build_toy_decisions.py / build_game_decisions.py): one
programmatic oracle (Pikafish) supplies `gold`, no human labeling, no paid API.

Positions come from self-play games that pick a uniformly random legal move at
each ply (seeded), which is enough to reach a diverse mix of openings/middlegames
for a first pilot dataset; it does not imitate real human play.
"""
import argparse
import json
from pathlib import Path
import random

import pikafish_engine as pk
import xiangqi_board as xq
import xiangqi_notation as notation
from serve_xiangqi import prune_moves

SPLITS = ("train", "dev", "calibration", "test", "ood")


def play_random_game(rng, max_plies):
    board = xq.initial_board()
    side = xq.RED
    positions = []  # (board_before_move, side, move_history_ucis)
    ucis = []
    for _ply in range(max_plies):
        legal = xq.legal_moves(board, side)
        if not legal:
            break
        positions.append((dict(board), side, list(ucis)))
        move = rng.choice(legal)
        board = xq.apply_move(board, move)
        ucis.append(pk.move_to_uci(move))
        side = xq.opposite(side)
    return positions


def label_position(engine, board, side, history_ucis, movetime_ms, rng):
    legal = xq.legal_moves(board, side)
    if len(legal) < 2:
        return None  # a `choice` question needs >=2 options; a forced move teaches nothing anyway
    candidates = prune_moves(board, legal)
    best_uci = engine.best_move_for_ucis(history_ucis, movetime_ms=movetime_ms)
    if best_uci is None:
        return None
    best_move = pk.uci_to_move(best_uci)
    if best_move not in legal:
        return None  # defensive: never trust an engine move our own rules engine disagrees with
    if best_move not in candidates:
        # prune_moves only drops moves when len(legal) > cap; candidates is already at the cap here.
        candidates = list(candidates)
        candidates[-1] = best_move
    rng.shuffle(candidates)
    by_id, criteria = notation.moves_to_choice_criteria(board, candidates)
    gold_id = next(mid for mid, mv in by_id.items() if mv == best_move)
    return criteria, gold_id


def build(output_dir, seed, games, max_plies, positions_per_game, movetime_ms, engine_path, counts):
    rng = random.Random(seed)
    engine = pk.PikafishEngine(engine_path)
    records_by_split = {s: [] for s in SPLITS}
    split_order = [s for s in SPLITS for _ in range(counts[s])]
    rng.shuffle(split_order)
    cursor = 0
    try:
        for game_idx in range(games):
            positions = play_random_game(rng, max_plies)
            if not positions:
                continue
            sample = rng.sample(positions, min(positions_per_game, len(positions)))
            for pos_idx, (board, side, history) in enumerate(sample):
                if cursor >= len(split_order):
                    break
                labeled = label_position(engine, board, side, history, movetime_ms, rng)
                if labeled is None:
                    continue
                criteria, gold_id = labeled
                state_id = f"xiangqi:g{game_idx}:p{pos_idx}"
                record = {
                    "id": state_id, "state_id": state_id, "family_id": "xiangqi_pikafish_v1",
                    "split": split_order[cursor],
                    "state": notation.render_board_text(board, side),
                    "questions": {"best_move": {"type": "choice",
                                                "instructions": "选择当前一方在中国象棋规则下的最佳走法。",
                                                "criteria": criteria}},
                    "gold": {"best_move": gold_id},
                    "metadata": {"source": "pikafish_self_play", "license": "engine-labeled",
                                "source_group_id": state_id, "history_length": len(history)},
                }
                records_by_split[record["split"]].append(record)
                cursor += 1
            if cursor >= len(split_order):
                break
    finally:
        engine.close()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in SPLITS:
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records_by_split[split])
        (out / f"{split}.jsonl").write_text(text, encoding="utf-8")
        summary[split] = len(records_by_split[split])
    print(json.dumps({"output_dir": str(out), "counts": summary, "requested": counts}, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="data/xiangqi_pilot")
    p.add_argument("--engine-path", default="engines/Pikafish-Windows-x86-64-universal.exe")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--max-plies", type=int, default=40)
    p.add_argument("--positions-per-game", type=int, default=6)
    p.add_argument("--movetime-ms", type=int, default=300)
    for split, default in zip(SPLITS, (160, 40, 40, 40, 40)):
        p.add_argument(f"--{split}-count", type=int, default=default)
    args = p.parse_args()
    counts = {s: getattr(args, f"{s}_count") for s in SPLITS}
    build(args.output_dir, args.seed, args.games, args.max_plies, args.positions_per_game,
          args.movetime_ms, args.engine_path, counts)
