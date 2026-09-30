#!/usr/bin/env python3
"""Human-vs-NanoJev Xiangqi server: rules engine here, AI moves via a running
serve_decisions.py instance's /api/evaluate. No model weights are loaded in
this process, so it can share the GPU-resident model instead of doubling it.
"""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import mimetypes
from pathlib import Path
import random
import urllib.error
import urllib.request

import xiangqi_board as xq
import xiangqi_notation as notation

GAME = {}
# Every candidate leaf re-encodes the full state text (no shared-prefix batching in this
# reference implementation), so a full move list (~40+ in the opening) makes one forward
# pass take minutes on a 4GB GPU. Code narrows the choice before asking the model, the same
# "code filters, model judges" split this repo's maze/snake demos use.
MAX_CANDIDATES = 12


def prune_moves(board, moves, cap=MAX_CANDIDATES, side=None, hanging_before=None):
    if len(moves) <= cap:
        return moves
    # Captures always survive pruning; so must any move that answers a piece currently
    # hanging (side/hanging_before are only passed once that's known — see play_ai_move).
    # Otherwise a random quiet-move sample could drop the one move that saves it, and no
    # amount of downstream reweighting can recover a candidate that was never offered.
    priority = [m for m in moves if m[1] in board]
    if hanging_before:
        priority += [m for m in moves if m not in priority
                    and resolves_threats(board, m, side, hanging_before)]
    if len(priority) >= cap:
        return random.sample(priority, cap)
    quiet = [m for m in moves if m not in priority]
    return priority + random.sample(quiet, cap - len(priority))


# The model has no lookahead: without this, it happily shuffles the same piece back and
# forth forever (observed in practice), since undoing its own last move can look locally
# fine to a single-step judgment. Code excludes moves that recreate a recent position,
# the same "code plans, model judges narrowly" split as prune_moves above.
REPETITION_WINDOW = 8


def board_key(board):
    return frozenset(board.items())


# The model also has no explicit notion of piece values (observed trading a Cannon for a
# Horse for nothing in return), because a single forward pass can't see the opponent's
# reply. Code does one ply of lookahead purely for material: if a capture leaves the moved
# piece immediately recapturable for less than it took, that candidate's probability is
# crushed near-zero before picking the argmax — any losing trade, mild or severe, is
# avoided whenever a better-adjusted candidate exists. It is never driven to exactly zero,
# so a losing trade can still be chosen when literally nothing else is on offer.
PIECE_VALUES = {"G": 100, "R": 9, "C": 4.5, "H": 4, "A": 2, "E": 2, "S": 1}
LOSING_TRADE_PENALTY = 0.02


def material_adjustment(board, move, side):
    (fr, fc), (tr, tc) = move
    target = board.get((tr, tc))
    if target is None:
        return 1.0  # quiet move: no capture to evaluate
    mover_value = PIECE_VALUES[board[(fr, fc)][1]]
    captured_value = PIECE_VALUES[target[1]]
    after = xq.apply_move(board, move)
    if not xq.is_attacked(after, tr, tc, xq.opposite(side)):
        return 1.0  # a capture the opponent can't immediately answer: free material, no penalty
    net = captured_value - mover_value
    if net >= 0:
        return 1.0  # even or favorable trade
    return LOSING_TRADE_PENALTY  # any losing trade, however small, is near-vetoed


# material_adjustment only judges the move being considered; it has no opinion on a piece
# that is hanging *before* the AI even moves (observed: a Cannon sat undefended for two
# full turns before Red's Rook took it for free). This scans for pieces the opponent could
# win outright or favorably right now, and discourages any candidate that leaves all of
# them exactly as exposed as they were.
THREAT_IGNORED_PENALTY = 0.15


def hanging_pieces(board, side):
    opponent = xq.opposite(side)
    hanging = []
    for (r, c), (s, piece) in board.items():
        if s != side or piece == "G":
            continue
        attackers = [sq for sq, occ in board.items()
                    if occ[0] == opponent and (r, c) in xq._pseudo_moves(board, *sq)]
        if not attackers:
            continue
        cheapest_attacker = min(PIECE_VALUES[board[sq][1]] for sq in attackers)
        # Check for defenders on a copy with this square vacated: pseudo-move generation
        # never lets a piece "capture" its own side, so checking the board as-is would
        # always find zero defenders for the square this very piece occupies.
        vacated = dict(board)
        del vacated[(r, c)]
        defended = xq.is_attacked(vacated, r, c, side)
        if not defended or cheapest_attacker <= PIECE_VALUES[piece]:
            hanging.append((r, c))
    return hanging


def resolves_threats(board, move, side, hanging_before):
    if not hanging_before:
        return True
    after = xq.apply_move(board, move)
    return not (set(hanging_before) & set(hanging_pieces(after, side)))


def threat_adjustment(board, move, side, hanging_before):
    return 1.0 if resolves_threats(board, move, side, hanging_before) else THREAT_IGNORED_PENALTY


def filter_repetitive(board, moves, history, window=REPETITION_WINDOW):
    recent = set(history[-window:])
    if not recent:
        return moves
    kept = [m for m in moves if board_key(xq.apply_move(board, m)) not in recent]
    return kept or moves  # never leave zero candidates just to dodge repetition


def new_game(human_side):
    GAME.clear()
    board = xq.initial_board()
    GAME.update(board=board, side_to_move=xq.RED, human_side=human_side,
                move_log=[], status="ongoing", winner=None, position_history=[board_key(board)])


def state_payload():
    status = GAME["status"]
    return {
        "board": notation.board_to_json(GAME["board"]),
        "side_to_move": GAME["side_to_move"],
        "human_side": GAME["human_side"],
        "status": status,
        "winner": GAME["winner"],
        "move_log": GAME["move_log"],
    }


def ask_nanojev(nanojev_url, board, side, moves, hanging_before):
    by_id, criteria = notation.moves_to_choice_criteria(board, moves)
    request = {"states": [{"id": "xiangqi_move", "state": notation.render_board_text(board, side),
                           "questions": {"best_move": {"type": "choice",
                                                       "instructions": "选择当前一方在中国象棋规则下的最佳走法。",
                                                       "criteria": criteria}}}]}
    body = json.dumps(request).encode("utf-8")
    req = urllib.request.Request(f"{nanojev_url}/api/evaluate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        result = json.load(resp)
    answer = result["states"][0]["answers"]["best_move"]
    probabilities = answer["probabilities"]
    adjusted = {mid: probabilities[mid] * material_adjustment(board, by_id[mid], side)
                     * threat_adjustment(board, by_id[mid], side, hanging_before)
                for mid in probabilities}
    chosen_id = max(adjusted, key=adjusted.get)
    overridden = chosen_id != answer["choice"]
    return by_id[chosen_id], probabilities[chosen_id], criteria[chosen_id], overridden


def play_ai_move(nanojev_url):
    board, side = GAME["board"], GAME["side_to_move"]
    moves = xq.legal_moves(board, side)
    if not moves:
        return None
    candidates = filter_repetitive(board, moves, GAME["position_history"])
    hanging_before = hanging_pieces(board, side)
    source, probability, description = "random_fallback", 1.0 / len(candidates), None
    move = random.choice(candidates)
    if nanojev_url:
        try:
            pruned = prune_moves(board, candidates, side=side, hanging_before=hanging_before)
            move, probability, description, overridden = ask_nanojev(nanojev_url, board, side,
                                                                      pruned, hanging_before)
            source = "nanojev_material_adjusted" if overridden else "nanojev"
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
            description = f"NanoJev service unavailable ({exc}); fell back to a random legal move"
    if description is None:
        description = notation.describe_move(board, move)
    description_en = notation.describe_move_en(board, move)
    GAME["board"] = xq.apply_move(board, move)
    GAME["side_to_move"] = xq.opposite(side)
    GAME["position_history"].append(board_key(GAME["board"]))
    GAME["move_log"].append({"side": side, "move": notation.move_to_square_pairs(move),
                             "description": description, "description_en": description_en,
                             "source": source, "probability": probability})
    update_status()
    return {"move": notation.move_to_square_pairs(move), "description": description,
            "description_en": description_en, "source": source, "probability": probability}


def update_status():
    status = xq.game_status(GAME["board"], GAME["side_to_move"])
    GAME["status"] = status
    GAME["winner"] = xq.opposite(GAME["side_to_move"]) if status in ("checkmate", "stalemate") else None


def server_class(web_root, nanojev_url):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, code, data):
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def read_json(self):
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1_000_000:
                raise ValueError("Request must contain 1..1000000 bytes")
            return json.loads(self.rfile.read(length))

        def do_GET(self):
            if self.path == "/api/state":
                if not GAME:
                    new_game(xq.RED)
                self.send_json(200, state_payload())
                return
            relative = self.path.lstrip("/") or "xiangqi.html"
            target = (web_root / relative).resolve()
            if not target.is_relative_to(web_root) or not target.is_file():
                self.send_json(404, {"error": "File not found"})
                return
            data = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(str(target))[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            try:
                if self.path == "/api/new":
                    payload = self.read_json() if int(self.headers.get("Content-Length", "0")) else {}
                    human_side = payload.get("human_side", xq.RED)
                    if human_side not in (xq.RED, xq.BLACK):
                        raise ValueError("human_side must be 'red' or 'black'")
                    new_game(human_side)
                    self.send_json(200, {**state_payload(), "ai_move": None})
                    return
                if self.path == "/api/legal-destinations":
                    payload = self.read_json()
                    r, c = int(payload["row"]), int(payload["col"])
                    dests = xq.legal_destinations(GAME["board"], GAME["side_to_move"], r, c)
                    self.send_json(200, {"destinations": [{"row": tr, "col": tc} for tr, tc in dests]})
                    return
                if self.path == "/api/move":
                    if GAME["status"] != "ongoing":
                        raise ValueError("Game is over; start a new game")
                    if GAME["side_to_move"] != GAME["human_side"]:
                        raise ValueError("It is not the human side's turn")
                    payload = self.read_json()
                    move = ((int(payload["from"]["row"]), int(payload["from"]["col"])),
                           (int(payload["to"]["row"]), int(payload["to"]["col"])))
                    if move not in xq.legal_moves(GAME["board"], GAME["side_to_move"]):
                        raise ValueError("Illegal move")
                    description = notation.describe_move(GAME["board"], move)
                    description_en = notation.describe_move_en(GAME["board"], move)
                    GAME["board"] = xq.apply_move(GAME["board"], move)
                    GAME["position_history"].append(board_key(GAME["board"]))
                    GAME["move_log"].append({"side": GAME["side_to_move"],
                                             "move": notation.move_to_square_pairs(move),
                                             "description": description, "description_en": description_en,
                                             "source": "human", "probability": None})
                    GAME["side_to_move"] = xq.opposite(GAME["side_to_move"])
                    update_status()
                    self.send_json(200, {**state_payload(), "ai_move": None})
                    return
                if self.path == "/api/ai-move":
                    length = int(self.headers.get("Content-Length", "0"))
                    if length:
                        self.rfile.read(length)  # drain any body; this endpoint takes no input
                    if GAME["status"] != "ongoing":
                        raise ValueError("Game is over; start a new game")
                    if GAME["side_to_move"] == GAME["human_side"]:
                        raise ValueError("It is the human side's turn")
                    ai_move = play_ai_move(nanojev_url)
                    self.send_json(200, {**state_payload(), "ai_move": ai_move})
                    return
                self.send_json(404, {"error": "Unknown endpoint"})
            except (ValueError, TypeError, KeyError) as exc:
                self.send_json(400, {"error": str(exc)})

        def log_message(self, fmt, *args):
            print(fmt % args, flush=True)
    return Handler


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--web-root", default="web")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--nanojev-url", default="http://127.0.0.1:8765",
                   help="Base URL of a running serve_decisions.py; empty string disables it (random fallback only)")
    args = p.parse_args()
    root = Path(args.web_root).resolve()
    new_game(xq.RED)
    server = HTTPServer((args.host, args.port), server_class(root, args.nanojev_url or None))
    print(json.dumps({"url": f"http://{args.host}:{args.port}/xiangqi.html", "nanojev_url": args.nanojev_url or None}),
          flush=True)
    server.serve_forever()
