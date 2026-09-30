#!/usr/bin/env python3
"""Xiangqi (Chinese chess) rules engine: board state, legal move generation, check/mate detection.

Board coordinates: row 0-9, column 0-8. Row 0 is Red's back rank, row 9 is
Black's back rank. The river lies between row 4 and row 5. Red's palace is
columns 3-5, rows 0-2; Black's palace is columns 3-5, rows 7-9.

No model code here; this module is the single source of truth for legality,
shared by the HTTP game server and by anything that later generates training
data from a stronger engine.
"""
import copy

RED, BLACK = "red", "black"
PIECES = "GAEHRCS"  # General, Advisor, Elephant, Horse, Chariot (Rook), Cannon, Soldier

PALACE_COLS = (3, 4, 5)
RED_PALACE_ROWS = (0, 1, 2)
BLACK_PALACE_ROWS = (7, 8, 9)


def opposite(side):
    return BLACK if side == RED else RED


def on_board(r, c):
    return 0 <= r <= 9 and 0 <= c <= 8


def in_palace(side, r, c):
    rows = RED_PALACE_ROWS if side == RED else BLACK_PALACE_ROWS
    return c in PALACE_COLS and r in rows


def in_own_half(side, r):
    return r <= 4 if side == RED else r >= 5


def initial_board():
    board = {}
    back = "RHEAGAEHR"
    for c, piece in enumerate(back):
        board[(0, c)] = (RED, piece)
        board[(9, c)] = (BLACK, piece)
    for c in (1, 7):
        board[(2, c)] = (RED, "C")
        board[(7, c)] = (BLACK, "C")
    for c in (0, 2, 4, 6, 8):
        board[(3, c)] = (RED, "S")
        board[(6, c)] = (BLACK, "S")
    return board


def find_general(board, side):
    for (r, c), (s, p) in board.items():
        if s == side and p == "G":
            return (r, c)
    return None


def _clear_path(board, r1, c1, r2, c2):
    """Squares strictly between two points on a shared row/column, exclusive."""
    if r1 == r2:
        step = 1 if c2 > c1 else -1
        return [(r1, c) for c in range(c1 + step, c2, step)]
    step = 1 if r2 > r1 else -1
    return [(r, c1) for r in range(r1 + step, r2, step)]


def _pseudo_moves(board, r, c):
    """Raw movement pattern for the piece at (r, c), ignoring self-check."""
    side, piece = board[(r, c)]
    targets = []

    def add(nr, nc):
        if on_board(nr, nc):
            occ = board.get((nr, nc))
            if occ is None or occ[0] != side:
                targets.append((nr, nc))

    if piece == "G":
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if in_palace(side, nr, nc):
                add(nr, nc)
    elif piece == "A":
        for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            nr, nc = r + dr, c + dc
            if in_palace(side, nr, nc):
                add(nr, nc)
    elif piece == "E":
        for dr, dc in ((2, 2), (2, -2), (-2, 2), (-2, -2)):
            nr, nc = r + dr, c + dc
            eye = (r + dr // 2, c + dc // 2)
            if on_board(nr, nc) and in_own_half(side, nr) and eye not in board:
                add(nr, nc)
    elif piece == "H":
        for dr, dc, leg in (
            (2, 1, (1, 0)), (2, -1, (1, 0)), (-2, 1, (-1, 0)), (-2, -1, (-1, 0)),
            (1, 2, (0, 1)), (-1, 2, (0, 1)), (1, -2, (0, -1)), (-1, -2, (0, -1)),
        ):
            leg_sq = (r + leg[0], c + leg[1])
            nr, nc = r + dr, c + dc
            if on_board(nr, nc) and leg_sq not in board:
                add(nr, nc)
    elif piece == "R":
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            while on_board(nr, nc):
                occ = board.get((nr, nc))
                if occ is None:
                    targets.append((nr, nc))
                else:
                    if occ[0] != side:
                        targets.append((nr, nc))
                    break
                nr, nc = nr + dr, nc + dc
    elif piece == "C":
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            screen = False
            while on_board(nr, nc):
                occ = board.get((nr, nc))
                if not screen:
                    if occ is None:
                        targets.append((nr, nc))
                    else:
                        screen = True
                else:
                    if occ is not None:
                        if occ[0] != side:
                            targets.append((nr, nc))
                        break
                nr, nc, = nr + dr, nc + dc
    elif piece == "S":
        forward = 1 if side == RED else -1
        add(r + forward, c)
        if not in_own_half(side, r):
            add(r, c + 1)
            add(r, c - 1)
    else:
        raise ValueError(f"Unknown piece {piece}")
    return targets


def _generals_face_off(board):
    red_g, black_g = find_general(board, RED), find_general(board, BLACK)
    if red_g is None or black_g is None:
        return False
    if red_g[1] != black_g[1]:
        return False
    return not any(sq in board for sq in _clear_path(board, red_g[0], red_g[1], black_g[0], black_g[1]))


def is_attacked(board, r, c, by_side):
    for (pr, pc), (s, _p) in board.items():
        if s == by_side and (r, c) in _pseudo_moves(board, pr, pc):
            return True
    return False


def is_in_check(board, side):
    general = find_general(board, side)
    if general is None:
        return True
    return is_attacked(board, general[0], general[1], opposite(side)) or _generals_face_off(board)


def apply_move(board, move):
    (fr, fc), (tr, tc) = move
    new_board = copy.copy(board)
    new_board[(tr, tc)] = new_board.pop((fr, fc))
    return new_board


def pseudo_legal_moves(board, side):
    moves = []
    for (r, c), (s, _p) in list(board.items()):
        if s == side:
            for tr, tc in _pseudo_moves(board, r, c):
                moves.append(((r, c), (tr, tc)))
    return moves


def legal_moves(board, side):
    result = []
    for move in pseudo_legal_moves(board, side):
        if not is_in_check(apply_move(board, move), side):
            result.append(move)
    return result


def legal_destinations(board, side, r, c):
    if board.get((r, c), (None, None))[0] != side:
        return []
    return [tr_tc for (fr, fc), tr_tc in legal_moves(board, side) if (fr, fc) == (r, c)]


def game_status(board, side_to_move):
    """'ongoing', 'checkmate' (side_to_move is mated) or 'stalemate' (no legal moves, also a loss)."""
    if legal_moves(board, side_to_move):
        return "ongoing"
    return "checkmate" if is_in_check(board, side_to_move) else "stalemate"
