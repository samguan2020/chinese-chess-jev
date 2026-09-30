"""Repetition-avoidance and material-weighting checks for the xiangqi game server."""
import unittest

import xiangqi_board as xq
from serve_xiangqi import (board_key, filter_repetitive, material_adjustment, LOSING_TRADE_PENALTY,
                           hanging_pieces, threat_adjustment, THREAT_IGNORED_PENALTY,
                           prune_moves, MAX_CANDIDATES)


class RepetitionFilterTests(unittest.TestCase):
    def test_undoing_a_recent_move_is_excluded(self):
        board = {(7, 1): (xq.BLACK, "H"), (9, 4): (xq.BLACK, "G")}
        forward = ((7, 1), (9, 2))
        after_forward = xq.apply_move(board, forward)
        history = [board_key(board)]
        moves = xq.legal_moves(after_forward, xq.BLACK)
        undo = next(m for m in moves if xq.apply_move(after_forward, m) == board)
        kept = filter_repetitive(after_forward, moves, history)
        self.assertNotIn(undo, kept)
        self.assertIn(undo, moves)  # still legal, just filtered out for the AI's own sake

    def test_never_returns_empty_even_if_everything_repeats(self):
        board = {(5, 4): (xq.RED, "H"), (2, 4): (xq.RED, "G")}
        moves = xq.legal_moves(board, xq.RED)
        history = [board_key(xq.apply_move(board, m)) for m in moves]  # every move "repeats" something
        kept = filter_repetitive(board, moves, history, window=len(history))
        self.assertEqual(set(kept), set(moves))

    def test_window_only_looks_at_recent_history(self):
        board = {(7, 1): (xq.BLACK, "H"), (9, 4): (xq.BLACK, "G")}
        forward = ((7, 1), (9, 2))
        after_forward = xq.apply_move(board, forward)
        old_history = [board_key(board)] + [frozenset({("filler", i)}) for i in range(10)]
        moves = xq.legal_moves(after_forward, xq.BLACK)
        undo = next(m for m in moves if xq.apply_move(after_forward, m) == board)
        kept = filter_repetitive(after_forward, moves, old_history, window=8)
        self.assertIn(undo, kept)  # the repeated position fell outside the recency window


class MaterialAdjustmentTests(unittest.TestCase):
    def test_quiet_move_is_unpenalized(self):
        board = {(5, 4): (xq.RED, "S")}
        self.assertEqual(material_adjustment(board, ((5, 4), (6, 4)), xq.RED), 1.0)

    def test_free_capture_is_unpenalized(self):
        board = {(5, 4): (xq.RED, "R"), (6, 4): (xq.BLACK, "S")}
        self.assertEqual(material_adjustment(board, ((5, 4), (6, 4)), xq.RED), 1.0)

    def test_severely_losing_trade_is_near_vetoed(self):
        # Rook takes a Soldier but a second Black Rook recaptures: gives up 9 for 1.
        board = {(5, 4): (xq.RED, "R"), (6, 4): (xq.BLACK, "S"), (7, 4): (xq.BLACK, "R")}
        self.assertEqual(material_adjustment(board, ((5, 4), (6, 4)), xq.RED), LOSING_TRADE_PENALTY)

    def test_mildly_losing_trade_is_also_near_vetoed(self):
        # Cannon takes a Horse but is recaptured: only gives up 4.5 for 4, a mild loss on
        # paper, but "any losing trade" means it gets crushed just as hard as a severe one.
        board = {(7, 1): (xq.BLACK, "C"), (0, 1): (xq.RED, "H"), (2, 1): (xq.RED, "C"), (0, 0): (xq.RED, "R")}
        self.assertEqual(material_adjustment(board, ((7, 1), (0, 1)), xq.BLACK), LOSING_TRADE_PENALTY)

    def test_favorable_trade_is_unpenalized_even_if_recapturable(self):
        # Soldier takes a Rook and a second Black Rook recaptures: still nets +8 for Red.
        board = {(5, 4): (xq.RED, "S"), (6, 4): (xq.BLACK, "R"), (8, 4): (xq.BLACK, "R")}
        self.assertEqual(material_adjustment(board, ((5, 4), (6, 4)), xq.RED), 1.0)

    def test_penalty_is_never_exactly_zero(self):
        # Even an extreme loss (General for Soldier) stays pickable if nothing else is on offer.
        board = {(5, 4): (xq.RED, "G"), (6, 4): (xq.BLACK, "S"), (7, 4): (xq.BLACK, "R")}
        adjustment = material_adjustment(board, ((5, 4), (6, 4)), xq.RED)
        self.assertEqual(adjustment, LOSING_TRADE_PENALTY)
        self.assertGreater(adjustment, 0.0)


class HangingPieceTests(unittest.TestCase):
    def test_undefended_piece_under_attack_is_hanging(self):
        board = {(3, 1): (xq.BLACK, "C"), (0, 1): (xq.RED, "R")}
        self.assertEqual(hanging_pieces(board, xq.BLACK), [(3, 1)])

    def test_adequately_defended_piece_is_not_hanging(self):
        # Rook attacked by an enemy Rook, but a Soldier of matching-or-better trade value guards it.
        board = {(3, 1): (xq.BLACK, "C"), (0, 1): (xq.RED, "R"), (4, 1): (xq.BLACK, "S")}
        self.assertEqual(hanging_pieces(board, xq.BLACK), [])

    def test_defended_piece_still_hanging_if_cheapest_attacker_is_cheaper(self):
        # A Rook guarded by a Soldier is still a bad trade for Black: Red's Soldier for Black's Rook.
        board = {(5, 4): (xq.BLACK, "R"), (4, 4): (xq.RED, "S"), (6, 4): (xq.BLACK, "S")}
        self.assertEqual(hanging_pieces(board, xq.BLACK), [(5, 4)])


class ThreatAdjustmentTests(unittest.TestCase):
    def test_no_penalty_when_nothing_is_hanging(self):
        board = {(3, 1): (xq.BLACK, "C")}
        self.assertEqual(threat_adjustment(board, ((3, 1), (3, 5)), xq.BLACK, hanging_before=[]), 1.0)

    def test_move_that_relocates_the_hanging_piece_is_unpenalized(self):
        board = {(3, 1): (xq.BLACK, "C"), (0, 1): (xq.RED, "R"), (6, 0): (xq.BLACK, "S")}
        hanging_before = hanging_pieces(board, xq.BLACK)
        move = ((3, 1), (3, 5))  # cannon slides away to safety
        self.assertEqual(threat_adjustment(board, move, xq.BLACK, hanging_before), 1.0)

    def test_move_that_ignores_the_hanging_piece_is_penalized(self):
        board = {(3, 1): (xq.BLACK, "C"), (0, 1): (xq.RED, "R"), (6, 0): (xq.BLACK, "S")}
        hanging_before = hanging_pieces(board, xq.BLACK)
        move = ((6, 0), (5, 0))  # unrelated soldier push; the cannon is still hanging
        self.assertEqual(threat_adjustment(board, move, xq.BLACK, hanging_before), THREAT_IGNORED_PENALTY)


class PruneMovesHangingPriorityTests(unittest.TestCase):
    def test_a_move_that_resolves_a_hanging_piece_survives_random_pruning(self):
        board = {(3, 1): (xq.BLACK, "C"), (0, 1): (xq.RED, "R"), (6, 0): (xq.BLACK, "S")}
        hanging_before = hanging_pieces(board, xq.BLACK)
        self.assertTrue(hanging_before)
        rescue = ((3, 1), (3, 5))
        padding = [((6, 0), (8, c)) for c in range(9)] + [((6, 0), (1, c)) for c in range(9)]
        moves = [rescue] + padding
        self.assertGreater(len(moves), MAX_CANDIDATES)
        # Run several times: without the fix this is a coin flip, so a single pass could pass by luck.
        for _ in range(20):
            pruned = prune_moves(board, moves, side=xq.BLACK, hanging_before=hanging_before)
            self.assertIn(rescue, pruned)
            self.assertLessEqual(len(pruned), MAX_CANDIDATES)

    def test_no_hanging_pieces_falls_back_to_plain_capture_priority_pruning(self):
        board = {(5, 4): (xq.RED, "R"), (6, 4): (xq.BLACK, "S")}
        capture = ((5, 4), (6, 4))
        padding = [((5, 4), (5, c)) for c in range(4)] + [((5, 4), (4, 4))] * 10
        moves = [capture] + padding
        pruned = prune_moves(board, moves, side=xq.RED, hanging_before=[])
        self.assertIn(capture, pruned)


if __name__ == "__main__":
    unittest.main()
