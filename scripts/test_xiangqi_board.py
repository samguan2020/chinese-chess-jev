"""Xiangqi rules engine checks: movement patterns, blocking, check/mate detection."""
import unittest

import xiangqi_board as xq


class SetupTests(unittest.TestCase):
    def test_initial_setup_piece_counts_and_symmetry(self):
        board = xq.initial_board()
        self.assertEqual(len(board), 32)
        by_side = {xq.RED: 0, xq.BLACK: 0}
        for side, piece in board.values():
            by_side[side] += 1
        self.assertEqual(by_side, {xq.RED: 16, xq.BLACK: 16})
        self.assertFalse(xq.is_in_check(board, xq.RED))
        self.assertFalse(xq.is_in_check(board, xq.BLACK))

    def test_opening_move_count_is_stable(self):
        board = xq.initial_board()
        self.assertEqual(len(xq.legal_moves(board, xq.RED)), 44)
        self.assertEqual(len(xq.legal_moves(board, xq.BLACK)), 44)


class HorseTests(unittest.TestCase):
    def test_leg_block_removes_two_of_eight_destinations(self):
        board = {(5, 4): (xq.RED, "H")}
        self.assertEqual(len(xq._pseudo_moves(board, 5, 4)), 8)
        board[(4, 4)] = (xq.BLACK, "S")  # blocks the northward leg
        moves = xq._pseudo_moves(board, 5, 4)
        self.assertEqual(len(moves), 6)
        self.assertNotIn((3, 3), moves)
        self.assertNotIn((3, 5), moves)


class ElephantTests(unittest.TestCase):
    def test_blocked_by_eye_and_cannot_cross_river(self):
        board = {(2, 2): (xq.RED, "E")}
        self.assertEqual(set(xq._pseudo_moves(board, 2, 2)), {(0, 0), (0, 4), (4, 0), (4, 4)})
        board[(3, 3)] = (xq.BLACK, "S")  # occupies the eye toward (4,4)
        moves = xq._pseudo_moves(board, 2, 2)
        self.assertNotIn((4, 4), moves)
        self.assertIn((4, 0), moves)

    def test_river_bound(self):
        board = {(4, 2): (xq.RED, "E")}
        self.assertNotIn((6, 4), xq._pseudo_moves(board, 4, 2))
        self.assertNotIn((6, 0), xq._pseudo_moves(board, 4, 2))


class CannonTests(unittest.TestCase):
    def test_needs_exactly_one_screen_to_capture(self):
        board = {(5, 0): (xq.RED, "C"), (5, 4): (xq.BLACK, "S"), (5, 8): (xq.BLACK, "R")}
        moves = xq._pseudo_moves(board, 5, 0)
        self.assertIn((5, 1), moves)
        self.assertIn((5, 3), moves)
        self.assertNotIn((5, 4), moves)  # cannot land on the screen itself
        self.assertIn((5, 8), moves)  # exactly one screen away: legal capture
        self.assertFalse(any(c > 8 for _r, c in moves))

    def test_two_screens_block_capture(self):
        board = {(5, 0): (xq.RED, "C"), (5, 3): (xq.BLACK, "S"), (5, 5): (xq.BLACK, "S"), (5, 8): (xq.BLACK, "R")}
        moves = xq._pseudo_moves(board, 5, 0)
        self.assertNotIn((5, 8), moves)


class SoldierTests(unittest.TestCase):
    def test_forward_only_before_river_then_sideways_after(self):
        board = {(3, 4): (xq.RED, "S")}
        self.assertEqual(set(xq._pseudo_moves(board, 3, 4)), {(4, 4)})
        board = {(5, 4): (xq.RED, "S")}
        self.assertEqual(set(xq._pseudo_moves(board, 5, 4)), {(6, 4), (5, 3), (5, 5)})
        self.assertNotIn((4, 4), xq._pseudo_moves(board, 5, 4))  # never backward


class GeneralTests(unittest.TestCase):
    def test_confined_to_palace(self):
        board = {(2, 4): (xq.RED, "G")}
        self.assertEqual(set(xq._pseudo_moves(board, 2, 4)), {(1, 4), (2, 3), (2, 5)})
        board = {(2, 3): (xq.RED, "G")}
        self.assertNotIn((2, 2), xq._pseudo_moves(board, 2, 3))

    def test_flying_general_exposure_is_illegal(self):
        board = {(0, 4): (xq.RED, "G"), (9, 4): (xq.BLACK, "G")}
        self.assertTrue(xq._generals_face_off(board))
        self.assertTrue(xq.is_in_check(board, xq.RED))
        self.assertTrue(xq.is_in_check(board, xq.BLACK))

    def test_move_that_would_expose_generals_is_filtered_out(self):
        board = {(0, 4): (xq.RED, "G"), (9, 4): (xq.BLACK, "G"), (5, 4): (xq.RED, "S")}
        moves = xq.legal_moves(board, xq.RED)
        self.assertIn(((5, 4), (6, 4)), moves)  # stays on the shared file: still screens the generals
        self.assertNotIn(((5, 4), (5, 3)), moves)  # steps off the file: would expose them
        self.assertNotIn(((5, 4), (5, 5)), moves)


class CheckmateTests(unittest.TestCase):
    def test_three_chariots_mate_lone_general(self):
        board = {
            (9, 4): (xq.BLACK, "G"),
            (6, 3): (xq.RED, "R"),
            (6, 4): (xq.RED, "R"),
            (6, 5): (xq.RED, "R"),
        }
        self.assertTrue(xq.is_in_check(board, xq.BLACK))
        self.assertEqual(xq.legal_moves(board, xq.BLACK), [])
        self.assertEqual(xq.game_status(board, xq.BLACK), "checkmate")

    def test_ongoing_when_moves_remain(self):
        board = xq.initial_board()
        self.assertEqual(xq.game_status(board, xq.RED), "ongoing")


if __name__ == "__main__":
    unittest.main()
