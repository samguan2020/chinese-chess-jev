"""Board/move-to-text encoding checks for the xiangqi NanoJev integration."""
import unittest

import xiangqi_board as xq
import xiangqi_notation as notation


class NotationTests(unittest.TestCase):
    def test_render_board_text_mentions_turn_and_all_pieces(self):
        board = xq.initial_board()
        text = notation.render_board_text(board, xq.RED)
        self.assertIn("红方", text)
        self.assertEqual(text.count("车"), 4)
        self.assertEqual(text.count("将"), 1)
        self.assertEqual(text.count("帅"), 1)

    def test_describe_move_reports_capture(self):
        board = {(5, 4): (xq.RED, "S"), (6, 4): (xq.BLACK, "S")}
        desc = notation.describe_move(board, ((5, 4), (6, 4)))
        self.assertIn("兵", desc)
        self.assertIn("吃掉", desc)
        self.assertIn("卒", desc)

    def test_moves_to_choice_criteria_round_trips(self):
        board = xq.initial_board()
        moves = xq.legal_moves(board, xq.RED)
        by_id, criteria = notation.moves_to_choice_criteria(board, moves)
        self.assertEqual(len(by_id), len(moves))
        self.assertEqual(set(by_id), set(criteria))
        for mid, move in by_id.items():
            self.assertIn(move, moves)
            self.assertTrue(criteria[mid])


if __name__ == "__main__":
    unittest.main()
