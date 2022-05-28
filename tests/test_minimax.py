import numpy as np
from agents.game_utils import string_to_board, PLAYER1, PLAYER2, NO_PLAYER, initialize_game_state

printed_board = \
    ("|==============|\n"
     "|              |\n"
     "|              |\n"
     "|              |\n"
     "|      X       |\n"
     "|      X       |\n"
     "|      X       |\n"
     "|==============|\n"
     "|0 1 2 3 4 5 6 |")
board = string_to_board(printed_board)


def test_alpha_beta_takes_win_pl1():
    from agents.agent_minimax.minimax import generate_move_minimax
    from agents.agent_minimax import base_heuristic
    action, _ = generate_move_minimax(board, PLAYER1, saved_state=None,
                                      depth=3, heuristic=base_heuristic)
    assert action == 3


def test_alpha_beta_prevent_lose_pl2():
    from agents.agent_minimax.minimax import generate_move_minimax
    from agents.agent_minimax import base_heuristic
    action, _ = generate_move_minimax(board, PLAYER2, saved_state=None,
                                      depth=3, heuristic=base_heuristic)
    assert action == 3


def test_alpha_beta_takes_win_pl2():
    from agents.agent_minimax.minimax import generate_move_minimax
    from agents.agent_minimax import base_heuristic
    inv_board = initialize_game_state(board.shape)
    inv_board[board == PLAYER1] = PLAYER2
    inv_board[board == PLAYER2] = PLAYER1
    action, _ = generate_move_minimax(inv_board, PLAYER2, saved_state=None,
                                      depth=3, heuristic=base_heuristic)
    assert action == 3


def test_alpha_beta_prevent_lose_pl1():
    from agents.agent_minimax.minimax import generate_move_minimax
    from agents.agent_minimax import base_heuristic
    inv_board = initialize_game_state(board.shape)
    inv_board[board == PLAYER1] = PLAYER2
    inv_board[board == PLAYER2] = PLAYER1
    action, _ = generate_move_minimax(inv_board, PLAYER1, saved_state=None,
                                      depth=3, heuristic=base_heuristic)
    assert action == 3


def test_x_in_a_row():
    from agents.agent_minimax.heuristics import _x_in_a_row
    complex_printed_board = \
        ("|==============|\n"
         "|              |\n"
         "|      X       |\n"
         "|    X X       |\n"
         "|    O X X     |\n"
         "|  O X O O     |\n"
         "|  O O X X O O |\n"
         "|==============|\n"
         "|0 1 2 3 4 5 6 |")
    complex_board = string_to_board(complex_printed_board)
    assert _x_in_a_row(complex_board, PLAYER1, 3) == 1,  'Player 1, 3'
    assert _x_in_a_row(complex_board, PLAYER2, 3) == 0,  'Player 2, 3'
    assert _x_in_a_row(complex_board, PLAYER1, 2) == 10, 'Player 2, 2'
    assert _x_in_a_row(complex_board, PLAYER2, 2) == 9,  'Player 2, 2'
