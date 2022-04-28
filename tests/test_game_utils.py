import numpy as np
from agents.game_utils import BoardPiece, NO_PLAYER, PLAYER1, PLAYER2, PlayerAction, initialize_game_state


##########################################################################################
# INITIALIZATION

def test_initialize_game_state():
    from agents.game_utils import initialize_game_state

    shape = (6, 7)
    ret = initialize_game_state(shape=shape)

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == shape
    assert np.all(ret == NO_PLAYER)


##########################################################################################
# PRETTY PRINTING

def _make_board():
    board = initialize_game_state(shape=(6, 7))
    p1_pos = [(0, 3), (0, 4), (1, 2), (2, 3), (2, 4), (3, 2), (3, 3)]
    p2_pos = [(0, 1), (0, 2), (1, 1), (1, 3), (1, 4), (2, 2)]
    for pos in p1_pos:
        board[pos] = PLAYER1
    for pos in p2_pos:
        board[pos] = PLAYER2
    return board


printed_board = \
    ("|==============|\n"
     "|              |\n"
     "|              |\n"
     "|    X X       |\n"
     "|    O X X     |\n"
     "|  O X O O     |\n"
     "|  O O X X     |\n"
     "|==============|\n"
     "|0 1 2 3 4 5 6 |")


def test_pretty_print_board():
    from agents.game_utils import pretty_print_board
    prints_to = pretty_print_board(_make_board())
    assert prints_to == printed_board


def test_string_to_board():
    from agents.game_utils import string_to_board
    unprinted_board = string_to_board(printed_board)
    assert (unprinted_board == _make_board()).all()


def test_pretty_print_board_inv_to_string_to_board():
    from agents.game_utils import string_to_board, pretty_print_board
    board = _make_board()
    assert (board == string_to_board(pretty_print_board(board))).all()


##########################################################################################
# PLAYER ACTIONS

def test_apply_player_action():
    from agents.game_utils import apply_player_action
    board = _make_board()
    board_after_action = board.copy()
    action = PlayerAction(1)
    board_after_action[2, action] = PLAYER2
    assert (board_after_action == apply_player_action(board, action, PLAYER2)).all()


def test_apply_player_action_fails_on_out_of_board_action():
    from agents.game_utils import apply_player_action
    failed = False
    board = _make_board()
    try:
        apply_player_action(board, board.shape[1]+5, PLAYER1)
    except ValueError:
        failed = True
    assert failed


def test_apply_player_action_fails_on_too_full_col():
    from agents.game_utils import apply_player_action
    failed = False
    board = _make_board()
    for i in range(board.shape[0]):
        board = apply_player_action(board, PlayerAction(0), PLAYER1)
    try:
        apply_player_action(board, PlayerAction(0), PLAYER1)
    except ValueError:
        failed = True
    assert failed


##########################################################################################
# CHECKING

def test_connected_four_no_4_connected():
    from agents.game_utils import connected_four
    board = _make_board()
    assert not connected_four(board, PLAYER1)
    assert not connected_four(board, PLAYER2)


def test_connected_four_horizontal():
    from agents.game_utils import connected_four
    board = initialize_game_state(shape=(6, 7))
    for i in range(4):
        i = PlayerAction(i)
        board[i, 0] = PLAYER1
    assert connected_four(board, PLAYER1)


def test_connected_four_vertical():
    from agents.game_utils import connected_four
    board = initialize_game_state(shape=(6, 7))
    for i in range(4):
        i = PlayerAction(i)
        board[0, i] = PLAYER1
    assert connected_four(board, PLAYER1)


def test_connected_four_diagonal_ascending():
    from agents.game_utils import connected_four
    board = initialize_game_state(shape=(6, 7))
    for i in range(4):
        i = PlayerAction(i)
        board[i, i] = PLAYER1
    assert connected_four(board, PLAYER1)


def test_connected_four_diagonal_descending():
    from agents.game_utils import connected_four
    board = initialize_game_state(shape=(6, 7))
    for i in range(4):
        i = PlayerAction(i)
        board[4-i, i] = PLAYER1
    assert connected_four(board, PLAYER1)


def test_check_end_state_still_playing_and_win():
    from agents.game_utils import check_end_state, GameState
    board = initialize_game_state(shape=(6, 7))
    assert check_end_state(board, PLAYER1) == GameState.STILL_PLAYING
    assert check_end_state(board, PLAYER2) == GameState.STILL_PLAYING
    for i in range(4):
        i = PlayerAction(i)
        board[0, i] = PLAYER1
    assert check_end_state(board, PLAYER1) == GameState.IS_WIN


def test_check_end_state_draw():
    """
    IS_DRAW could be more elaborate, and detect a state where no win will be possible in the future
    """
    from agents.game_utils import string_to_board, check_end_state, GameState
    draw_board = \
        ("|==============|\n"
         "|O O X X O X O |\n"
         "|X X X O O X X |\n"
         "|O O X X O O X |\n"
         "|X X O X X X O |\n"
         "|X O X O O O X |\n"
         "|O O O X X O O |\n"
         "|==============|\n"
         "|0 1 2 3 4 5 6 |")
    board = string_to_board(draw_board)
    assert check_end_state(board, PLAYER1) == GameState.IS_DRAW
    assert check_end_state(board, PLAYER2) == GameState.IS_DRAW

