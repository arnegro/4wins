from enum import Enum
import numpy as np
from typing import List

BoardPiece = np.int8  # The data type (dtype) of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 (player to move first) has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 (player to move second) has a piece

BoardPiecePrint = str  # dtype for string representation of BoardPiece
NO_PLAYER_PRINT = BoardPiecePrint(' ')
PLAYER1_PRINT = BoardPiecePrint('X')
PLAYER2_PRINT = BoardPiecePrint('O')
_FIELD_REPR = {PLAYER1: PLAYER1_PRINT,
               PLAYER2: PLAYER2_PRINT,
               NO_PLAYER: NO_PLAYER_PRINT}

PlayerAction = np.int8  # The column to be played


class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0


##########################################################################################
# INITIALIZATION

def initialize_game_state(shape=(6, 7)) -> np.ndarray:
    """
    Returns an ndarray, shape (6, 7) and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).
    """
    board = np.empty(shape, dtype=BoardPiece)
    board.fill(NO_PLAYER)
    return board


##########################################################################################
# PRETTY PRINTING

def pretty_print_board(board: np.ndarray) -> str:
    """
    Should return `board` converted to a human readable string representation,
    to be used when playing or printing diagnostics to the console (stdout). The piece in
    board[0, 0] should appear in the lower-left. Here's an example output, note that we use
    PLAYER1_Print to represent PLAYER1 and PLAYER2_Print to represent PLAYER2):
    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |
    """
    board_repr = [' '.join([_FIELD_REPR[field] for field in row]) + ' ' for row in board]
    board_repr.reverse()
    num_cols = board.shape[1]
    bar = num_cols*2*'='
    nums = ' '.join(str(col) for col in range(num_cols)) + ' '
    rows = [bar] + board_repr + [bar, nums]
    pp_board = '\n'.join(['|'+row+'|' for row in rows])
    return pp_board


def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.
    """
    _FIELD_REPR_INV = {val: key for key,val in _FIELD_REPR.items()}
    rows = pp_board.split('\n')
    rows = rows[1:-2]  # drop bars and nums
    board_repr = [row.strip('|')[::2] for row in rows]  # drop bars and interspersed spaces
    board = [[_FIELD_REPR_INV[field] for field in row] for row in board_repr]
    board.reverse()
    return np.array(board)


##########################################################################################
# PLAYER ACTIONS

def apply_player_action(board: np.ndarray, action: PlayerAction, player: BoardPiece) -> np.ndarray:
    """
    Sets board[i, action] = player, where i is the lowest open row. Raises a ValueError
    if action is not a legal move. If it is a legal move, the modified version of the
    board is returned and the original board should remain unchanged (i.e., either set
    back or copied beforehand).
    """
    if action not in _legal_moves(board):
        raise ValueError(f'column {action} is not an admissible action')
    col = board[:, action]
    fill_level = np.argmin(np.where(col == NO_PLAYER, 0, 1))
    board = board.copy()
    board[fill_level, action] = player
    return board


def _legal_moves(board: np.ndarray) -> List[PlayerAction]:
    _, cols = board.shape
    return [PlayerAction(col) for col in range(cols) if board[-1, col] == NO_PLAYER]


##########################################################################################
# CHECKING

def connected_four(board: np.ndarray, player: BoardPiece) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    """
    board_pl = board == player
    return _check_horizontal(board_pl) or _check_vertical(board_pl) or _check_diagonal(board_pl)


def _check_horizontal(board: np.ndarray) -> bool:
    for row in board:
        for col in range(row.shape[0]-4):
            if (row[col: col+4]).all():
                return True
    return False


def _check_vertical(board: np.ndarray) -> bool:
    return _check_horizontal(board.T)


def _check_diagonal(board: np.ndarray) -> bool:
    # shift columns so that diagonal connections become straight ones
    rolled_boards = []
    for roll_direction in [1, -1]:  # shift left to get descending diagonal, right for ascending
        rolled_board = np.array([np.roll(board[i], roll_direction*i) for i in range(len(board))])
        rolled_boards.append(rolled_board)
    return np.any([_check_vertical(board) for board in rolled_boards])


def check_end_state(board: np.ndarray, player: BoardPiece) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still ongoing (GameState.STILL_PLAYING)?
    """
    won = connected_four(board, player)
    board_full = len(_legal_moves(board)) == 0
    if won:
        return GameState.IS_WIN
    elif board_full:
        return GameState.IS_DRAW
    else:
        return GameState.STILL_PLAYING
