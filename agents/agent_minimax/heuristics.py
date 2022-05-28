import numpy as np
from scipy.signal import convolve2d
from agents.agent_minimax.minimax import Value
from agents.game_utils import BoardPiece, PLAYER1, PLAYER2, check_end_state, GameState
import numba


def base_heuristic(board: np.ndarray) -> Value:
    for p in [PLAYER1, PLAYER2]:
        if check_end_state(board, p) == GameState.IS_WIN:
            return np.inf if p == PLAYER1 else -np.inf
    threes = _x_in_a_row(board, PLAYER1, 3) - _x_in_a_row(board, PLAYER2, 3)
    twos = _x_in_a_row(board, PLAYER1, 2) - _x_in_a_row(board, PLAYER2, 2)
    return threes**2 + twos


def _x_in_a_row2(board: np.ndarray, player: BoardPiece, x: int) -> int:
    board = (board == player)
    kernels = [np.ones((x, 1)),  # horizontals
               np.ones((1, x)),  # verticals
               np.eye(x),  # ascending diagonals
               np.fliplr(np.eye(x))  # descending diagonals
               ]
    n = 0
    for kernel in kernels:
        conv = convolve2d(board, kernel, mode='valid')
        n += (conv == x).sum()
    return n


#@numba.njit()
def _x_in_a_row(board: np.ndarray, player: BoardPiece, x: int) -> int:
    rows, cols = board.shape
    board = board == player

    n = 0
    for i in range(rows):
        for j in range(cols - x + 1):
            n += np.all(board[i, j:j + x])

    for i in range(rows - x + 1):
        for j in range(cols):
            n += np.all(board[i:i + x, j])

    for i in range(rows - x + 1):
        for j in range(cols - x + 1):
            block = board[i:i + x, j:j + x]
            n += np.all(np.diag(block))
            n += np.all(np.diag(block[::-1, :]))

    return n
