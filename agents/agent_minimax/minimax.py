import numpy as np
from agents.game_utils import \
    BoardPiece, PLAYER1, PLAYER2, PlayerAction, NO_PLAYER, SavedState, \
    check_end_state, GameState, apply_player_action
from typing import Optional, Tuple, Callable, List
import time


Value = float
Heuristic = Callable[[np.ndarray], Value]


def generate_move_minimax(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState], depth: int,
    heuristic: Heuristic, max_time: Optional[float]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Function to generate a move based on the minimax algorithm.

    :param board: current game state
    :param player: current player
    :param saved_state: N/A in this case, simply for signature reasons
    :param depth: depth of the underlying minimax algorithm
    :param heuristic: evaluate function of game states in the leave nodes of the minimax algorithm
    :param max_time: if not None the agent will iteratively deepen from depth until max_time is reached
                     -- this is not fail-safe!
    :return: the optimal action given `depth` and `heuristic` as well as the passed and unused `saved_state`
    """
    t0 = time.time()
    actions = _get_legal_moves(board)
    values = []
    best_action = actions[0]
    ts = []
    while True:
        for action in actions:
            t02 = time.time()
            child = apply_player_action(board, action, player)
            value = alpha_beta(child, depth, -np.inf, np.inf, _OTHER_PLAYER[player], heuristic)
            values.append((action, value))
            ts += [time.time() - t02]
            if max_time is not None and time.time() - t0 > max_time - max(ts) - 0.1:  # break iterative deepening
                return best_action, saved_state
        f = max if player == PLAYER1 else min
        best_action = f(values, key=lambda av: av[1])[0]
        if max_time is None:  # no iterative deepening
            return best_action, saved_state
        else:  # iteratively deepen
            depth += 1


_OTHER_PLAYER = {PLAYER1: PLAYER2, PLAYER2: PLAYER1}


# conceptually this function could imo be part of `game_utils`
# the game_utils interface should stay unchanged in this setting though to makes agents transferable
# --> hence the code copy
def _get_legal_moves(board: np.ndarray) -> List[PlayerAction]:
    """Takes a board and returns all possible columns in which a move could be applied"""
    _, cols = board.shape
    mvs = [PlayerAction(col) for col in range(cols) if board[-1, col] == NO_PLAYER]
    mvs = sorted(mvs, key=lambda a: (a - cols/2)**2)  # focus first on center columns
    return mvs


def alpha_beta(
    board: np.array, depth: int, a: float, b: float, player: BoardPiece, heuristic: Heuristic
) -> Value:
    """
    Minimax algorithm with alpha-beta-pruning.

    :param board: current game state
    :param depth: depth of the underlying minimax algorithm
    :param a: alpha value
    :param b: beta value
    :param player: current player
    :param heuristic: evaluate function of game states in the leave nodes of the minimax algorithm
    :return: value of the best action given `board`
    """
    if depth <= 0 or check_end_state(board, player) != GameState.STILL_PLAYING:
        return heuristic(board)
    color = 1 if player == PLAYER1 else -1
    value = -color*np.inf
    for action in _get_legal_moves(board):
        child = apply_player_action(board, action, player)
        value = max(color*value,
                    color*alpha_beta(child, depth-1, a, b, _OTHER_PLAYER[player], heuristic))
        value *= color
        a, b, prune = _update_alpha_beta(a, b, value, player)
        if prune:
            break
    return value


def _update_alpha_beta(a: float, b: float,
                       value: Value,
                       player: BoardPiece) \
        -> Tuple[float, float, Value]:

    if player == PLAYER1:
        a = max(a, value)
        prune = value >= b
    else:
        b = min(b, value)
        prune = value <= a

    return a, b, prune
