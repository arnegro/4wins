import numpy as np
from agents.game_utils import \
    BoardPiece, PLAYER1, PLAYER2, PlayerAction, NO_PLAYER, SavedState, \
    check_end_state, GameState, apply_player_action, pretty_print_board
from typing import Optional, Tuple, Callable, List


Value = float
Heuristic = Callable[[np.ndarray], Value]


def generate_move_minimax(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState], depth: int, heuristic: Heuristic
) -> Tuple[PlayerAction, Optional[SavedState]]:
    actions = get_legal_moves(board)
    values = []
    for action in actions:
        child = apply_player_action(board, action, player)
        value = alpha_beta(child, depth, -np.inf, np.inf, _OTHER_PLAYER[player], heuristic)
        values.append((action, value))
    f = max if player == PLAYER1 else min
    action = f(values, key=lambda av: av[1])[0]
    return action, saved_state


_OTHER_PLAYER = {PLAYER1: PLAYER2, PLAYER2: PLAYER1}


def get_legal_moves(board: np.ndarray) -> List[PlayerAction]:
    _, cols = board.shape
    mvs = [PlayerAction(col) for col in range(cols) if board[-1, col] == NO_PLAYER]
    mvs = sorted(mvs, key=lambda a: (a - cols/2)**2)  # focus first on center columns
    return mvs


def alpha_beta(
    board: np.array, depth: int, a: float, b: float, player: BoardPiece, heuristic: Heuristic
) -> Value:

    if depth <= 0 or check_end_state(board, player) != GameState.STILL_PLAYING:
        return heuristic(board)

    if player == PLAYER1:
        value = -np.inf
        for action in get_legal_moves(board):
            child = apply_player_action(board, action, player)
            value = max(value, alpha_beta(child, depth-1, a, b, _OTHER_PLAYER[player], heuristic))
            a = max(a, value)
            if value >= b:
                break
        return value
    else:
        value = np.inf
        for action in get_legal_moves(board):
            child = apply_player_action(board, action, player)
            value = min(value, alpha_beta(child, depth-1, a, b, _OTHER_PLAYER[player], heuristic))
            b = min(b, value)
            if value <= a:
                break
        return value
