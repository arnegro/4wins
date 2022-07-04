import numpy as np
from agents.game_utils import BoardPiece, PlayerAction, SavedState, NO_PLAYER
from typing import Optional, Tuple, List


def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    legal_moves = _get_legal_moves(board)
    action = PlayerAction(np.random.choice(legal_moves))
    return action, saved_state


# conceptually this function could imo be part of `game_utils`
# the game_utils interface should stay unchanged in this setting though to makes agents transferable
# --> hence the code copy
def _get_legal_moves(board: np.ndarray) -> List[PlayerAction]:
    """Takes a board and returns all possible columns in which a move could be applied"""
    _, cols = board.shape
    mvs = [PlayerAction(col) for col in range(cols) if board[-1, col] == NO_PLAYER]
    return mvs
