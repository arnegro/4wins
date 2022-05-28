import numpy as np
from agents.game_utils import BoardPiece, PlayerAction, SavedState, get_legal_moves
from typing import Optional, Tuple


def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    legal_moves = get_legal_moves(board)
    action = PlayerAction(np.random.choice(legal_moves))
    return action, saved_state
