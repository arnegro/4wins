import numpy as np
from agents.agent_mcts.searchtree import Stats
from agents.game_utils import PLAYER2, PLAYER1, BoardPiece, PlayerAction
from typing import Dict
from agents.agent_mcts.mcts_logics import MCTSLogic


class UCB1(MCTSLogic):

    def __init__(self, c: float = np.sqrt(2), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c

    def select(self, stats: Stats, parent_stats: Stats, player: BoardPiece) -> float:
        return _frac_wins(stats, player) + self.c * np.sqrt(np.log(parent_stats['s']) / stats['s'])

    def update(self, stats: Stats, winner: BoardPiece) -> Stats:
        if stats == {}:
            stats = {'w': 0, 's': 0}
        if winner == PLAYER1:
            stats['w'] += 1
        stats['s'] += 1
        return stats

    def _action_choice_metric(
            self, action_stats: Dict[PlayerAction, Stats], player: BoardPiece
    ) -> Dict[PlayerAction, float]:
        return {a: _frac_wins(stats, player) for a,stats in action_stats.items()}


def _frac_wins(stats: Stats, player: BoardPiece) -> float:
    w = stats['w']
    if player == PLAYER2:
        w = stats['s'] - w
    return w / stats['s']
