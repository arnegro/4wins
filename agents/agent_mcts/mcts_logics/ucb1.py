import numpy as np
from agents.agent_mcts.searchtree import Stats
from agents.game_utils import PLAYER2, PLAYER1, NO_PLAYER, BoardPiece, PlayerAction
from typing import Dict
from agents.agent_mcts.mcts_logics import MCTSLogic


class UCB1(MCTSLogic):
    """
    Upper Confidence Bound 1 (UCB1) selection and choice logic.
    """

    def __init__(self, c: float = np.sqrt(2), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c

    def select(self, stats: Stats, parent_stats: Stats, player: BoardPiece) -> float:
        return _frac_wins(stats, player) + self.c * np.sqrt(np.log(parent_stats['s']) / stats['s'])

    def update(self, stats: Stats, winner: BoardPiece) -> Stats:
        if stats == {}:
            stats = {'w': 0, 's': 0, 'draw': 0}
        if winner == NO_PLAYER:
            stats['draw'] += 1
        if winner == PLAYER1:
            stats['w'] += 1
        stats['s'] += 1
        return stats

    def _action_choice_metric(
            self, action_stats: Dict[PlayerAction, Stats], player: BoardPiece
    ) -> Dict[PlayerAction, float]:
        return {a: _frac_wins(stats, player) for a, stats in action_stats.items()}

    def merge_stats(self, stats1: Stats, stats2: Stats) -> Stats:
        if stats1 == {}:
            stats1 = {'w': 0, 's': 0, 'draw': 0}
        if stats2 == {}:
            stats2 = {'w': 0, 's': 0, 'draw': 0}
        stats = {**stats1}
        stats['w'] += stats2['w']
        stats['s'] += stats2['s']
        return stats


def _frac_wins(stats: Stats, player: BoardPiece) -> float:
    w = stats['w']
    if player == PLAYER2:
        w = stats['s'] - w
    n = stats['s'] - stats['draw']
    if n == 0:
        return 0
    else:
        return w / n
