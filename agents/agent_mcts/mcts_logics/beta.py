from agents.agent_mcts.mcts_logics import MCTSLogic
from agents.agent_mcts.searchtree import Stats
from agents.game_utils import BoardPiece, PLAYER1, PlayerAction
from typing import Dict
from scipy.special import gamma, digamma
import numpy as np
import matplotlib.pyplot as plt


class Beta(MCTSLogic):

    def __init__(self, c: float = np.sqrt(2), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c

    def select(self, stats: Stats, parent_stats: Stats, player: BoardPiece) -> float:
        return _mean_p_win(player, **stats) + self.c*np.sqrt(Beta.var(**stats))

    def update(self, stats: Stats, winner: BoardPiece) -> Stats:
        if stats == {}:
            stats = {'a': 1, 'b': 1}
        if winner == PLAYER1:
            stats['a'] += 1
        else:
            stats['b'] += 1
        return stats

    def _action_choice_metric(
            self, action_stats: Dict[PlayerAction, Stats], player: BoardPiece
    ) -> Dict[PlayerAction, float]:
        # this plotting is just uglily hacked in -- no nice code structuring here
        # should be overriding MCTSLogic._show_action_vals, that has only the values and not the parameters in the
        # dict though so would need to be refactored -- hasn't happened yet
        ps = np.linspace(0, 1, 100)
        for a, stats in action_stats.items():
            plt.plot(ps, Beta.pdf(ps, **stats), label=f'col: {a}')
        plt.legend()
        plt.show()
        return {a: _mean_p_win(player, **stats) for a, stats in action_stats.items()}

    @staticmethod
    def mean(a: int, b: int):
        return a / (a + b)

    @staticmethod
    def var(a: int, b: int):
        return a * b / (a + b)**2 / (a + b + 1)

    @staticmethod
    def pdf(p: float, a: int, b: int):
        B = gamma(a) * gamma(b) / gamma(a + b)
        return p**(a-1) * (1 - p)**(b-1) / B


def _mean_p_win(player: BoardPiece, a: int, b: int) -> float:
    if player == PLAYER1:
        return Beta.mean(a, b)
    else:
        return 1 - Beta.mean(a, b)
