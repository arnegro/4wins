from agents.agent_mcts.mcts_logics import MCTSLogic
from agents.agent_mcts.searchtree import Stats
from agents.game_utils import BoardPiece, PLAYER1, PlayerAction
from typing import Dict
from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt


class Beta(MCTSLogic):

    def __init__(self, c: float = np.sqrt(2), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c

    def select(self, stats: Stats, parent_stats: Stats, player: BoardPiece) -> float:
        n = lambda a, b: a + b - 2
        # basically UCB1... --> find a way to use variance information nicely
        return _mean_p_win(player, **stats) + self.c * np.sqrt(np.log(n(**parent_stats)) / n(**stats))

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
        if self.verbose:
            ps = np.linspace(0, 1, 102)[1:-1]
            if player != PLAYER1:
                ps_eval = ps[::-1]
            else:
                ps_eval = ps
            for a, stats in sorted(action_stats.items()):
                p_of_p = Beta.pdf(ps_eval, **stats)
                p, = plt.plot(ps, p_of_p, label=f'col: {a}')
                plt.scatter([_mean_p_win(player, **stats)], [0], c=p.get_color(), marker='x')
            plt.scatter([], [], c='k', marker='x', label='mean')
            plt.legend()
            plt.gca().set(ylabel='$pdf(p | col, samples)$',
                          xlabel='probability of winning $p$',
                          title='estimated probability densities of winning for possible actions')
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
        return beta.pdf(p, a, b)


def _mean_p_win(player: BoardPiece, a: int, b: int) -> float:
    if player == PLAYER1:
        return Beta.mean(a, b)
    else:
        return 1 - Beta.mean(a, b)
