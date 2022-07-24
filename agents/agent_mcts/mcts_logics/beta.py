from agents.agent_mcts.mcts_logics import MCTSLogic
from agents.agent_mcts.searchtree import Stats
from agents.game_utils import BoardPiece, PLAYER1, PLAYER2, PlayerAction
from typing import Dict
from scipy.stats import beta
from scipy.special import digamma
import numpy as np
import matplotlib.pyplot as plt


class Beta(MCTSLogic):
    """
    Node statistics based on the property, that the beta distribution is the conjugate prior of a bernoulli experiment
    (like win-vs-lose, possibility of ties is a bit non-represented).
    This gives the option to quantify the uncertainty of the estimates.
    In practice the UCB1 selection mechanism is used, as attempts with variance or quantile based selection mechanisms
    have not proven to work well.
    """

    def __init__(self, c: float = np.sqrt(2), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c

    def select(self, stats: Stats, parent_stats: Stats, player: BoardPiece) -> float:
        # return Beta.quantile(self.quantile, stats['a'], stats['b'])
        # return _mean_p_win(player, stats['a'], stats['b']) + Beta.expected_entropy_change(stats['a'], stats['b'])
        #  --> some ideas, didn't work...
        # so go back to use UCB1 :/
        n = lambda a, b, draw: a + b + draw - 2
        return _mean_p_win(player, stats['a'], stats['b']) + self.c * np.sqrt(np.log(n(**parent_stats)) / n(**stats))

    def update(self, stats: Stats, winner: BoardPiece) -> Stats:
        if stats == {}:
            stats = {'a': 1, 'b': 1, 'draw': 0}
        if winner == PLAYER1:
            stats['a'] += 1
        elif winner == PLAYER2:
            stats['b'] += 1
        else:
            stats['draw'] += 1
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
                p_of_p = Beta.pdf(ps_eval, stats['a'], stats['b'])
                p, = plt.plot(ps, p_of_p, label=f'col: {a}')
                plt.scatter([_mean_p_win(player, stats['a'], stats['b'])], [0], c=p.get_color(), marker='x')
            plt.scatter([], [], c='k', marker='x', label='mean')
            plt.legend()
            plt.gca().set(ylabel='$pdf(p | col, samples)$',
                          xlabel='probability of winning $p$',
                          title='estimated probability densities of winning for possible actions')
            plt.show()
        return {a: _mean_p_win(player, stats['a'], stats['b']) for a, stats in action_stats.items()}

    def merge_stats(self, stats1: Stats, stats2: Stats) -> Stats:
        if stats1 == {}:
            stats1 = {'a': 1, 'b': 1, 'draw': 0}
        if stats2 == {}:
            stats2 = {'a': 1, 'b': 1, 'draw': 0}
        stats = {**stats1}
        stats['a'] += stats2['a'] - 1
        stats['b'] += stats2['b'] - 1
        return stats

    @staticmethod
    def mean(a: int, b: int) -> float:
        return a / (a + b)

    @staticmethod
    def var(a: int, b: int) -> float:
        return a * b / (a + b)**2 / (a + b + 1)

    @staticmethod
    def pdf(p: float, a: int, b: int) -> float:
        return beta.pdf(p, a, b)

    @staticmethod
    def quantile(q: float, a: int, b: int) -> float:
        return beta.ppf(q, a, b)

    @staticmethod
    def delta_entropy_delta_a(a: int, b:int) -> float:
        return np.log(a / (a + b)) - digamma(a) + digamma(a + b) - 1 / (a + b)

    @staticmethod
    def delta_entropy_delta_b(a: int, b: int) -> float:
        return Beta.delta_entropy_delta_a(a=b, b=a)

    @staticmethod
    def expected_entropy_change(a: int, b: int) -> float:
        pr_lose = beta.cdf(0.5, a, b)  # Pr(p < 0.5)
        pr_win = 1 - pr_lose
        return pr_win * Beta.delta_entropy_delta_a(a, b) + pr_lose * Beta.delta_entropy_delta_b(a, b)


def _mean_p_win(player: BoardPiece, a: int, b: int) -> float:
    if player == PLAYER1:
        return Beta.mean(a, b)
    else:
        return 1 - Beta.mean(a, b)
