from typing import Dict

from agents.agent_mcts.searchtree import Stats
from agents.game_utils import BoardPiece, PlayerAction
import time
import numpy as np


class MCTSLogic:

    def __init__(
            self, time_based: bool = False, iterations: int = 1000,
            runtime: float = 5, verbose: bool = False,
            progressbar: bool = False
    ):
        self.iterations = iterations
        self.verbose = verbose
        self.time_based = time_based
        self.runtime = runtime  # s
        self.progressbar = progressbar
        self._len_progressbar: int = 20
        self._exit_time_buffer: float = 0.01  # s

    def select(self, stats: Stats, parent_stats: Stats, player: BoardPiece) -> float:
        pass

    def update(self, stats: Stats, winner: BoardPiece) -> Stats:
        pass

    def _action_choice_metric(
            self, action_stats: Dict[PlayerAction, Stats], player: BoardPiece
    ) -> Dict[PlayerAction, float]:
        pass

    def choose_action(self, action_stats: Dict[PlayerAction, Stats], player: BoardPiece) -> PlayerAction:
        action_values = self._action_choice_metric(action_stats, player)
        if self.verbose:
            print(self._show_action_vals(action_values))
        return max(action_values, key=action_values.get)

    @staticmethod
    def _show_action_vals(action_values: Dict[PlayerAction, float]) -> str:
        return 'move estimate: ' + '; '.join(f'{a}->{v:.4f}' for a, v in sorted(action_values.items()))

    def get_iter(self):
        if self.time_based:
            iterator = self._t_iterator()
            pbar = self._progressbar_t
        else:
            iterator = range(self.iterations)
            pbar = self._progressbar_iter
        if self.progressbar:
            iterator = pbar(iterator)
        return iterator

    def _t_iterator(self):
        t_ = time.time()
        t = t_ + self.runtime - self._exit_time_buffer
        while t_ < t:
            t_ = time.time()
            yield t_

    def _progressbar_iter(self, rng):
        n = rng[-1]
        t0 = time.time()
        ts = [t0]
        for i in rng:
            done = i/n
            t = time.time()
            ts += [t]
            progress = f'{i+1}/{n+1} iterations in {t - t0:.2f}s'
            _print_progressbar(done, self._len_progressbar, progress, ts)
            yield i
        print()

    def _progressbar_t(self, iterator):
        t0 = time.time()
        i = 0
        max_t = self.runtime - self._exit_time_buffer
        ts = [t0]
        for t in iterator:
            ts += [t]
            i += 1
            dt = t - t0
            done = dt / max_t
            progress = f'{i} iterations in {dt:.2f}/{max_t:.2f}s'
            _print_progressbar(done, self._len_progressbar, progress, ts)
            yield t
        print()


def _print_progressbar(done, total_len, progress, ts):
    done = int(done * total_len)
    not_done = total_len - done
    iter_per_s = 1 / np.diff(ts).mean() if len(ts) > 1 else 0
    progress += f' ({iter_per_s:.2f} iterations/s)'
    pbar = '|' + ('#' * done) + ('-' * not_done) + '|'
    print(f'sample: {pbar} [{progress}]', end='\r')
