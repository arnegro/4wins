from typing import Dict, Callable, Iterator, List

from agents.agent_mcts.searchtree import Stats, SearchTree
from agents.game_utils import BoardPiece, PlayerAction
import time
import numpy as np
from multiprocessing import Process, Queue


class MCTSLogic:

    def __init__(
            self, time_based: bool = False, iterations: int = 1000,
            runtime: float = 5, verbose: bool = False,
            num_processes: int = 1,
            progressbar: bool = False
    ):
        """
        Basically the implementation of a concrete protocol for the monte carlo tree search (MCTS) algorithm
        :param time_based: whether the algorithm should terminate based on a time limit (True),
                           or an iteration limit (False, default)
        :param iterations: number of iterations (only relevant if time_based=False)
        :param runtime: time until termination (only relevant if time_based=True)
        :param verbose: whether to spit out some prints, like the values for the next actions
        :param num_processes: number of processes, if > 1 then the logic will run the mcts several times in parallel
                              and merge the results
        :param progressbar: whether to print a progressbar or not
        """
        self.iterations = iterations
        self.verbose = verbose
        self.time_based = time_based
        self.runtime = runtime  # s
        self.num_processes = num_processes if num_processes >= 1 else 1
        self.progressbar = progressbar
        self._len_progressbar: int = 20
        self._exit_time_buffer: float = 0.01  # s
        self._sleep_time_process = 0.01

    def _init_processes(self, f: Callable) -> None:
        """
        Initialize the processes for parallel search, such that they don't need to be initialized every step,
        which creates a lot of overhead.
        Basically the processes will constantly run throughout the entire game (also during opponent turns)
        and wait for something to be pushed to the shared input queue.
        Then they will perform a function on the pushed input arguments (i.e. a monte carlo tree search) and push the
        result of that computation to the shared output queue
        :param f: function to call on incoming inputs
        :return: nothing
        """
        self._input_queue = Queue()
        self._output_queue = Queue()

        def _run_process(q_in: Queue, q_out: Queue) -> None:
            while True:
                if not q_in.empty():
                    kwargs = q_in.get()
                    res = f(**kwargs)
                    q_out.put(res)
                time.sleep(self._sleep_time_process)

        self._processes = [Process(target=_run_process, args=(self._input_queue, self._output_queue))
                           for _ in range(self.num_processes)]
        for p in self._processes:
            p.start()

    def terminate(self) -> None:
        """
        Terminate possibly started processes
        """
        if hasattr(self, '_processes'):
            for p in self._processes:
                p.kill()
            delattr(self, '_processes')

    def select(self, stats: Stats, parent_stats: Stats, player: BoardPiece) -> float:
        """
        Function to select the successor in a traversal down selection process of a mcts
        :param stats: statistics of current node
        :param parent_stats: statistics of parent node
        :param player: player who last made a move
        :return: a value, over which will be maximized in the selection procedure
        """
        pass

    def update(self, stats: Stats, winner: BoardPiece) -> Stats:
        """
        Update a node's statistics according to the winner of a simulated game down the tree
        :param stats: initial statistics of the node
        :param winner: player who won the simulation
        :return: updated statistics of the node
        """
        pass

    def _action_choice_metric(
            self, action_stats: Dict[PlayerAction, Stats], player: BoardPiece
    ) -> Dict[PlayerAction, float]:
        """
        Choose an action depending on this metric, which will be maximized for that cause.
        :param action_stats: statistics of children of the current game state, indexed by the action that leads to them
        :param player: player who would next make a move
        :return: values for all the given children, i.e. the actions that lead to them
        """
        pass

    def merge_stats(self, stats1: Stats, stats2: Stats) -> Stats:
        """
        Take the stats of two children and merge them into one.
        :param stats1: stats of child 1
        :param stats2: stats of child 2
        :return: combined stats
        """
        pass

    def choose_action(self, action_stats: Dict[PlayerAction, Stats], player: BoardPiece) -> PlayerAction:
        """
        The actual action choice, which will take the action that has maximal _action_choice_metric
        :param action_stats: statistics of children of the current game state, indexed by the action that leads to them
        :param player: player who would next make a move
        :return: the best next move according to the utilised metric
        """
        action_values = self._action_choice_metric(action_stats, player)
        if self.verbose:
            print(self._show_action_vals(action_values))
        return max(action_values, key=action_values.get)

    @staticmethod
    def _show_action_vals(action_values: Dict[PlayerAction, float]) -> str:
        return 'move estimate: ' + '; '.join(f'{a}->{v:.4f}' for a, v in sorted(action_values.items()))

    def run(self, mcts: Callable[[SearchTree,
                                  Callable[[Stats, Stats, BoardPiece], float],
                                  Callable[[Stats, BoardPiece], Stats]],
                                 None],
            search_tree: SearchTree) -> SearchTree:
        """
        Run an MCTS single iteration function several times, depending on the implemented operational logic
        :param mcts: the mcts single iteration function
        :param search_tree: the current search tree
        :return: the updated search tree, after the iterations are done
        """
        if self.num_processes <= 1:

            for _ in self.get_iter():
                mcts(search_tree, self.select, self.update)
            return search_tree

        else:  # multi process

            if not hasattr(self, '_processes'):
                def _mcts(search_tree: SearchTree, supress_pbar: bool) -> SearchTree:
                    for _ in self.get_iter(supress_pbar=supress_pbar):
                        mcts(search_tree, self.select, self.update)
                    return search_tree
                self._init_processes(_mcts)

            supress_pbar = False
            for i in range(self.num_processes):
                self._input_queue.put(dict(search_tree=search_tree, supress_pbar=supress_pbar))
                supress_pbar = True
            search_trees = []
            for _ in range(self.num_processes):
                search_trees += [self._output_queue.get()]

            merged_tree = search_trees[0]
            for tree in search_trees[1:]:
                merged_tree = SearchTree.merge_trees(merged_tree, tree, self.merge_stats)

            return merged_tree

    def get_iter(self, supress_pbar=None) -> Iterator:
        """
        Get an iterable over which to loop for the individual mcts iterations.
        Will be a range if the number of iterations is specified (time_based=False),
        or a generator if the run time is specified (time_based=True).
        Also possibly the iterator will be decorated by a progressbar.
        :param supress_pbar: can be set to True, to forcefully not generate a progressbar, even though
                             self.progressbar=True. Useful for multi-processing, as otherwise the prints will collide.
        :return: the iterable
        """
        if self.time_based:
            iterator = self._t_iterator()
            pbar = self._progressbar_t
        else:
            iterator = range(self.iterations // self.num_processes)
            pbar = self._progressbar_iter
        if (supress_pbar is None and self.progressbar) or (type(supress_pbar) is bool and not supress_pbar):
            iterator = pbar(iterator)
        return iterator

    def _t_iterator(self) -> Iterator[float]:
        """
        :return: a generator, that yields the current time as long as a time limit is not passed, then exits
        """
        t_ = time.time()
        t = t_ + self.runtime - self._exit_time_buffer
        while t_ < t:
            t_ = time.time()
            yield t_

    def _progressbar_iter(self, rng: range) -> Iterator[int]:
        """
        Decorate a range by a progressbar, that will be based on iteration numbers.
        :param rng: the range to decorate
        :return: an iterator that yields equivalently to the given range, but prints the progress each step
        """
        n = rng[-1] * self.num_processes
        t0 = time.time()
        ts = [t0]
        for i in rng:
            i *= self.num_processes
            done = i / n
            t = time.time()
            ts += [t]
            progress = f'{i + 1}/{n + 1} iterations in {t - t0:.2f}s'
            self._print_progressbar(done, progress, ts)
            yield i
        print()

    def _progressbar_t(self, iterator: Iterator[float]) -> Iterator[float]:
        """
        Decorate a generator by a progressbar, that will be based on runtime.
        :param iterator: the generator to decorate, yields time stamps
        :return: an iterator that yields equivalently to the given generator, but prints the progress each step
        """
        t0 = time.time()
        i = 0
        max_t = self.runtime - self._exit_time_buffer
        ts = [t0]
        for t in iterator:
            ts += [t]
            i += self.num_processes
            dt = t - t0
            done = dt / max_t
            progress = f'{i} iterations in {dt:.2f}/{max_t:.2f}s'
            self._print_progressbar(done, progress, ts)
            yield t
        print()

    def _print_progressbar(self, done: float, progress: str, ts: List[float]) -> None:
        """
        Print any progressbar
        :param done: percentage done
        :param progress: progress information as string
        :param ts: time stamps of previous iterations
        :return: nothing, only prints
        """
        done = int(done * self._len_progressbar)
        not_done = self._len_progressbar - done
        iter_per_s = self.num_processes / np.diff(ts).mean() if len(ts) > 2 else 0
        progress += f' ({iter_per_s:.2f} iterations/s)'
        pbar = '|' + ('#' * done) + ('-' * not_done) + '|'
        print(f'sample: {pbar} [{progress}]', end='\r')
