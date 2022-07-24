import time

from agents.game_utils import initialize_game_state, PLAYER1, PLAYER2, apply_player_action
from agents.agent_minimax import generate_move, base_heuristic
import numpy as np
import timeit


def get_boards(n):
    other = {PLAYER1: PLAYER2, PLAYER2: PLAYER1}
    boards = []
    t0 = time.time()
    for _ in range(n):
        board = initialize_game_state()
        a, b = board.shape
        n_moves = np.random.randint(a*b)
        player = PLAYER1
        for _ in range(n_moves):
            player = other[player]
            mv, _ = generate_move(board, player, None, depth=2, heuristic=base_heuristic, max_time=None)
            board = apply_player_action(board, mv, player)

        boards += [board]
    print(f'generated {n} boards in {time.time()-t0:.2f}s')
    return boards


def time_simulate():
    from agents.agent_mcts.mcts import _simulate
    number = 10 ** 4
    ts = []
    for board in get_boards(50):
        res = timeit.timeit("_simulate(board, player)",
                            setup="_simulate(board, player)",
                            number=number,
                            globals=dict(_simulate=_simulate,
                                         board=board,
                                         player=PLAYER1))
        ts += [res/number]
    print(f'_simulate {np.mean(ts) * 1e3 : .3f}±{np.std(ts) * 1e3 : .3f} ms per call')


time_simulate()


def time_x_in_a_row():
    from agents.agent_minimax.heuristics import _x_in_a_row
    number = 10**4
    xs = range(2, 5)
    r = {x: [] for x in xs}
    for board in get_boards(50):
        for x in xs:
            res = timeit.timeit("_x_in_a_row(board, player, x)",
                                setup="_x_in_a_row(board, player, x)",
                                number=number,
                                globals=dict(_x_in_a_row=_x_in_a_row,
                                             board=board,
                                             player=PLAYER1,
                                             x=x))
            r[x] += [res/number]

    for x, rs in r.items():
        print(f'_{x}_in_a_row: {np.mean(rs) * 1e6 : .1f}±{np.std(rs) * 1e6 : .1f} us per call')


# time_x_in_a_row()


def time_alpha_beta():
    from agents.agent_minimax.minimax import alpha_beta
    from agents.agent_minimax import base_heuristic
    number = 10
    depths = range(2, 8)
    r = {d: [] for d in depths}
    for board in get_boards(100):
        for depth in depths:
            res = timeit.timeit("alpha_beta(board, depth, -np.inf, np.inf, player, heuristic)",
                                setup="alpha_beta(board, depth, -np.inf, np.inf, player, heuristic)",
                                number=number,
                                globals=dict(alpha_beta=alpha_beta,
                                             board=board,
                                             depth=depth,
                                             player=PLAYER1,
                                             heuristic=base_heuristic,
                                             np=np))
            r[depth] += [res/number]

    for depth, rs in r.items():
        q1, q3 = np.quantile(rs, [0.25, 0.75]) * 1e3
        print(f'alpha_beta; {depth=} {np.mean(rs) * 1e3 : .1f} '
              f'(quartiles: {q1 : .1f} & {q3 : .1f})'
              ' ms per call')


# time_alpha_beta()
