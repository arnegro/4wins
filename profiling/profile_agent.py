import cProfile
from agents.agent_minimax import generate_move, base_heuristic
from main import human_vs_agent
import pstats
from agents.agent_minimax.heuristics import _x_in_a_row
from agents.game_utils import initialize_game_state, PLAYER1, PLAYER2, apply_player_action
from agents.agent_random import generate_move
import numpy as np
import timeit

profile = False
if profile:
    cProfile.run(
        "human_vs_agent(generate_move, generate_move, args_1=(5, base_heuristic), args_2=(5, base_heuristic))", "mmab"
    )

#p = pstats.Stats("mmab")
#p.sort_stats("tottime").print_stats(50)

other = {PLAYER1: PLAYER2, PLAYER2: PLAYER1}

number = 10**4

r = {x: [] for x in range(2, 5)}
for _ in range(20):
    board = initialize_game_state()
    a, b = board.shape
    n_moves = np.random.randint(a*b)
    player = PLAYER1
    for _ in range(n_moves):
        player = other[player]
        mv, _ = generate_move(board, player, None)
        board = apply_player_action(board, mv, player)

    for x in range(2, 5):
        res = timeit.timeit("_x_in_a_row(board, player, x)",
                            setup="_x_in_a_row(board, player, x)",
                            number=number,
                            globals=dict(_x_in_a_row=_x_in_a_row,
                                         board=board,
                                         player=player,
                                         x=x))
        r[x] += [res/number]

for x, rs in r.items():
    print(f'_{x}_in_a_row: {np.mean(rs) * 1e6 : .1f}±{np.std(rs) * 1e6 : .1f} us per call')
