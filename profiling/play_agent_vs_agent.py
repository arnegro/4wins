from agents.agent_mcts import generate_move as mcts
from agents.agent_minimax import generate_move as minimax
from agents.agent_minimax import base_heuristic
from agents.agent_mcts import UCB1


def agent_vs_agent(
    generate_move_1,
    generate_move_2,
    num_games=20,
    player_1='1',
    player_2='2',
    args_1: tuple = (),
    args_2: tuple = ()
):
    import time
    from agents.game_utils import PLAYER1, PLAYER2, GameState
    from agents.game_utils import initialize_game_state, apply_player_action, check_end_state

    players = (PLAYER1, PLAYER2)
    winners = []
    for play_first in [1, -1]*num_games:

        saved_state = {PLAYER1: None, PLAYER2: None}
        board = initialize_game_state()
        gen_moves = (generate_move_1, generate_move_2)[::play_first]
        player_names = (player_1, player_2)[::play_first]
        gen_args = (args_1, args_2)[::play_first]

        playing = True
        t0 = time.time()
        while playing:
            for player, player_name, gen_move, args in zip(
                players, player_names, gen_moves, gen_args,
            ):
                action, saved_state[player] = gen_move(
                    board.copy(), player, saved_state[player], *args
                )
                board = apply_player_action(board, action, player)
                end_state = check_end_state(board, player)
                if end_state != GameState.STILL_PLAYING:
                    if end_state == GameState.IS_DRAW:
                        winners += [None]
                    else:
                        winners += [player_name]
                    print(winners[-1], (time.time()-t0)/60)
                    playing = False
                    break
    return winners


play = False
fl_name = 'winners.dat'

if __name__ == "__main__" and play:
    runtime = 5
    logic = UCB1(time_based=True, runtime=runtime, num_processes=3)
    args_2 = (5, base_heuristic, runtime)
    winners = agent_vs_agent(mcts, minimax, num_games=25, args_1=(logic,), args_2=args_2)
    with open(fl_name, 'w') as fl:
        fl.write('\n'.join(str(w) for w in winners))
    logic.terminate()

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'MCTS Agent', 'Minimax Agent'
    wins = {str(i): 0 for i in [1, 2]}
    with open(fl_name) as fl:
        for ln in fl.readlines():
            ln = ln.strip()
            wins[ln] += 1

    sizes = list(wins.values())
    explode = (0, 0.1)  # only "explode" the 2nd slice i.e. minimax

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=False, startangle=0, colors=['tab:blue', 'tab:red'])
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    ax1.set(title=f'Wins amongst {sum(sizes)} games')

    plt.savefig('pie_for_the_winner.pdf')
    plt.show()
