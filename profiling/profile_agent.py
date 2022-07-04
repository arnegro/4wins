import cProfile
from agents.agent_mcts import generate_move
from main import human_vs_agent
import pstats

profile = True
if profile:
    cProfile.run(
        "human_vs_agent(generate_move, generate_move)",# "
                    #"args_1=(5, base_heuristic, None), "
                    #"args_2=(5, base_heuristic, None))",
        "mmab_mcts"
    )

p = pstats.Stats("mmab_mcts")
p.sort_stats("tottime").print_stats(50)
