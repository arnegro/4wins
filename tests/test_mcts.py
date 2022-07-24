from agents.agent_mcts.mcts \
    import _move_root, _find_connecting_action_sequence, \
    _select_and_expand, _get_legal_moves, _simulate, _propagate_up
from agents.agent_mcts.searchtree import SearchTree, _OTHER_PLAYER
from agents.agent_mcts import generate_move
from agents.agent_mcts.mcts_logics import UCB1, Beta
from agents.game_utils import PLAYER1, PLAYER2, initialize_game_state, apply_player_action, PlayerAction
import pytest
import numpy as np


# Test SearchTree ######################################################################################################

def test_expand_child():
    search_tree = SearchTree(root_board=initialize_game_state(), player=PLAYER1)
    move = 4
    assert move not in search_tree.children
    child = search_tree.expand_child(PlayerAction(move))
    assert move in search_tree.children
    assert child.root_board is search_tree.root_board  # same object
    assert child.player == PLAYER2
    assert child.parent is search_tree


def test_set_root_board():
    search_tree = SearchTree(root_board=initialize_game_state(), player=PLAYER1)
    nodes = [search_tree]
    for move in [1, 4, 2, 6]:
        nodes.append(nodes[-1].expand_child(move))
    new_board = initialize_game_state()
    new_board[0, 3] = PLAYER1
    search_tree.set_root_board(new_board)
    for node in nodes:
        assert (node.root_board == new_board).all()


def test_merge_trees():
    search_tree1 = SearchTree(root_board=initialize_game_state(), player=PLAYER1)
    search_tree2 = SearchTree(root_board=initialize_game_state(), player=PLAYER1)
    for search_tree, mvs in [(search_tree1, [1, 4]), (search_tree2, [1, 3])]:
        nodes = [search_tree]
        for move in mvs:
            nodes.append(nodes[-1].expand_child(move))
    merged_tree = SearchTree.merge_trees(search_tree1, search_tree2, lambda stats1, stats2: {})
    expected_tree = SearchTree(root_board=initialize_game_state(), player=PLAYER1)
    node = expected_tree.expand_child(1)
    for mv in [4, 3]:
        node.expand_child(mv)
    assert expected_tree.show_recursive() == merged_tree.show_recursive()




# Test mcts ############################################################################################################

def _make_board():
    board = initialize_game_state(shape=(6, 7))
    p1_pos = [(0, 3), (0, 4), (1, 2), (2, 3), (2, 4), (3, 2), (3, 3)]
    p2_pos = [(0, 1), (0, 2), (1, 1), (1, 3), (1, 4), (2, 2)]
    for pos in p1_pos:
        board[pos] = PLAYER1
    for pos in p2_pos:
        board[pos] = PLAYER2
    return board


def test_move_root_throws_error_on_incompatible_boards():
    board = _make_board()
    player = PLAYER1
    search_tree = SearchTree(root_board=board.copy(), player=player)
    with pytest.raises(ValueError):
        _move_root(search_tree, initialize_game_state())


def test_move_root_dont_move_for_fitting_root():
    board = _make_board()
    search_tree = SearchTree(board.copy(), PLAYER1)
    assert search_tree == _move_root(search_tree, board)
    assert (search_tree.root_board == board).all()


def test_move_root_helper():
    board = _make_board()
    root_board = board.copy()
    player = PLAYER1
    moves = [3, 6]
    for move in moves:
        board = apply_player_action(board, PlayerAction(move), player)
        player = _OTHER_PLAYER[player]
    assert not _find_connecting_action_sequence(board, root_board, player)[0]
    found, mvs = _find_connecting_action_sequence(root_board, board, player)
    assert moves == mvs
    assert found


def test_move_root_moves_root():
    board = _make_board()
    search_tree = SearchTree(root_board=board.copy(), player=PLAYER1)
    moves = [3, 6]
    node = search_tree
    for move in moves:
        move = PlayerAction(move)
        board = apply_player_action(board, move, node.player)
        node = node.expand_child(move)
    new_root = _move_root(search_tree, board)
    assert new_root is node
    assert (new_root.root_board == board).all()


def test_select_and_expand_choose_root_if_not_all_expanded():
    board = initialize_game_state()
    player = PLAYER1
    search_tree = SearchTree(root_board=board.copy(), player=player)
    move_sequence = [1, 1, 4]
    node = search_tree
    for move in move_sequence:
        node = node.expand_child(PlayerAction(move))
    terminal_node, terminal_board = _select_and_expand(search_tree, selection_function=lambda s, ps, p: None)
    assert terminal_node.parent == search_tree
    assert (board != terminal_board).sum() == 1


def _make_search_tree():
    board = initialize_game_state()
    player = PLAYER1
    search_tree = SearchTree(root_board=board.copy(), player=player)
    node = search_tree
    mv = PlayerAction(6)
    for depth in range(3):
        for move in _get_legal_moves(board):
            child = node.expand_child(move)
            child.stats = {'w': move, 's': 10}
        board = apply_player_action(board, mv, node.player)
        node = node.children[mv]
    return search_tree, node, board, mv


def test_select_and_expand_traverse_down_if_all_expanded_then_propagate_up():
    search_tree, node, board, _ = _make_search_tree()
    terminal_node, terminal_board = _select_and_expand(search_tree, selection_function=lambda stats, ps, p: stats['w'])
    assert terminal_node.parent == node
    assert (board != terminal_board).sum() == 1


def test_propagate_up():
    search_tree, node, board, mv = _make_search_tree()
    terminal_node = node.expand_child(PlayerAction(0))
    _propagate_up(terminal_node, PLAYER1, UCB1().update)
    for n in [search_tree, terminal_node]:
        assert n.stats == {'draw': 0, 'w': 1, 's': 1}
    while node.parent is not None:
        assert node.stats['w'] == mv+1
        assert node.stats['s'] == 11
        node = node.parent


def test_simulate_non_biased():
    player = PLAYER1
    ws = []
    for _ in range(1000):
        ws.append(_simulate(initialize_game_state(), player))
        player = _OTHER_PLAYER[player]
    assert np.abs(np.mean(ws) - 1.5) < 0.1


def test_simulate_realises_win():
    board = initialize_game_state()
    board[:] = PLAYER1
    assert _simulate(board, PLAYER2) == PLAYER1


logics = [UCB1(iterations=1000), Beta(iterations=1000)]


def test_takes_win():
    board = initialize_game_state()
    col = 2
    board[:3, col] = PLAYER1
    for logic in logics:
        move, search_tree = generate_move(board, PLAYER1, None, logic)
        assert move == 2


def test_prevents_lose():
    board = initialize_game_state()
    col = 2
    board[:3, col] = PLAYER1
    for logic in logics:
        move, search_tree = generate_move(board, PLAYER2, None, logic)
        assert move == 2


def test_takes_win2():
    board = initialize_game_state()
    board[0, 2:4] = PLAYER1
    board[1, 2:4] = PLAYER2
    for logic in logics:
        move, search_tree = generate_move(board, PLAYER1, None, logic)
        assert move in [1, 4]


def test_prevents_lose2():
    board = initialize_game_state()
    board[0, 2:4] = PLAYER1
    board[1, 2:4] = PLAYER2
    for logic in logics:
        move, search_tree = generate_move(board, PLAYER2, None, logic)
        assert move in [1, 4]


def test_takes_win3():
    board = initialize_game_state()
    board[[0, 0, 0, 1], [3, 4, 5, 3]] = PLAYER1
    board[[0, 2, 3, 4], [2, 3, 3, 3]] = PLAYER2
    for logic in logics:
        move, search_tree = generate_move(board, PLAYER1, None, logic)
        assert move == 6


def test_prevents_lose3():
    board = initialize_game_state()
    board[[0, 0, 0, 1], [3, 4, 5, 3]] = PLAYER1
    board[[0, 2, 3], [2, 3, 3]] = PLAYER2
    for logic in logics:
        move, search_tree = generate_move(board, PLAYER2, None, logic)
        assert move == 6


def test_gen_move_exit_after_5s():
    import time
    runtime = 5
    t0 = time.time()
    generate_move(initialize_game_state(), PLAYER1, None, UCB1(time_based=True, runtime=runtime))
    assert time.time() - t0 < runtime
