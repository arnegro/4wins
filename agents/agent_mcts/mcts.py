import numpy as np
from agents.game_utils import BoardPiece, PlayerAction, NO_PLAYER, check_end_state, GameState, apply_player_action
from typing import Optional, Tuple, Callable, Dict, List
from agents.agent_mcts.searchtree import SearchTree, Stats, _OTHER_PLAYER
from agents.agent_mcts.mcts_logics import MCTSLogic, UCB1


# TYPES ################################################################################################################
SelectionFunction = Callable[[Stats, Stats, BoardPiece],  # node stats, parent stats  # ->
                             float]

# agnostic to player -- stats will be not dependent on player, has to be handled in action selection
UpdateFunction = Callable[[Stats, BoardPiece],  # ->
                          Stats]

ActionChoiceFunction = Callable[[Dict[PlayerAction, Stats], BoardPiece],  # ->
                                PlayerAction]


# GENERATE MOVE ########################################################################################################
def generate_move_mcts(
        board: np.ndarray, player: BoardPiece, saved_state: Optional[SearchTree],
        logic: MCTSLogic = UCB1()
) -> Tuple[PlayerAction, Optional[SearchTree]]:
    if saved_state is None:
        search_tree = SearchTree(root_board=board, player=player)
    else:
        search_tree = _move_root(saved_state, board)
        # saved_state will be 1 or 2 moves back, update tree according to the moves that have been actually made

    for _ in logic.get_iter():
        _monte_carlo_tree_search(search_tree, logic.select, logic.update)

    move_stats = {move: child.stats for move, child in search_tree.children.items()}
    next_action = logic.choose_action(move_stats, player)

    return next_action, search_tree


def _monte_carlo_tree_search(
        search_tree: SearchTree, selection_function: SelectionFunction, update_function: UpdateFunction
) -> None:
    terminal_node, terminal_board = _select_and_expand(search_tree, selection_function)
    winner = _simulate(terminal_board, terminal_node.player)
    _propagate_up(terminal_node, winner, update_function)


# MOVE ROOT ############################################################################################################
def _move_root(
        search_tree: SearchTree, board: np.ndarray
) -> SearchTree:
    found, moves = _find_connecting_action_sequence(search_tree.root_board, board, search_tree.player)
    if not found:
        raise ValueError('incompatible game states: root board of SearchTree can not be a predecessor of given board')
    else:
        for move in moves:
            search_tree = search_tree.expand_child(move)
        search_tree.set_root_board(board)
        return search_tree


def _find_connecting_action_sequence(
        root_board: np.ndarray, board: np.ndarray, player: BoardPiece
) -> Tuple[bool, List[PlayerAction]]:
    if (board == root_board).all():
        return True, []
    if not _is_possible_predecessor(root_board, board):
        return False, []
    for move in _get_legal_moves(root_board):
        child = apply_player_action(root_board, move, player)
        found, moves = _find_connecting_action_sequence(child, board, _OTHER_PLAYER[player])
        if found:
            return True, [move] + moves
    return False, []


def _is_possible_predecessor(board0: np.ndarray, board1: np.ndarray) -> bool:
    """
    Where board1 (future board) is empty, board0 (initial board) has to be empty as well.
    And where board0 is non_empty, board1 needs to have the same stones played there.
    """
    board1_empty = board1 == NO_PLAYER
    board0_empty_match = (board1[board1_empty] == board0[board1_empty]).all()
    board0_non_empty = board0 != NO_PLAYER
    board1_non_empty_match = (board0[board0_non_empty] == board1[board0_non_empty]).all()
    return board0_empty_match and board1_non_empty_match


# SELECT AND EXPAND ####################################################################################################
def _select_and_expand(
        search_tree: SearchTree, selection_function: SelectionFunction
) -> Tuple[SearchTree, np.ndarray]:
    return _selection_helper(search_tree, search_tree.root_board, selection_function)


def _selection_helper(
        node: SearchTree, board: np.ndarray, selection_function: SelectionFunction
) -> Tuple[SearchTree, np.ndarray]:
    if getattr(node, 'is_end_state', False):  # do not expand any child if the game is over in this node already
        return node, board
    moves = _get_legal_moves(board)
    if len(moves) > len(node.children):
        move = np.random.choice([move for move in moves if move not in node.children])
        new_child = node.expand_child(move)
        new_board = apply_player_action(board, move, node.player)
        setattr(new_child, 'is_end_state', check_end_state(new_board, node.player) != GameState.STILL_PLAYING)
        return new_child, new_board
    else:
        move = max(node.children, key=lambda mv: selection_function(node.children[mv].stats, node.stats, node.player))
        board = apply_player_action(board, move, node.player)
        return _selection_helper(node.children[move], board, selection_function)


# SIMULATE #############################################################################################################
def _simulate(board: np.ndarray, player: BoardPiece) -> BoardPiece:
    while check_end_state(board, _OTHER_PLAYER[player]) == GameState.STILL_PLAYING:
        move = np.random.choice(_get_legal_moves(board))
        board = apply_player_action(board, move, player)
        player = _OTHER_PLAYER[player]
    player = _OTHER_PLAYER[player]
    end_state = check_end_state(board, player)
    if end_state == GameState.IS_WIN:
        return player
    else:
        return NO_PLAYER


# PROPAGATE UP #########################################################################################################
# in place
def _propagate_up(terminal_node: SearchTree, winner: BoardPiece, update_function: UpdateFunction) -> None:
    node = terminal_node
    while node is not None:
        node.stats = update_function(node.stats, winner)
        node = node.parent


# HELPERS ##############################################################################################################
def _get_legal_moves(board: np.ndarray) -> np.ndarray:
    """Takes a board and returns all possible columns in which a move could be applied"""
    mvs = np.argwhere(board[-1] == NO_PLAYER).flatten()
    return mvs
