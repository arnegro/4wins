import numpy as np
from agents.game_utils \
    import BoardPiece, PlayerAction, NO_PLAYER, check_end_state, GameState, apply_player_action, PLAYER1, PLAYER2
from typing import Optional, Tuple, Callable, Dict, List
from agents.agent_mcts.searchtree import SearchTree, Stats, _OTHER_PLAYER
from agents.agent_mcts.mcts_logics import MCTSLogic, UCB1
import numba


# TYPES ################################################################################################################
SelectionFunction = Callable[[Stats, Stats, BoardPiece],  # node stats, parent stats, player ->
                             float]

# agnostic to player -- stats will be not dependent on player, has to be handled in action selection
UpdateFunction = Callable[[Stats, BoardPiece],  # node stats, winner ->
                          Stats]

ActionChoiceFunction = Callable[[Dict[PlayerAction, Stats], BoardPiece],  # stats of child nodes (actions), player ->
                                PlayerAction]


# GENERATE MOVE ########################################################################################################
def generate_move_mcts(
        board: np.ndarray, player: BoardPiece, saved_state: Optional[SearchTree],
        logic: MCTSLogic = UCB1()
) -> Tuple[PlayerAction, Optional[SearchTree]]:
    """
    Generate a move by utilizing monte carlo tree search (MCTS).
    :param board: current game state
    :param player: current player
    :param saved_state: search tree from previous iterations (optional)
    :param logic: the logic to follow in the selection, iteration, etc...
    :return: the generated move, as well as the updated search tree
    """
    if saved_state is None:
        search_tree = SearchTree(root_board=board, player=player)
    else:
        search_tree = _move_root(saved_state, board)
        # saved_state will be 1 or 2 moves back, update tree according to the moves that have been actually made

    search_tree = logic.run(_monte_carlo_tree_search, search_tree)

    move_stats = {move: child.stats for move, child in search_tree.children.items()}
    next_action = logic.choose_action(move_stats, player)

    return next_action, search_tree


def _monte_carlo_tree_search(
        search_tree: SearchTree, selection_function: SelectionFunction, update_function: UpdateFunction
) -> None:
    """
    One MCTS iteration.
    Consists of:
        - successively selecting nodes in the search tree, until a not-yet-tried move is possible
        - expanding that child
        - simulating a game from that child
        - propagating up the result of the simulation through the tree
          and updating the statistics of the nodes on the path
    :param search_tree: the current search tree
    :param selection_function: function to select a node in the traverse down
    :param update_function: function to update the statistics of the nodes depending on the winner on the propagation up
    :return: nothing, search tree will be altered in place
    """
    terminal_node, terminal_board = _select_and_expand(search_tree, selection_function)
    winner = _simulate(terminal_board, terminal_node.player)
    _propagate_up(terminal_node, winner, update_function)


# MOVE ROOT ############################################################################################################
def _move_root(
        search_tree: SearchTree, board: np.ndarray
) -> SearchTree:
    """
    Update the search tree according to moves that have happened in the real world.
    i.e. if a move sequence [a, b, c] has happened since the root of the search tree, move the root down the children
    [a, b, c] successively
    :param search_tree: possibly outdated search tree
    :param board: current game state
    :return: updated search tree -- in place update of the root_board though!
    """
    found, moves = _find_connecting_action_sequence(search_tree.root_board, board, search_tree.player)
    if not found:
        raise ValueError('incompatible game states: root board of SearchTree can not be a predecessor of given board')
    else:
        for move in moves:
            search_tree = search_tree.expand_child(move)
        search_tree.make_root(board)
        return search_tree


def _find_connecting_action_sequence(
        root_board: np.ndarray, board: np.ndarray, player: BoardPiece
) -> Tuple[bool, List[PlayerAction]]:
    """
    For two boards find that sequence of actions that would lead from the first board to the second
    :param root_board: starting board
    :param board: goal board
    :param player: starting player
    :return: bool, whether the procedure succeeded,
             as well as the sequence of connecting actions (empty, if not succeeded)
    """
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
    Check, if there could potentially be a sequence of actions that leads from board0 to board1.
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
    """
    Traverse down the search tree by the selection function given, and as soon as a non-expanded possible move is hit,
    expand and return it
    :param search_tree: the search tree
    :param selection_function: the function by which the successor child in the traversal down is selected
    :return: newly expanded child, as well as the game state there
    """
    return _selection_helper(search_tree, search_tree.root_board, selection_function)


def _selection_helper(
        node: SearchTree, board: np.ndarray, selection_function: SelectionFunction
) -> Tuple[SearchTree, np.ndarray]:
    """
    The actual functionality of _select_and_expand
    :param node: current node
    :param board: current game state
    :param selection_function: function by which the successor child in the traversal down is selected
    :return: newly expanded child, as well as the game state there
    """
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
USE_NUMBA = True

if USE_NUMBA:
    @numba.njit()
    def _simulate(board: np.ndarray, player: BoardPiece) -> BoardPiece:
        """
        Simulate a random game
        :param board: initial board
        :param player: player who last played
        :return: winner
        """
        # a lot of code copy to make numba work...
        board = board.copy()
        rows, cols = board.shape
        fill_level = np.sum(board != NO_PLAYER, axis=0)
        __OTHER_PLAYER = {PLAYER1: PLAYER2, PLAYER2: PLAYER1}
        player = __OTHER_PLAYER[player]
        while True:
            player_board = board == player
            x = 4
            for i in range(rows):  # horizontal
                for j in range(cols - x + 1):
                    if np.all(player_board[i, j:j + x]):
                        return player

            for i in range(rows - x + 1):  # vertical
                for j in range(cols):
                    if np.all(player_board[i:i + x, j]):
                        return player

            for i in range(rows - x + 1):  # diagonals
                for j in range(cols - x + 1):
                    block = player_board[i:i + x, j:j + x]
                    if np.all(np.diag(block)) or np.all(np.diag(block[::-1, :])):
                        return player

            player = __OTHER_PLAYER[player]

            legal_moves = np.arange(cols)[board[-1] == NO_PLAYER]
            if len(legal_moves) == 0:
                return NO_PLAYER
            move = np.random.choice(legal_moves)
            board[fill_level[move], move] = player
            fill_level[move] += 1

else:

    def _simulate(board: np.ndarray, player: BoardPiece) -> BoardPiece:
        """
        Simulate a random game
        :param board: initial board
        :param player: player who last played
        :return: winner
        """
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
    """
    Propagate the result of a simulated game up through the search tree and update the parameters accordingly.
    :param terminal_node: node, from which the simulation originated
    :param winner: player who won
    :param update_function: function to update the stats of the nodes according to the winner
    :return: nothing, will change search tree in place
    """
    node = terminal_node
    while node is not None:
        node.stats = update_function(node.stats, winner)
        node = node.parent


# HELPERS ##############################################################################################################
def _get_legal_moves(board: np.ndarray) -> np.ndarray:
    """Takes a board and returns all possible columns in which a move could be applied"""
    mvs = np.argwhere(board[-1] == NO_PLAYER).flatten()
    return mvs
