import numpy as np
from agents.game_utils import SavedState, BoardPiece, PlayerAction, PLAYER1, PLAYER2
from typing import Optional, Dict


_OTHER_PLAYER = {PLAYER1: PLAYER2, PLAYER2: PLAYER1}
Stats = Dict


class SearchTree(SavedState):

    def __init__(self, root_board: np.ndarray, player: BoardPiece) -> None:
        """
        A search tree to be utilised by a monte carlo tree search (MCTS) algorithm.
        Each node will correspond to a game state and the children (indexed by actions) are possible successors of
        that node's game state (initially empty).
        Also, each node stores statistics (for example offering the option to store information about likelihoods of
        winning or losing, also initially empty), and is associated the player that would take the next move.
        :param root_board: game state at the root of this tree: current game state, if this node is the root
        :param player: player of current turn
        """
        self.stats: Stats = {}
        self.children: Dict[PlayerAction, SearchTree] = {}
        self.parent: Optional[SearchTree] = None
        self.player: BoardPiece = player
        self.root_board: np.ndarray = root_board

    def make_root(self, board: np.ndarray) -> None:
        """
        Make this node the root of the tree.
        :param board: new root board, so game state at this node
        :return: noting, in place
        """
        self.set_root_board(board)
        self.parent = None

    def expand_child(self, action: PlayerAction):  # -> SearchTree:
        """
        Expand an action from the current node.
        If the corresponding child is already expanded, do nothing and return that.
        :param action: the action
        :return: the expanded child node
        """
        if action in self.children.keys():  # child already expanded, do nothing
            return self.children[action]
        else:
            new_child = SearchTree(self.root_board, _OTHER_PLAYER[self.player])
            new_child.parent = self
            self.children[action] = new_child
            return new_child

    def set_root_board(self, board: np.ndarray) -> None:
        """
        Set root board of entire tree (root_board is a shared state between all members of the tree)
        :param board: new root board
        :return: nothing, in place
        """
        self.root_board[:, :] = board

    # MERGE TREES ######################################################################################################
    @staticmethod
    def merge_trees(search_tree_1, search_tree_2, stats_merge_function):  # -> SearchTree
        """
        Merge two search trees into one.
        :param search_tree_1: first search tree
        :param search_tree_2: second search tree
        :param stats_merge_function: function to combine the statistics dictionaries of two nodes that are shared
                                     between both trees into one
        :return: merged search tree
        """
        if (search_tree_1.root_board != search_tree_2.root_board).any():
            raise ValueError('SearchTrees can not be merged, their boards mismatch')
        if search_tree_1.player != search_tree_2.player:
            raise ValueError('SearchTrees can not be merged, their players mismatch')
        merged_tree = SearchTree(search_tree_1.root_board.copy(), search_tree_1.player)
        merged_tree.stats = stats_merge_function(search_tree_1.stats, search_tree_2.stats)
        common_children = set(search_tree_1.children) & set(search_tree_2.children)  # set intersection
        for move, child in list(search_tree_1.children.items()) + list(search_tree_2.children.items()):
            if move in common_children:
                continue
            else:
                new_child = merged_tree.expand_child(move)
                new_child.stats = {**child.stats}  # copy stats
        for move in common_children:
            child = SearchTree.merge_trees(search_tree_1.children[move],
                                           search_tree_2.children[move],
                                           stats_merge_function)
            child.parent = merged_tree
            merged_tree.children[move] = child
        return merged_tree

    # TO STRING ########################################################################################################
    def __str__(self) -> str:
        children = [action for action in self.children.keys()]
        fields = f'stats: {self.stats}; player: {self.player}; children: {children}'
        return f'SearchTree<{fields}>'

    def __repr__(self) -> str:
        return self.__str__()

    def show_recursive(self, _indent=0) -> str:
        """
        Recursively print the tree.
        :param _indent: current indent -- do not use, only for recursive call, would be indent of entire str block
        :return: recursive string representation of tree
        """
        this = str(self)
        children = '\n'.join(' ' * (_indent + 2) + f'{a} -> {node.show_recursive(_indent + 2)}'
                             for a, node in self.children.items())
        return ' ' * _indent + f'{this}\n{children}'
