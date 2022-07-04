import numpy as np
from agents.game_utils import SavedState, BoardPiece, PlayerAction, PLAYER1, PLAYER2
from typing import Optional, Dict


_OTHER_PLAYER = {PLAYER1: PLAYER2, PLAYER2: PLAYER1}
Stats = Dict


class SearchTree(SavedState):

    def __init__(self, root_board: np.ndarray, player: BoardPiece) -> None:

        self.stats: Stats = {}
        self.children: Dict[PlayerAction, SearchTree] = {}
        self.parent: Optional[SearchTree] = None
        self.player: BoardPiece = player
        self.root_board: np.ndarray = root_board

    def make_root(self) -> None:
        self.parent = None

    def expand_child(self, action: PlayerAction):  # -> SearchTree:
        if action in self.children.keys():  # child already expanded, do nothing
            return self.children[action]
        else:
            new_child = SearchTree(self.root_board, _OTHER_PLAYER[self.player])
            new_child.parent = self
            self.children[action] = new_child
            return new_child

    def set_root_board(self, board: np.ndarray) -> None:
        self.root_board[:, :] = board

    # TO STRING ########################################################################################################
    def __str__(self) -> str:
        children = [action for action in self.children.keys()]
        fields = f'stats: {self.stats}; player: {self.player}; children: {children}'
        return f'SearchTree<{fields}>'

    def __repr__(self) -> str:
        return self.__str__()

    def show_recursive(self, indent=0) -> str:
        this = str(self)
        children = '\n'.join(' ' * (indent + 2) + f'{a} -> {node.show_recursive(indent + 2)}'
                             for a, node in self.children.items())
        return ' ' * indent + f'{this}\n{children}'
