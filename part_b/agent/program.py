# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Game Playing Agent

from __future__ import annotations

from referee.game import Action, PlayerColor
from referee.game.board import Board

from .config import EvalWeights, SearchConfig
from .search import SearchEngine


class Agent:
    """
    Cascade agent entry point required by the referee.

    The game-playing logic lives in the helper modules so it is easier to test
    and tune independently.
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        """
        Called once at the beginning of game.
        """
        self._color = color    # fixed player color for entire game
        self._board = Board(initial_player=PlayerColor.RED)    # local copy of current game state 
        search_config: SearchConfig | None = referee.get("search_config")
        eval_weights: EvalWeights | None = referee.get("eval_weights")
        self._search = SearchEngine(
            color,
            config=search_config,
            eval_weights=eval_weights,
        )    # search engine that selects actions using adversarial search

    def action(self, **referee: dict) -> Action:
        """
        Called by referee whenever it's our turn to move.
        """
        return self._search.choose_action(self._board, **referee)

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        Called after any player makes a legal move.
        
        We simply apply that move to our internal board so that our local state
        stays synchronized with the referee's true state.
        """
        self._board.apply_action(action)

    @property
    def last_search_stats(self):
        """
        Convenience property for debugging / experimentation
        
        Gives access to stats from the most recent search:
        - total nodes searched
        - quiescence nodes
        - transposition table hits
        - deepest compeleted depth
        """
        return self._search.last_stats
