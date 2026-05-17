from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from referee.game import Action, PlayerColor
from referee.game.board import Board, GamePhase
from referee.game.constants import MAX_TURNS, PLACEMENT_TURNS

from .config import (
    DEFAULT_EVAL_WEIGHTS,
    DEFAULT_SEARCH_CONFIG,
    EvalWeights,
    SearchConfig,
)
from .evaluation import Evaluator
from .game_utils import (
    action_priority,
    board_key,
    generate_legal_actions,
    generate_loud_actions,
    is_tactical_action,
)

# Transposition table bound types
EXACT = 0    # exact minimax value known
LOWER_BOUND = 1    # lowerbound of true value
UPPER_BOUND = 2    # upperbound of true value


class SearchTimeout(RuntimeError):
    """
    Raised when the current search exceeds its time budget
    """


@dataclass(slots=True)
class TTEntry:
    """
    One transposition table entry
    """
    depth: int    # search depth at which this result was completed
    score: int    # stored evaluation / search result
    bound: int    # whether the score is exact/upper/lower bound
    best_action: Action | None    # best move fonud from this position at the stored depth 


@dataclass(slots=True)
class SearchStats:
    """
    Per move diagnostic stats
    """
    nodes: int = 0    # number of full negamax nodes searched
    tt_hits: int = 0      # number of times a transposition table is used
    cutoffs: int = 0      # number of alpha-beta cutoffs
    qnodes: int = 0       # number of quiescence saerch nodes
    deepest_depth: int = 0    # deepest fully attempted iterative-deepening depth


class SearchEngine:
    """
    Main adversarial-search component
    
    Search strategy:
    - iterative deepening
    - negamax 
    - alpha-beta pruning
    - transposition table
    - history heuristic for move ordering
    - quiescence search
    - tactical search extensions
    - adaptive time allocation
    """
    
    def __init__(
        self,
        color: PlayerColor,
        *,
        config: SearchConfig | None = None,
        eval_weights: EvalWeights | None = None,
    ):
        """
        Initializes the side (color) the engine is searching for.
        """
        self._color = color
        self._config = config or DEFAULT_SEARCH_CONFIG
        self._evaluator = Evaluator(eval_weights or DEFAULT_EVAL_WEIGHTS)    # heuristic board evaluator
        self._transposition_table: dict[tuple, TTEntry] = {}    # caches previously searched positions
        self._history: dict[Action, int] = {}    # move-ordering history heuristic based on past cutoffs 
        self._last_completed_action: Action | None = None    # best root move from previous completed search
        self.last_stats = SearchStats()    # search stats from latest move

    def choose_action(self, board: Board, **referee: dict) -> Action:
        """
        Searchs for the best move from the current position.

        Iterative deepening ensures always have a fallback move if a deeper
        search iteration runs out of time.
        
        Workflow:
        1. Generate legal actions
        2. Reutrn if theres only one move
        3. Allocate a time budget for this turn
        4. Run iterative deepening
        5. If time expires during a deeper iteration, fall back to last compelted best move
        """
        legal_actions = generate_legal_actions(board)
        
        if not legal_actions:
            raise RuntimeError("No legal actions available.")
        if len(legal_actions) == 1:
            return legal_actions[0]

        # Convert remaining total time into a budget for this move
        time_budget = self._allocate_time_budget(board, referee)
        deadline = perf_counter() + time_budget
        
        # Reset stats for this move
        self.last_stats = SearchStats()

        # Use transposition table best move if available, else reuse the previous completed root move
        key = board_key(board)
        preferred = self._preferred_action(key)
        
        # Safe fall back move before deep saerch begins
        best_action = self._ordered_actions(board, legal_actions, preferred, ply=0)[0]
        depth = 1

        try:
            while True:
                self.last_stats.deepest_depth = depth
                score, candidate = self._search_root(board, depth, deadline)
                if candidate is not None:
                    best_action = candidate
                    self._last_completed_action = candidate

                # Placement has a large branching factor, so cap depth
                if board.phase == GamePhase.PLACEMENT and depth >= self._config.max_placement_depth:
                    break

                # In unresolved late game positions, someimtes increase depth faster
                if (
                    board.phase == GamePhase.PLAY
                    and board.play_phase_turn_count > self._config.late_game_turn_threshold
                    and abs(score) < self._config.win_score // 2
                ):
                    depth += self._config.late_game_depth_increment
                else:
                    depth += 1

        except SearchTimeout:
            # Timeout is expected behavior in iterative deepening
            # Simply return the last completed best move
            pass

        return best_action

    def _search_root(self, board: Board, depth: int, deadline: float) -> tuple[int, Action | None]:
        """
        Search one full depth iteration from the root position
        """

        alpha = -self._config.win_score
        beta = self._config.win_score
        best_score = -self._config.win_score
        best_action = None

        key = board_key(board)
        actions = self._ordered_actions(
            board,
            generate_legal_actions(board),
            self._preferred_action(key),
            ply=0,
        )

        for action in actions:
            self._check_timeout(deadline)
            
            # Tactical actions can be extended near frontier
            extension = self._extension(board, action, depth)
            
            board.apply_action(action)
            try:
                # Tactical moves get a small extension so that captures and
                # dangerous cascades are less likely to be cut off too early.
                next_depth = depth - 1 + extension
                score = -self._negamax(board, next_depth, -beta, -alpha, deadline, ply=1)
            finally:
                board.undo_action()

            if score > best_score or best_action is None:
                best_score = score
                best_action = action

            if score > alpha:
                alpha = score

        # If finished this root search, store it as exact
        if best_action is not None:
            self._transposition_table[key] = TTEntry(depth, best_score, EXACT, best_action)

        return best_score, best_action

    def _negamax(
        self,
        board: Board,
        depth: int,
        alpha: int,
        beta: int,
        deadline: float,
        ply: int,
    ) -> int:
        """
        Main recursive negamax search with alpha-beta pruning
        
        - Search from current side-to-move's perspective
        - After making a move, recurse and negate the returned score
        """
        self.last_stats.nodes += 1
        self._check_timeout(deadline)

        # Terminal states always override normal evaluation
        if board.game_over:
            return self._terminal_value(board)

        # When depth is exhausted, switch to quiescence instead of static eval
        if depth <= 0:
            return self._quiescence(
                board,
                alpha,
                beta,
                deadline,
                ply,
                depth_limit=self._config.quiescence_depth,
            )

        key = board_key(board)
        alpha_orig = alpha
        
        # Probe trasnposition table (TT)
        entry = self._transposition_table.get(key)
        if entry is not None and entry.depth >= depth:
            self.last_stats.tt_hits += 1
            
            # Exact TT value can be resused immediately
            if entry.bound == EXACT:
                return entry.score
            
            # Bounds can tighten the alpha-beta window
            if entry.bound == LOWER_BOUND:
                alpha = max(alpha, entry.score)
            elif entry.bound == UPPER_BOUND:
                beta = min(beta, entry.score)
                
            # If window collapses, reuse TT score 
            if alpha >= beta:
                return entry.score

        # Generate and order actions
        actions = self._ordered_actions(
            board,
            generate_legal_actions(board),
            entry.best_action if entry is not None else None,
            ply=ply,
        )

        # Fall back if no actions exist for some nonterminal reason
        if not actions:
            return self._evaluate_current_player(board)

        best_score = -self._config.win_score
        best_action = None

        for action in actions:
            extension = self._extension(board, action, depth)
            board.apply_action(action)
            try:
                next_depth = depth - 1 + extension
                score = -self._negamax(board, next_depth, -beta, -alpha, deadline, ply + 1)
            finally:
                board.undo_action()

            if score > best_score:
                best_score = score
                best_action = action

            if score > alpha:
                alpha = score

            # Alpha-beta cutoff
            if alpha >= beta:
                self.last_stats.cutoffs += 1
                # Reward cutoff-causing moves in the history heuristic
                self._history[action] = self._history.get(action, 0) + depth * depth
                break

        # Store whether the returned score is exact or just a bound 
        bound = EXACT
        if best_score <= alpha_orig:
            bound = UPPER_BOUND
        elif best_score >= beta:
            bound = LOWER_BOUND

        self._transposition_table[key] = TTEntry(depth, best_score, bound, best_action)
        return best_score

    def _quiescence(
        self,
        board: Board,
        alpha: int,
        beta: int,
        deadline: float,
        ply: int,
        depth_limit: int,
    ) -> int:
        """
        Quiescence search
        
        Instead of evaluating immediately at depth 0, continue searching only tactical/noisy moves.
        This reduces horizon effects without exploding the full branching factor of the ordinary search.
        """
        self.last_stats.qnodes += 1
        self._check_timeout(deadline)

        # Static score if stops here
        stand_pat = self._evaluate_current_player(board)
        
        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat

        # Stop quiescence if depth limit exhausted or is in placement phase
        if depth_limit <= 0 or board.phase == GamePhase.PLACEMENT:
            return stand_pat

        # Only tactical moves are search in quiescence
        actions = self._ordered_actions(
            board,
            generate_loud_actions(board),
            preferred=None,
            ply=ply,
        )

        for action in actions:
            board.apply_action(action)
            try:
                score = -self._quiescence(board, -beta, -alpha, deadline, ply + 1, depth_limit - 1)
            finally:
                board.undo_action()

            if score >= beta:
                return score
            if score > alpha:
                alpha = score

        return alpha

    def _evaluate_current_player(self, board: Board) -> int:
        """
        Evaluator always scores from our fixed perspective (color)
        
        Negamax wants to score from the current side-to-move's perspective, so:
        - if it's our turn, use score as is
        - if it's opponent's turn, negate it
        """
        score = self._evaluator.evaluate(board, self._color)
        return score if board.turn_color == self._color else -score

    def _terminal_value(self, board: Board) -> int:
        """
        Convert a finished game into a very large score
        
        Draw -> 0
        Win for side to move -> +win_score
        Lose for side to omove -> -win_score
        """
        winner = board.winner_color
        if winner is None:
            return 0
        return self._config.win_score if winner == board.turn_color else -self._config.win_score

    def _ordered_actions(
        self,
        board: Board,
        actions: list[Action],
        preferred: Action | None,
        ply: int,
    ) -> list[Action]:
        """
        Order moves using:
        1. static domain-specific action_priority()
        2. history heuristic bonus
        3. preferred move from TT / previous root
        """
        ordered = sorted(
            actions,
            key=lambda action: action_priority(
                board,
                action,
                history_score=self._history.get(action, 0),
            ),
            reverse=True,
        )

        # If a preferred move exists, move it to the front
        if preferred is not None:
            for index, action in enumerate(ordered):
                if action == preferred:
                    ordered.insert(0, ordered.pop(index))
                    break

        return ordered

    def _preferred_action(self, key: tuple) -> Action | None:
        """
        Move ordering hint priority:
        1. TT best move for this exact position
        2. last completed root best move
        """
        entry = self._transposition_table.get(key)
        if entry is not None and entry.best_action is not None:
            return entry.best_action
        return self._last_completed_action

    def _extension(self, board: Board, action: Action, depth: int) -> int:
        """
        Small tactical extension near the search frontier
        
        - see a little further in forcing capture/cascade lines
        - avoid full-depth explosion on every move
        """
        if depth > self._config.tactical_extension_depth_limit:
            return 0
        if is_tactical_action(board, action):
            return self._config.tactical_extension_size
        return 0

    def _allocate_time_budget(self, board: Board, referee: dict) -> float:
        """
        Convert remaining total time into a conservative budget.

        Strategy:
        - keep reserve time for later
        - use smaller budgets in placement
        - gradually allow larger budgets as game progresses
        """
        time_remaining = referee.get("time_remaining")

        # No time info from referee -> use safe defaults
        if time_remaining is None:
            return (
                self._config.default_placement_budget
                if board.phase == GamePhase.PLACEMENT
                else self._config.default_play_budget
            )

        if board.phase == GamePhase.PLACEMENT:
            remaining_own_turns = max(1, (PLACEMENT_TURNS - board.turn_count + 1) // 2)
            reserve = self._config.placement_reserve
        else:
            remaining_total_turns = max(1, MAX_TURNS - board.play_phase_turn_count)
            remaining_own_turns = max(1, remaining_total_turns // 2)
            reserve = self._config.play_reserve

        budget = self._config.budget_fraction * max(0.05, (time_remaining - reserve) / remaining_own_turns)

        if board.phase == GamePhase.PLACEMENT:
            return min(self._config.placement_budget_cap, max(0.05, budget))
        if board.play_phase_turn_count < self._config.opening_turn_threshold:
            return min(self._config.opening_budget_cap, max(0.12, budget))
        if board.play_phase_turn_count < self._config.midgame_turn_threshold:
            return min(self._config.midgame_budget_cap, max(0.18, budget))
        return min(self._config.endgame_budget_cap, max(0.25, budget))

    def _check_timeout(self, deadline: float):
        """
        Abort the current search iteration once the move budget expires
        """
        if perf_counter() >= deadline:
            raise SearchTimeout
