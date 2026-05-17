from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvalWeights:
    """
    Tunable weights for the evaluation function
    """
    placement_center: int = 24
    placement_support: int = 14
    placement_mobility: int = 6
    placement_edge: int = 5
    placement_push: int = 8
    placement_stack: int = 4

    play_token_early: int = 150
    play_token_late: int = 180
    play_stack: int = 22
    play_mobility_early: int = 6
    play_mobility_late: int = 10
    play_center: int = 8
    play_support: int = 8
    play_attack: int = 18
    play_threat: int = 20
    play_cascade: int = 8
    play_edge: int = 10
    play_push: int = 22
    play_vulnerable: int = 16


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """
    Tunable search parameters
    
    These control:
    - depth behavior
    - tactical extensions
    - quiescence
    - per-move time budgeting
    """
    win_score: int = 1_000_000
    max_placement_depth: int = 4
    late_game_turn_threshold: int = 200
    late_game_depth_increment: int = 2
    tactical_extension_depth_limit: int = 2
    tactical_extension_size: int = 1
    quiescence_depth: int = 2

    default_placement_budget: float = 0.25
    default_play_budget: float = 0.75

    placement_budget_cap: float = 0.8
    opening_budget_cap: float = 1.6
    midgame_budget_cap: float = 2.2
    endgame_budget_cap: float = 3.0

    opening_turn_threshold: int = 80
    midgame_turn_threshold: int = 200
    placement_reserve: float = 8.0
    play_reserve: float = 12.0
    budget_fraction: float = 0.85

# Default configuration used by the main agent unless a variant overrides it
DEFAULT_EVAL_WEIGHTS = EvalWeights()
DEFAULT_SEARCH_CONFIG = SearchConfig()
