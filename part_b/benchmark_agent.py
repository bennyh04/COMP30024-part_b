from __future__ import annotations

import argparse
import random
import sys
import types
from dataclasses import dataclass
from statistics import mean

# To get stats:
# python benchmark_agent.py --variants balanced --opponents random greedy shallow --move-time 1.0

def install_websockets_stub():
    """
    The referee may import websockets even though this benchmark script does not actually need live websocket funcitonality
    
    This stub makes local benchmarking possible even if websockets is missing
    """
    try:
        import websockets  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    stub = types.ModuleType("websockets")

    class _DummyServer:
        connections = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def serve(*args, **kwargs):
        return _DummyServer()

    stub.serve = serve
    sys.modules["websockets"] = stub


install_websockets_stub()

from agent import Agent
from agent.config import DEFAULT_EVAL_WEIGHTS, DEFAULT_SEARCH_CONFIG, EvalWeights, SearchConfig
from agent.evaluation import Evaluator
from agent.game_utils import action_priority, generate_legal_actions
from agent.search import SearchEngine
from referee.game import PlayerColor
from referee.game.board import Board


@dataclass(frozen=True, slots=True)
class Variant:
    """
    Named bundle of evaluation weights + search config
    
    Used to test different versions of the agent systematically
    """
    name: str
    eval_weights: EvalWeights
    search_config: SearchConfig


@dataclass(slots=True)
class GameResult:
    """
    Summary of one compelted benchmark game
    """
    winner: PlayerColor | None
    turns: int
    red_tokens: int
    blue_tokens: int
    search_nodes: int
    qnodes: int
    tt_hits: int
    depth: int


class RandomAgent:
    """
    Very weak baseline: chooses uniformly at random from legal actions
    """
    def __init__(self, color: PlayerColor):
        self._color = color
        self._board = Board(initial_player=PlayerColor.RED)

    def action(self, **referee):
        return random.choice(generate_legal_actions(self._board))

    def update(self, color, action, **referee):
        self._board.apply_action(action)


class GreedyAgent:
    """
    Slightly stronger baseline: one-ply lookahead using evaluation only, no deeper adversarial search
    """
    def __init__(self, color: PlayerColor):
        self._color = color
        self._board = Board(initial_player=PlayerColor.RED)
        self._evaluator = Evaluator()

    def action(self, **referee):
        actions = sorted(
            generate_legal_actions(self._board),
            key=lambda action: action_priority(self._board, action),
            reverse=True,
        )

        best_action = actions[0]
        best_score = -10**18
        for action in actions:
            self._board.apply_action(action)
            try:
                score = self._evaluator.evaluate(self._board, self._color)
            finally:
                self._board.undo_action()

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def update(self, color, action, **referee):
        self._board.apply_action(action)


class ShallowSearchAgent:
    """
    Search baseline: uses the same main search engine but with much weaker budgets, settings
    """
    
    def __init__(self, color: PlayerColor):
        self._color = color
        self._board = Board(initial_player=PlayerColor.RED)
        self._search = SearchEngine(
            color,
            config=SearchConfig(
                max_placement_depth=2,
                late_game_turn_threshold=999,
                late_game_depth_increment=1,
                tactical_extension_depth_limit=0,
                tactical_extension_size=0,
                quiescence_depth=1,
                default_placement_budget=0.10,
                default_play_budget=0.20,
                placement_budget_cap=0.25,
                opening_budget_cap=0.35,
                midgame_budget_cap=0.45,
                endgame_budget_cap=0.55,
            ),
        )

    def action(self, **referee):
        return self._search.choose_action(self._board, **referee)

    def update(self, color, action, **referee):
        self._board.apply_action(action)


def make_project_agent(variant: Variant):
    """
    Factory wrapper so benchmarking code can initialize our main agent with diffrent configurations cleanly
    """
    def factory(color: PlayerColor):
        return Agent(
            color,
            search_config=variant.search_config,
            eval_weights=variant.eval_weights,
        )

    return factory


def variant_library() -> dict[str, Variant]:
    """
    Define the named agent variants we want to compare 
    
    "balanced": default configuration
    
    "pressure": more tactically aggressive, especially push, attack, threat
    
    "mobility": values flexibility and centrality more heavily
    """
    
    balanced = Variant(
        "balanced",
        DEFAULT_EVAL_WEIGHTS,
        DEFAULT_SEARCH_CONFIG,
    )
    pressure = Variant(
        "pressure",
        EvalWeights(
            placement_center=22,
            placement_support=13,
            placement_mobility=5,
            placement_edge=5,
            placement_push=10,
            placement_stack=4,
            play_token_early=142,
            play_token_late=180,
            play_stack=22,
            play_mobility_early=6,
            play_mobility_late=10,
            play_center=7,
            play_support=7,
            play_attack=24,
            play_threat=26,
            play_cascade=10,
            play_edge=10,
            play_push=30,
            play_vulnerable=20,
        ),
        SearchConfig(
            max_placement_depth=4,
            late_game_turn_threshold=180,
            late_game_depth_increment=2,
            tactical_extension_depth_limit=3,
            tactical_extension_size=1,
            quiescence_depth=3,
            default_placement_budget=0.25,
            default_play_budget=0.90,
            placement_budget_cap=0.8,
            opening_budget_cap=1.9,
            midgame_budget_cap=2.5,
            endgame_budget_cap=3.2,
        ),
    )
    mobility = Variant(
        "mobility",
        EvalWeights(
            placement_center=26,
            placement_support=16,
            placement_mobility=8,
            placement_edge=5,
            placement_push=8,
            placement_stack=4,
            play_token_early=145,
            play_token_late=172,
            play_stack=20,
            play_mobility_early=10,
            play_mobility_late=14,
            play_center=10,
            play_support=10,
            play_attack=16,
            play_threat=18,
            play_cascade=8,
            play_edge=8,
            play_push=18,
            play_vulnerable=14,
        ),
        SearchConfig(
            max_placement_depth=4,
            late_game_turn_threshold=210,
            late_game_depth_increment=2,
            tactical_extension_depth_limit=2,
            tactical_extension_size=1,
            quiescence_depth=2,
            default_placement_budget=0.20,
            default_play_budget=0.70,
            placement_budget_cap=0.7,
            opening_budget_cap=1.4,
            midgame_budget_cap=2.0,
            endgame_budget_cap=2.8,
        ),
    )
    return {variant.name: variant for variant in (balanced, pressure, mobility)}


def opponent_library():
    """
    Library of baseline opponenets for local experiments
    """
    return {
        "random": RandomAgent,
        "greedy": GreedyAgent,
        "shallow": ShallowSearchAgent,
    }


def play_game(red_factory, blue_factory, move_time: float, seed: int) -> GameResult:
    """
    Play one local benchmark game between the two agents
    """
    random.seed(seed)

    board = Board(initial_player=PlayerColor.RED)
    red = red_factory(PlayerColor.RED)
    blue = blue_factory(PlayerColor.BLUE)
    agents = {
        PlayerColor.RED: red,
        PlayerColor.BLUE: blue,
    }

    project_move_stats: list[tuple[int, int, int, int]] = []
    turns = 0
    while not board.game_over and turns < 400:
        color = board.turn_color
        actor = agents[color]
        action = actor.action(time_remaining=move_time)
        board.apply_action(action)
        red.update(color, action)
        blue.update(color, action)
        turns += 1

        # Collect stats only for our main agent implementation
        if isinstance(actor, Agent):
            stats = actor.last_search_stats
            project_move_stats.append(
                (stats.nodes, stats.qnodes, stats.tt_hits, stats.deepest_depth)
            )

    if project_move_stats:
        avg_nodes = round(mean(item[0] for item in project_move_stats))
        avg_qnodes = round(mean(item[1] for item in project_move_stats))
        avg_tt_hits = round(mean(item[2] for item in project_move_stats))
        avg_depth = round(mean(item[3] for item in project_move_stats), 2)
    else:
        avg_nodes = avg_qnodes = avg_tt_hits = 0
        avg_depth = 0

    return GameResult(
        winner=board.winner_color,
        turns=turns,
        red_tokens=board.red_tokens,
        blue_tokens=board.blue_tokens,
        search_nodes=avg_nodes,
        qnodes=avg_qnodes,
        tt_hits=avg_tt_hits,
        depth=avg_depth,
    )


def evaluate_variant(
    variant: Variant,
    opponent_name: str,
    games_per_side: int,
    move_time: float,
    seed: int,
):
    """
    Evaluate one variant against one opponent
    
    Play both colors equally often so results are less biased by first-player advantage
    """
    opponent_factory = opponent_library()[opponent_name]
    project_factory = make_project_agent(variant)
    results: list[GameResult] = []

    for index in range(games_per_side):
        results.append(
            play_game(
                project_factory,
                opponent_factory,
                move_time,
                seed + 100 * index,
            )
        )
        results.append(
            play_game(
                opponent_factory,
                project_factory,
                move_time,
                seed + 100 * index + 1,
            )
        )

    wins = 0
    draws = 0
    losses = 0

    for game_index, result in enumerate(results):
        project_as_red = game_index % 2 == 0
        if result.winner is None:
            draws += 1
        elif (project_as_red and result.winner == PlayerColor.RED) or (
            not project_as_red and result.winner == PlayerColor.BLUE
        ):
            wins += 1
        else:
            losses += 1

    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "points": wins + 0.5 * draws,
        "avg_turns": round(mean(result.turns for result in results), 2),
        "avg_nodes": round(mean(result.search_nodes for result in results)),
        "avg_qnodes": round(mean(result.qnodes for result in results)),
        "avg_tt_hits": round(mean(result.tt_hits for result in results)),
        "avg_depth": round(mean(result.depth for result in results), 2),
    }


def main():
    """
    Commandline entry points for local experiments
    
    Example: python benchmark_agent.py --variants balanced --opponents random greedy shallow --move-time 1.0
    """
    parser = argparse.ArgumentParser(description="Local benchmark runner for the Cascade agent.")
    parser.add_argument("--variants", nargs="+", default=["balanced", "pressure", "mobility"])
    parser.add_argument("--opponents", nargs="+", default=["random", "greedy", "shallow"])
    parser.add_argument("--games-per-side", type=int, default=2)    # e.g. plays as RED 2 times, BLUE 2 times: total=4 games
    parser.add_argument("--move-time", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    variants = variant_library()
    total_scores: dict[str, float] = {}

    for variant_name in args.variants:
        variant = variants[variant_name]
        total_scores[variant.name] = 0.0
        print(f"\nVariant: {variant.name}")
        print("opponent   w-d-l   points   avg_turns   avg_nodes   avg_qnodes   avg_tt_hits   avg_depth")
        for opponent_name in args.opponents:
            summary = evaluate_variant(
                variant,
                opponent_name,
                args.games_per_side,
                args.move_time,
                args.seed,
            )
            total_scores[variant.name] += summary["points"]
            print(
                f"{opponent_name:<9} "
                f"{summary['wins']}-{summary['draws']}-{summary['losses']:<5} "
                f"{summary['points']:<8.1f} "
                f"{summary['avg_turns']:<11} "
                f"{summary['avg_nodes']:<11} "
                f"{summary['avg_qnodes']:<12} "
                f"{summary['avg_tt_hits']:<13} "
                f"{summary['avg_depth']}"
            )

    print("\nTotal Scores")
    for name, score in sorted(total_scores.items(), key=lambda item: item[1], reverse=True):
        print(f"{name:<9} {score:.1f}")


if __name__ == "__main__":
    main()
