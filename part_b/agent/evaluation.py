from __future__ import annotations

from dataclasses import dataclass

from referee.game import PlayerColor
from referee.game.board import Board, GamePhase
from referee.game.constants import MAX_TURNS

from .config import DEFAULT_EVAL_WEIGHTS, EvalWeights
from .game_utils import (
    CARDINAL_DIRECTIONS,
    adjacent_coord,
    center_value,
    count_legal_actions,
    distance_to_edge,
    edge_penalty,
    friendly_neighbour_count,
    reachable_distance,
    ray_coord,
)


@dataclass(frozen=True, slots=True)
class PositionProfile:
    """
    Feature bundle for one side in one position
    
    Each field captures one strategic aspect of the position
    The evaluator later subtracts opponent features from our faetures
    """
    tokens: int = 0    # total material / token count
    stacks: int = 0    # number of stacks (reflecting felxibility/board coverage)
    mobility: int = 0   # number of legal actions
    center: int = 0    # central control
    support: int = 0    # adjacent friendly stacks
    attacks: int = 0    # immediate EAT opportunities
    threats: int = 0    # immediate enemy EAT threats
    cascade: int = 0    # potential to create useful CASCADEs
    edge_risk: int = 0    # danger from being near board edge
    push_threat: int = 0    # ability to push enemy stacks, espcially off the edge
    vulnerable: int = 0    # tactical fragility / exposure


class Evaluator:
    """
    Feature-based evaluation function
    
    Returns a score from a fixed player's perspective
    Search is responsible for negating when the side to move changes
    """
    
    def __init__(self, weights: EvalWeights | None = None):
        self._weights = weights or DEFAULT_EVAL_WEIGHTS

    def evaluate(self, board: Board, perspective: PlayerColor) -> int:
        """
        Score the position from a fixed perspective
        
        Process:
        1. Build PositionProfile for us
        2. Build PositionProfile for opponent
        3. Compute feature differences
        4. Apply phase sensitive weights
        """
        
        weights = self._weights
        my_profile = self._profile_for(board, perspective)
        opp_profile = self._profile_for(board, perspective.opponent)

        token_diff = my_profile.tokens - opp_profile.tokens
        stack_diff = my_profile.stacks - opp_profile.stacks
        mobility_diff = my_profile.mobility - opp_profile.mobility
        center_diff = my_profile.center - opp_profile.center
        support_diff = my_profile.support - opp_profile.support
        attack_diff = my_profile.attacks - opp_profile.attacks
        threat_diff = opp_profile.threats - my_profile.threats
        cascade_diff = my_profile.cascade - opp_profile.cascade
        edge_diff = opp_profile.edge_risk - my_profile.edge_risk
        push_diff = my_profile.push_threat - opp_profile.push_threat
        vulnerable_diff = opp_profile.vulnerable - my_profile.vulnerable

        # Placement phase emphasizes structure and future potential more than raw material
        if board.phase == GamePhase.PLACEMENT:
            return (
                weights.placement_center * center_diff
                + weights.placement_support * support_diff
                + weights.placement_mobility * mobility_diff
                + weights.placement_edge * edge_diff
                + weights.placement_push * push_diff
                + weights.placement_stack * stack_diff
            )

        # Play phase shifts emphasis towards material and tactical moves
        turn_progress = board.play_phase_turn_count / MAX_TURNS
        token_weight = weights.play_token_early if turn_progress < 0.7 else weights.play_token_late
        mobility_weight = (
            weights.play_mobility_early if turn_progress < 0.5 else weights.play_mobility_late
        )

        return (
            token_weight * token_diff
            + weights.play_stack * stack_diff
            + mobility_weight * mobility_diff
            + weights.play_center * center_diff
            + weights.play_support * support_diff
            + weights.play_attack * attack_diff
            + weights.play_threat * threat_diff
            + weights.play_cascade * cascade_diff
            + weights.play_edge * edge_diff
            + weights.play_push * push_diff
            + weights.play_vulnerable * vulnerable_diff
        )

    def _profile_for(self, board: Board, color: PlayerColor) -> PositionProfile:
        """
        Extract all the feature values for one side
        
        - structural features (center, support, stacks)
        - tactical features (attacks, threats, push threats, vulnerability)
        - dyncamic features (mobility, cascade potential)
        """
        tokens = 0
        stacks = 0
        center = 0
        support = 0
        attacks = 0
        threats = 0
        cascade = 0
        edge_risk = 0
        push_threat = 0
        vulnerable = 0

        mobility = count_legal_actions(board, color)

        for coord, cell in board._state.items():
            if cell.color != color:
                continue

            # Material and structure
            tokens += cell.height
            stacks += 1
            
            # CEnter contribution: capped by min (4, height), so very tall stacks
            # does not dominate purely because of size
            center += center_value(coord) * min(4, cell.height)
            
            # Local friendly support
            support += friendly_neighbour_count(board, coord, color)
            
            # Cascade potential: taller stakcs with more room are more valuable
            cascade += max(0, cell.height - 1) * (2 + reachable_distance(coord))
            
            # Edge risk grows with both proximity and height
            edge_risk += cell.height * edge_penalty(coord)
            
            # Push-based offensive potential
            push_threat += self._push_threat_score(board, coord, color)
            
            # Tactical fragility
            vulnerable += self._vulnerability_score(board, coord, color)

            # Local attack/threat structure
            for direction in CARDINAL_DIRECTIONS:
                neighbour = adjacent_coord(coord, direction)
                if neighbour is None:
                    continue

                neighbour_cell = board._state[neighbour]
                if neighbour_cell.color == color.opponent:
                    if cell.height >= neighbour_cell.height:
                        attacks += 6 + 3 * neighbour_cell.height + (cell.height - neighbour_cell.height)
                    if neighbour_cell.height >= cell.height:
                        threats += 6 + 3 * cell.height + (neighbour_cell.height - cell.height)

        return PositionProfile(
            tokens=tokens,
            stacks=stacks,
            mobility=mobility,
            center=center,
            support=support,
            attacks=attacks,
            threats=threats,
            cascade=cascade,
            edge_risk=edge_risk,
            push_threat=push_threat,
            vulnerable=vulnerable,
        )

    def _push_threat_score(self, board: Board, coord, color: PlayerColor) -> int:
        """
        Estimate how dangerous this stack's cascades are
        
        Positive:
        - enemy stacks that can be pressured or pushed off 
        
        Negative: 
        - friendly stacks that our own cascade would disrupt or endanger
        """
        
        cell = board._state[coord]
        if cell.height < 2:
            return 0

        score = 0
        for direction in CARDINAL_DIRECTIONS:
            for step in range(1, cell.height + 1):
                target = ray_coord(coord, direction, step)
                if target is None:
                    break

                target_cell = board._state[target]
                if target_cell.is_empty:
                    continue

                pushes_available = cell.height - step + 1
                off_edge = pushes_available > distance_to_edge(target, direction)
                if target_cell.color == color.opponent:
                    score += 3 + target_cell.height
                    if off_edge:
                        score += 12 * target_cell.height
                else:
                    score -= 2 + target_cell.height
                    if off_edge:
                        score -= 6 * target_cell.height

        return score

    def _vulnerability_score(self, board: Board, coord, color: PlayerColor) -> int:
        """
        Penalize stacks that are easy to capture or easy to push off-board
        
        Sources of vulnerability:
        - adjacent enemy stacks that can eat
        - edge proximity, because enemy cascades become more dangerous
        """
        cell = board._state[coord]
        score = 0

        for direction in CARDINAL_DIRECTIONS:
            neighbour = adjacent_coord(coord, direction)
            if neighbour is None:
                continue

            neighbour_cell = board._state[neighbour]
            if neighbour_cell.color == color.opponent and neighbour_cell.height >= cell.height:
                score += 4 + 2 * cell.height

        if edge_penalty(coord) > 0:
            score += edge_penalty(coord) * cell.height

        return score
