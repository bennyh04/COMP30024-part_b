from __future__ import annotations

from referee.game import (
    Action,
    CascadeAction,
    Coord,
    Direction,
    EatAction,
    MoveAction,
    PlaceAction,
    PlayerColor,
)
from referee.game.board import Board, GamePhase
from referee.game.constants import BOARD_N

# All gameplay in Cascade uses the four cardinal direction
CARDINAL_DIRECTIONS: tuple[Direction, ...] = (
    Direction.Up,
    Direction.Down,
    Direction.Left,
    Direction.Right,
)


def adjacent_coord(coord: Coord, direction: Direction) -> Coord | None:
    """
    Return the adjacent coordinate in a direction, or None if it's off the board
    """
    next_r = coord.r + direction.r
    next_c = coord.c + direction.c
    if 0 <= next_r < BOARD_N and 0 <= next_c < BOARD_N:
        return Coord(next_r, next_c)
    return None


def ray_coord(coord: Coord, direction: Direction, distance: int) -> Coord | None:
    """
    Return a coordinate along a cardinal ray, or None if it's off the board
    
    mainly for CASCADE analysis
    """
    target_r = coord.r + direction.r * distance
    target_c = coord.c + direction.c * distance
    if 0 <= target_r < BOARD_N and 0 <= target_c < BOARD_N:
        return Coord(target_r, target_c)
    return None


def center_value(coord: Coord) -> int:
    """
    Heuristic centrality score
    
    Higher means closer to the center of the board
    """
    return 14 - abs(2 * coord.r - 7) - abs(2 * coord.c - 7)


def edge_penalty(coord: Coord) -> int:
    """
    Heuristic edge danger score
    
    Higher means closer to the edge
    """
    nearest_edge = min(coord.r, coord.c, BOARD_N - 1 - coord.r, BOARD_N - 1 - coord.c)
    return 3 - min(3, nearest_edge)


def reachable_distance(coord: Coord) -> int:
    """
    Rough measure of how much room a stack has in the board geometry
    """
    return max(coord.r, coord.c, BOARD_N - 1 - coord.r, BOARD_N - 1 - coord.c)


def distance_to_edge(coord: Coord, direction: Direction) -> int:
    """
    Number of cells from coord to the edge of the board in a given direction
    
    Used to estimate whether a CASCADE could push to eliminate a stack
    """
    if direction == Direction.Up:
        return coord.r
    if direction == Direction.Down:
        return BOARD_N - 1 - coord.r
    if direction == Direction.Left:
        return coord.c
    return BOARD_N - 1 - coord.c


def adjacent_to_color(board: Board, coord: Coord, color: PlayerColor) -> bool:
    """
    Check whether a square is adjacent to a given color
    """
    for direction in CARDINAL_DIRECTIONS:
        neighbour = adjacent_coord(coord, direction)
        if neighbour is not None and board._state[neighbour].color == color:
            return True
    return False


def friendly_neighbour_count(board: Board, coord: Coord, color: PlayerColor) -> int:
    """
    Count adjacent friendly stacks
    """
    count = 0
    for direction in CARDINAL_DIRECTIONS:
        neighbour = adjacent_coord(coord, direction)
        if neighbour is not None and board._state[neighbour].color == color:
            count += 1
    return count


def open_neighbour_count(board: Board, coord: Coord) -> int:
    """
    Count adjacent empty cells
    """
    count = 0
    for direction in CARDINAL_DIRECTIONS:
        neighbour = adjacent_coord(coord, direction)
        if neighbour is not None and board._state[neighbour].is_empty:
            count += 1
    return count


def board_key(board: Board) -> tuple:
    """
    Build a hashable representation of the entire search state
    
    Includes:
    - phase
    - side to move
    - placement count
    - board contents
    
    Used for transposition table cahcing
    """
    return (
        board.phase,
        board.turn_color,
        board._placement_count,
        tuple(
            (cell.color, cell.height)
            for _, cell in sorted(board._state.items())
        ),
    )


def generate_legal_actions(board: Board) -> list[Action]:
    """
    Dispatch to the correct legal move generator for the current phase
    """
    if board.phase == GamePhase.PLACEMENT:
        return generate_placement_actions(board, board.turn_color)
    return generate_play_actions(board, board.turn_color)


def generate_placement_actions(board: Board, color: PlayerColor) -> list[Action]:
    """
    Generate all legal PLACE actions during placement phase
    
    - can only place on empty cells
    - cannot place adjacent to opponent cells
    """
    actions: list[Action] = []

    for coord, cell in board._state.items():
        if cell.is_stack:
            continue
        if board._placement_count > 0 and adjacent_to_color(board, coord, color.opponent):
            continue
        actions.append(PlaceAction(coord))

    return actions


def generate_play_actions(board: Board, color: PlayerColor) -> list[Action]:
    """
    Generate all legal MOVE/EAT/CASCADE actions for one side
    
    - MOVE onto an empty cell or friendly stack
    - EAT only if adjacent opponent stack exists and attacker height >= target height
    - CASCADE only if stack height >= 2
    """
    actions: list[Action] = []

    for coord, cell in board._state.items():
        if cell.color != color:
            continue

        for direction in CARDINAL_DIRECTIONS:
            dest = adjacent_coord(coord, direction)
            if dest is None:
                continue

            dest_cell = board._state[dest]
            if dest_cell.is_empty or dest_cell.color == color:
                actions.append(MoveAction(coord, direction))
            elif cell.height >= dest_cell.height:
                actions.append(EatAction(coord, direction))

        if cell.height >= 2:
            for direction in CARDINAL_DIRECTIONS:
                actions.append(CascadeAction(coord, direction))

    return actions


def count_legal_actions(board: Board, color: PlayerColor) -> int:
    """
    Count legal moves for a player
    """
    if board.phase == GamePhase.PLACEMENT:
        return len(generate_placement_actions(board, color))
    return len(generate_play_actions(board, color))


def cascade_push_pressure(board: Board, coord: Coord, direction: Direction) -> int:
    """
    Heuristic estimate of how forcing a CASCADE in this direction would be

    Positive values:
    - pressure on enemy stacks
    
    Negative values:
    - friendly stacks in the path
    """
    src = board._state[coord]
    if src.is_empty or src.height < 2:
        return 0

    pressure = 0
    for step in range(1, src.height + 1):
        target = ray_coord(coord, direction, step)
        if target is None:
            # Some pressure credit even if later cascade tokens go off the board
            pressure += 4
            continue

        target_cell = board._state[target]
        if target_cell.is_empty:
            continue

        pushes_available = src.height - step + 1
        off_edge = pushes_available > distance_to_edge(target, direction)
        off_edge_bonus = 18 * target_cell.height if off_edge else 0
        if target_cell.color == board.turn_color.opponent:
            pressure += 20 + 6 * target_cell.height + off_edge_bonus
        else:
            pressure -= 8 + 3 * target_cell.height + off_edge_bonus // 2

    return pressure


def is_tactical_action(board: Board, action: Action) -> bool:
    """
    Identify actions as tactical / loud
    
    - all EAT moves are tactical
    - CASCADE is tactical if push pressure is high enough
    - MOVE is tactical if it merges into a friendly stack   
    """
    match action:
        case EatAction():
            return True
        case CascadeAction(coord, direction):
            return cascade_push_pressure(board, coord, direction) > 10
        case MoveAction(coord, direction):
            dest = adjacent_coord(coord, direction)
            if dest is None:
                return False
            dest_cell = board._state[dest]
            return dest_cell.is_stack and dest_cell.color == board.turn_color
        case _:
            return False


def generate_loud_actions(board: Board) -> list[Action]:
    """
    Generate only the tactical actions
    
    Used in quiiescence search to extend unstable positions without expanding the full branching factor
    """
    if board.phase == GamePhase.PLACEMENT:
        return []

    actions: list[Action] = []
    for action in generate_play_actions(board, board.turn_color):
        if is_tactical_action(board, action):
            actions.append(action)
    return actions


def action_priority(board: Board, action: Action, history_score: int = 0) -> int:
    """
    Static move ordering heuristic used before deeper search information is available
    
    higher score = search earlier
    
    - PLACE: central, open, "safer" squares first
    - EAT: very high priority
    - CASCADE: high if stack is big and push pressure is strong
    - MOVE: merging moves outrank quiet relocations
    - history_score: learned bonus from privous cutoffs
    """
    match action:
        case PlaceAction(coord):
            return (
                30 * center_value(coord)
                + 8 * open_neighbour_count(board, coord)
                - 10 * edge_penalty(coord)
                - 4 * friendly_neighbour_count(board, coord, board.turn_color)
                + history_score
            )

        case EatAction(coord, direction):
            target = board._state[coord + direction]
            return 900 + 80 * target.height + 4 * center_value(coord + direction) + history_score

        case CascadeAction(coord, direction):
            src = board._state[coord]
            return 400 + 28 * src.height + cascade_push_pressure(board, coord, direction) + history_score

        case MoveAction(coord, direction):
            src = board._state[coord]
            dest = board._state[coord + direction]
            destination_coord = coord + direction
            if dest.is_empty:
                return (
                    140
                    + 14 * center_value(destination_coord)
                    - 10 * edge_penalty(destination_coord)
                    + history_score
                )
            return (
                260
                + 20 * (src.height + dest.height)
                + 5 * center_value(destination_coord)
                - 12 * edge_penalty(destination_coord)
                + history_score
            )

        case _:
            return history_score
