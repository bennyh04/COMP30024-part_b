from __future__ import annotations

from referee.game import Action, PlayerColor
from referee.game.board import Board, GamePhase
from .game_utils import board_key, generate_legal_actions

DEPTH_LIMIT = 3

# THIS CANNOT BE 1 BECAUSE HEURISTIC IS BIGGER THAN 1 WHICH WILL MAKE THE HEURISTIC STRONGER THATN WINNING SCORE, basically bad
WIN_SCORE = 1000


class Node:

    def __init__(self, action, children, hash):
        self.action = action
        self.score = 0
        self.children = children
        self.hash = hash


class MiniMaxABSearch:

    def __init__(self, color: PlayerColor):
        self.color = color
        self.depth_limit = DEPTH_LIMIT
        self.search_tree = None
        self.opps_best_child = None

    def choose_action(self, board: Board) -> Action:

        root = None
        if self.search_tree is not None:
            current_key = board_key(board)
            # basically if opps best move is what we expect, aka them minimizing, we dont re generate the tree and just use the subtree immediately
            if self.opps_best_child and current_key == self.opps_best_child.hash:
                root = self.opps_best_child

        # Basically if no root or current search tree is at leaf stage
        if root is None or root.children == []:
            root = Node(action=None, children=[], hash=board_key(board))
            self.minimax(board, self.depth_limit, None, None, True, root)

        self.search_tree = root

        best_child = None
        for child in root.children:
            if best_child is None or child.score > best_child.score:
                best_child = child

        worst_child = None
        for child in best_child.children:
            if worst_child is None or child.score < worst_child.score:
                worst_child = child
        self.opps_best_child = worst_child

        return best_child.action

    def minimax(self, board, depth, alpha, beta, is_our_turn, parent_node):
        if board.game_over:
            winner = board.winner_color
            if winner is None:
                return 0
            return WIN_SCORE if winner == self.color else -WIN_SCORE

        if depth == 0:

            # This is our simple heuristic eval (change for better heuris)
            # BTW heuristic here kinda sucks during placement
            our_tokens = board._count_tokens(self.color)
            opp_tokens = board._count_tokens(self.color.opponent)
            our_stacks = board._count_stacks(self.color)
            opp_stacks = board._count_stacks(self.color.opponent)


            # CHANGE WEIGHT HERE
            return 10*(our_tokens - opp_tokens) + 3*(our_stacks - opp_stacks)

        actions = generate_legal_actions(board)
        if not actions:
            return 0

        best_score_for_current_self = None

        for action in actions:
            board.apply_action(action)
            child_node = Node(action=action, children=[], hash=board_key(board))
            parent_node.children.append(child_node)

            score = self.minimax(board, depth - 1, alpha, beta, not is_our_turn, child_node)
            child_node.score = score
            board.undo_action()

            if is_our_turn:
                if best_score_for_current_self is None or score > best_score_for_current_self:
                    best_score_for_current_self = score
                if beta is not None:
                    if best_score_for_current_self >= beta:
                        break
                if alpha is None or best_score_for_current_self > alpha:
                    alpha = best_score_for_current_self
            else:
                if best_score_for_current_self is None or score < best_score_for_current_self:
                    best_score_for_current_self = score
                if alpha is not None:
                    if best_score_for_current_self <= alpha:
                        break
                if beta is None or best_score_for_current_self < beta:
                    beta = best_score_for_current_self


        return  best_score_for_current_self