"""
NPC AI player implementations for Dablo game

Provides strategic AI opponents at various difficulty levels and play styles.
Includes smart, aggressive, defensive, and random NPC variants.
"""

import random

from ..core.game import DabloGame
from ..core.pieces import PieceType, get_piece_value
from ..core.player import Player
from ..core.rules import Move
from .base import NPCPlayer
from .config import npc_config


P1_KING = PieceType.P1_KING
P2_KING = PieceType.P2_KING
EMPTY = PieceType.EMPTY


class SmartNPCPlayer(NPCPlayer):
    """Smart NPC that uses strategic heuristics"""

    def __init__(self, player_id: Player, difficulty: str = "medium"):
        super().__init__(player_id, difficulty)

        # Get difficulty settings from config
        self.settings = npc_config.get_difficulty_settings(difficulty)
        self.opponent_id = -self.player_id

        # Cache for expensive calculations
        self._threat_cache = {}

        # Cache statistics
        self._threat_cache_hits = 0
        self._threat_cache_misses = 0

    def clear_cache(self):
        """Clear caches to prevent memory buildup between games"""
        self._threat_cache.clear()

    def get_move(self, game: DabloGame) -> Move | None:
        """
        Selects the best move by evaluating all valid options.

        Applies randomness based on difficulty and uses a weighted selection from the
        top-scoring moves to ensure varied and less predictable gameplay.
        """
        valid_moves = game.get_all_valid_moves()
        if not valid_moves:
            return None

        # Apply a chance for a completely random move based on difficulty
        if random.random() < self.settings.randomness:
            return random.choice(valid_moves)

        # Score all moves and sort them from best to worst
        move_scores = [(move, self._evaluate_move(game, move)) for move in valid_moves]
        move_scores.sort(key=lambda x: x[1], reverse=True)

        # Select a subset of the best moves to introduce variety
        top_count = npc_config.move_selection.top_moves_count
        top_moves_with_scores = move_scores[:top_count]

        if not top_moves_with_scores:
            # This case is unlikely if valid_moves is not empty, but it's safe to handle
            return valid_moves[0] if valid_moves else None

        # Perform a weighted random choice among the top moves
        top_moves = [ms[0] for ms in top_moves_with_scores]
        weights = npc_config.move_selection.selection_weights[: len(top_moves)]

        # random.choices returns a list, so we select the first element
        return random.choices(top_moves, weights=weights, k=1)[0]

    def _get_board_hash(self, board_state: dict) -> tuple:
        """Create hashable representation of board state for caching"""
        return tuple(sorted(board_state.items()))

    def _evaluate_move(self, game: DabloGame, move: Move) -> float:
        """
        Calculates a score for a given move based on multiple strategic heuristics.

        This is the core of the AI's decision-making process. It simulates the
        move once and passes the resulting state to various evaluation helpers.
        """
        # Simulate the move once to get the resulting board state
        board_after_move = self._simulate_move_on_board(game.board_state, move)

        # Score the move based on its immediate impact (e.g., captures)
        score = 0.0
        if move.is_capture:
            captured_piece = game.board_state[move.capture_pos]
            score += get_piece_value(captured_piece) * self.settings.capture_preference

            # Check for potential chain captures after the first one
            game_after_capture = self._create_temp_game_state(game, board_after_move)
            next_moves = game_after_capture.get_valid_moves(move.to_pos)
            chain_captures = [m for m in next_moves if m.is_capture]

            if chain_captures:
                score += npc_config.evaluation.chain_capture_bonus
                # Add the value of the best subsequent capture to the score
                best_follow_up_value = max(
                    get_piece_value(
                        game_after_capture.board_state.get(c.capture_pos, EMPTY)
                    )
                    for c in chain_captures
                )
                score += best_follow_up_value

        # Add scores from various strategic heuristics
        score += (
            self._evaluate_king_safety(game, move, board_after_move)
            * self.settings.king_safety
        )
        score += (
            self._evaluate_center_control(move) * npc_config.evaluation.center_control
        )
        score += (
            self._evaluate_piece_protection(game, move, board_after_move)
            * npc_config.evaluation.piece_protection
        )
        score += (
            self._evaluate_threat_creation(game, board_after_move)
            * npc_config.evaluation.threat_creation
        )

        return score

    def _evaluate_king_safety(
        self, game: DabloGame, move: Move, board_after_move: dict
    ) -> float:
        """Evaluate how this move affects king safety"""
        # Find our king position
        king_type = P1_KING if self.player_id == Player.P1 else P2_KING

        # Find the king's position on the new board
        king_pos = None
        for pos, piece in board_after_move.items():
            if piece == king_type:
                king_pos = pos
                break

        if not king_pos:
            # This would mean the king was captured, which is a game-losing move
            return npc_config.king_safety.immediate_threat_penalty

        # Create a temporary game state to check the opponent's reply
        temp_game = self._create_temp_game_state(
            game, board_after_move, player_to_move=self.opponent_id
        )

        # Check for immediate capture threats to the king
        for opp_move in temp_game.get_all_valid_moves():
            if opp_move.is_capture and opp_move.capture_pos == king_pos:
                return npc_config.king_safety.capture_danger_penalty

        # Evaluate king safety based on distance to the nearest enemy piece ---
        kr, kc = king_pos

        enemy_positions = [
            (((kr - r) ** 2 + (kc - c) ** 2) ** 0.5)
            for (r, c), piece in board_after_move.items()
            if piece != EMPTY and (piece.value > 0) != (self.player_id.value > 0)
        ]

        # If no enemies left, king is very safe
        if not enemy_positions:
            return npc_config.king_safety.very_safe_bonus

        min_dist = min(enemy_positions)

        thresholds = [
            (
                npc_config.king_safety.immediate_danger_threshold,
                npc_config.king_safety.close_distance_penalty,
            ),
            (
                npc_config.king_safety.close_danger_threshold,
                npc_config.king_safety.medium_distance_penalty,
            ),
            (
                npc_config.king_safety.medium_safety_threshold,
                npc_config.king_safety.safe_distance_bonus,
            ),
        ]
        for threshold, score in thresholds:
            if min_dist < threshold:
                return score

        return npc_config.king_safety.very_safe_bonus

    def _evaluate_center_control(self, move: Move) -> float:
        """Rewards moves that place pieces in the center of the board."""
        to_r, to_c = move.to_pos
        bonus = 0.0
        if to_r in npc_config.center_control.center_rows:
            bonus += npc_config.center_control.row_bonus
        if to_c in npc_config.center_control.center_cols:
            bonus += npc_config.center_control.col_bonus
        return bonus

    def _evaluate_piece_protection(
        self, game: DabloGame, move: Move, board_after_move: dict
    ) -> float:
        """Rewards moves that reduce threats to valuable pieces."""
        moving_piece = game.board_state.get(move.from_pos)
        if not moving_piece:
            return 0.0

        piece_value = get_piece_value(moving_piece)
        if piece_value < npc_config.move_selection.min_piece_value_for_protection:
            return 0.0  # Only evaluate for more valuable pieces (e.g., Princes, Kings)

        # To evaluate threat reduction, we check threats before and after the move.
        threat_before = self._get_threat_level(game, move.from_pos, self.player_id)
        if threat_before == 0:
            return 0.0  # Piece was not threatened, so no bonus for moving it.

        game_after_move = self._create_temp_game_state(game, board_after_move)
        threat_after = self._get_threat_level(
            game_after_move, move.to_pos, self.player_id
        )

        threat_reduction = threat_before - threat_after
        return threat_reduction * piece_value  # Saving a King is worth more

    def _get_threat_level(
        self, game: DabloGame, pos: tuple[int, int], for_player_id: Player
    ) -> int:
        """
        Counts how many opponent pieces can capture a given position.

        This is a simplified threat analysis. A full analysis would require a deeper search.
        """
        # Check cache first
        board_hash = self._get_board_hash(game.board_state)
        cache_key = (board_hash, pos, for_player_id)
        if cache_key in self._threat_cache:
            self._threat_cache_hits += 1
            return self._threat_cache[cache_key]

        self._threat_cache_misses += 1
        temp_game = self._create_temp_game_state(
            game, game.board_state, player_to_move=-for_player_id
        )

        # Count immediate capture threats
        immediate_threats = sum(
            1
            for mv in temp_game.get_all_valid_moves()
            if mv.is_capture and mv.capture_pos == pos
        )
        if immediate_threats > 0:
            return immediate_threats

        # Approximate potential threats by checking if an opponent can move adjacent
        r_pos, c_pos = pos
        potential_threats = 0
        for move in temp_game.get_all_valid_moves():
            if not move.is_capture:
                r_to, c_to = move.to_pos
                # Check if the move lands on a square adjacent to the target position
                if abs(r_to - r_pos) <= 1 and abs(c_to - c_pos) <= 1:
                    potential_threats += 1

        # Cache and return result
        result = immediate_threats if immediate_threats > 0 else potential_threats
        self._threat_cache[cache_key] = result
        return result

    def _evaluate_threat_creation(
        self, game: DabloGame, board_after_move: dict
    ) -> float:
        """Rewards moves that create new capture threats against the opponent."""
        temp_game = self._create_temp_game_state(
            game, board_after_move, player_to_move=self.player_id
        )

        # Find all capture moves we can make from the new state
        capture_moves = [m for m in temp_game.get_all_valid_moves() if m.is_capture]

        threat_value = sum(
            get_piece_value(temp_game.board_state.get(m.capture_pos, EMPTY))
            for m in capture_moves
        )

        # Apply a multiplier and a cap to keep the value in a reasonable range
        return min(threat_value * 0.3, npc_config.evaluation.max_threat_value)

    @staticmethod
    def _simulate_move_on_board(board_state: dict, move: Move) -> dict:
        """
        Applies a move to a board state dictionary and returns the new state.
        This is a lightweight, pure function that avoids object mutation.
        """
        new_board = board_state.copy()
        moving_piece = new_board.pop(move.from_pos, EMPTY)
        new_board[move.to_pos] = moving_piece

        if move.is_capture and move.capture_pos:
            new_board[move.capture_pos] = EMPTY

        return new_board

    def _create_temp_game_state(
        self,
        original_game: DabloGame,
        board_state: dict,
        player_to_move: Player | None = None,
    ) -> DabloGame:
        """
        Creates a temporary, lightweight DabloGame instance for analysis.
        """
        temp_game = DabloGame(config=original_game.config, initial_state="empty")
        temp_game.board_state = board_state.copy()
        # If a player is specified, set them as the current player. Otherwise, use the original.
        temp_game.current_player = (
            player_to_move if player_to_move is not None else self.player_id
        )
        return temp_game


class RandomNPCPlayer(NPCPlayer):
    """A simple NPC that chooses a random valid move."""

    def get_move(self, game: DabloGame) -> Move | None:
        valid_moves = game.get_all_valid_moves()
        return random.choice(valid_moves) if valid_moves else None


class AggressiveNPCPlayer(SmartNPCPlayer):
    """An aggressive NPC that prioritizes captures and forward movement."""

    def __init__(self, player_id: Player, difficulty: str = "hard"):
        super().__init__(player_id, difficulty)
        # Override default difficulty settings with aggressive-specific values
        self.settings = npc_config.aggressive


class DefensiveNPCPlayer(SmartNPCPlayer):
    """A defensive NPC that prioritizes king safety and piece protection."""

    def __init__(self, player_id: Player, difficulty: str = "medium"):
        super().__init__(player_id, difficulty)
        # Override default difficulty settings with defensive-specific values
        self.settings = npc_config.defensive


def create_npc_player(
    player: Player, npc_type: str = "smart", difficulty: str = "medium"
) -> NPCPlayer:
    """Factory function to create different types of NPC players."""
    npc_classes = {
        "random": RandomNPCPlayer,
        "smart": SmartNPCPlayer,
        "aggressive": AggressiveNPCPlayer,
        "defensive": DefensiveNPCPlayer,
    }

    npc_class = npc_classes.get(npc_type.lower(), SmartNPCPlayer)
    return npc_class(player, difficulty=difficulty)
