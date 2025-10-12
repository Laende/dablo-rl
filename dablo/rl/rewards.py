"""
Reward calculation system for Dablo RL environment.

Multi-component reward system including material evaluation, positional
bonuses, and terminal rewards to guide RL agent learning.
"""

from ..core.game import DabloGame
from ..core.pieces import PieceType, get_piece_value
from ..core.player import Player


class DabloRewardCalculator:
    """Calculates rewards for Dablo RL training."""

    def __init__(
        self,
        material_weight: float = 1.0,
        positional_weight: float = 0.1,
        capture_bonus: float = 5.0,
        win_reward: float = 100.0,
        loss_penalty: float = -100.0,
        draw_reward: float = 0.0,
        step_penalty: float = -0.01,
    ):
        """
        Initialize reward calculator with component weights.

        Args:
            material_weight: Multiplier for material advantage
            positional_weight: Multiplier for positional bonuses
            capture_bonus: Extra reward bonus for capturing pieces
            win_reward: Reward for winning the game
            loss_penalty: Penalty for losing the game
            draw_reward: Reward for draw
            step_penalty: Small penalty per move to encourage decisive play
        """
        self.material_weight = material_weight
        self.positional_weight = positional_weight
        self.capture_bonus = capture_bonus
        self.win_reward = win_reward
        self.loss_penalty = loss_penalty
        self.draw_reward = draw_reward
        self.step_penalty = step_penalty

        # Cache piece values for efficiency
        self.piece_values = {
            PieceType.P1_WARRIOR: get_piece_value(PieceType.P1_WARRIOR),
            PieceType.P1_PRINCE: get_piece_value(PieceType.P1_PRINCE),
            PieceType.P1_KING: get_piece_value(PieceType.P1_KING),
            PieceType.P2_WARRIOR: get_piece_value(PieceType.P2_WARRIOR),
            PieceType.P2_PRINCE: get_piece_value(PieceType.P2_PRINCE),
            PieceType.P2_KING: get_piece_value(PieceType.P2_KING),
            PieceType.EMPTY: 0.0,
        }

    def calculate_material_value(
        self, board_state: dict[tuple[float, float], PieceType], player: Player
    ) -> float:
        """Calculate total material value for a player."""
        total_value = 0.0

        for piece_type in board_state.values():
            if piece_type == PieceType.EMPTY:
                continue

            # Check if piece belongs to player
            piece_player = Player.P1 if piece_type.name.startswith("P1") else Player.P2
            if piece_player == player:
                total_value += self.piece_values[piece_type]

        return total_value

    def calculate_material_advantage(
        self, board_state: dict[tuple[float, float], PieceType], player: Player
    ) -> float:
        """Calculate material advantage (own material - opponent material)."""
        own_material = self.calculate_material_value(board_state, player)
        opponent_player = Player.P2 if player == Player.P1 else Player.P1
        opponent_material = self.calculate_material_value(board_state, opponent_player)

        return own_material - opponent_material

    def calculate_positional_bonus(self, game: DabloGame, player: Player) -> float:
        """
        Calculate positional bonuses for strategic positioning.

        Factors:
        - King safety (distance from center and enemy pieces)
        - Piece advancement toward opponent
        - Piece mobility (number of valid moves)
        """
        bonus = 0.0
        board_state = game.board_state

        # King safety bonus
        king_type = PieceType.P1_KING if player == Player.P1 else PieceType.P2_KING
        king_positions = [
            pos for pos, piece in board_state.items() if piece == king_type
        ]

        if king_positions:
            king_pos = king_positions[0]  # Should only be one king

            # Bonus for king being away from center (safer)
            center_x, center_y = 2.5, 2.0  # Approximate board center
            distance_from_center = abs(king_pos[0] - center_x) + abs(
                king_pos[1] - center_y
            )
            bonus += distance_from_center * 0.1

            # Bonus for king being away from enemy pieces
            enemy_pieces = [
                pos
                for pos, piece in board_state.items()
                if piece != PieceType.EMPTY
                and not piece.name.startswith(f"P{player.value}")
            ]

            if enemy_pieces:
                min_enemy_distance = min(
                    abs(king_pos[0] - enemy_pos[0]) + abs(king_pos[1] - enemy_pos[1])
                    for enemy_pos in enemy_pieces
                )
                bonus += min_enemy_distance * 0.05

        # Piece advancement bonus (simple heuristic)
        own_pieces = [
            pos
            for pos, piece in board_state.items()
            if piece != PieceType.EMPTY and piece.name.startswith(f"P{player.value}")
        ]

        for piece_pos in own_pieces:
            if player == Player.P1:
                # P1 advances "upward" (decreasing x coordinate)
                advancement = 5.0 - piece_pos[0]  # Max x is 5
            else:
                # P2 advances "downward" (increasing x coordinate)
                advancement = piece_pos[0]

            bonus += advancement * 0.02

        return bonus

    def calculate_step_reward(
        self, game: DabloGame, player: Player, previous_game_state: dict | None = None
    ) -> float:
        """
        Calculate reward for a single step.

        Combines material advantage change and positional bonuses.
        """
        reward = 0.0

        # Material advantage reward
        material_advantage = self.calculate_material_advantage(game.board_state, player)
        reward += material_advantage * self.material_weight

        # Capture bonus - check if this move resulted in a capture
        if game.last_captured_piece is not None:
            captured_piece_type = (
                previous_game_state.get(game.last_captured_piece)
                if previous_game_state
                else None
            )
            if captured_piece_type:
                captured_value = self.piece_values.get(captured_piece_type, 1.0)
                reward += self.capture_bonus * captured_value
            else:
                # Default capture bonus if we can't determine piece type
                reward += self.capture_bonus

        # Positional bonus
        positional_bonus = self.calculate_positional_bonus(game, player)
        reward += positional_bonus * self.positional_weight

        # Step penalty to encourage decisive play
        reward += self.step_penalty

        return reward

    def calculate_terminal_reward(self, game: DabloGame, player: Player) -> float:
        """Calculate reward for terminal game state."""
        if not game.game_over:
            return 0.0

        if game.winner == player:
            return self.win_reward
        elif game.winner is None:  # Draw
            return self.draw_reward
        else:  # Loss
            return self.loss_penalty

    def get_reward(
        self, game: DabloGame, player: Player, previous_game_state: dict | None = None
    ) -> float:
        """
        Get total reward for current game state.

        Combines step reward and terminal reward if game is over.
        """
        reward = self.calculate_step_reward(game, player, previous_game_state)

        if game.game_over:
            reward += self.calculate_terminal_reward(game, player)

        return reward
