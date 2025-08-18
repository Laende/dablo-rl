"""
Game rules and move validation for sørsamisk dablo

Implements the traditional movement rules, capture mechanics, and win conditions
as documented by Anders and Hanna Nilsson from Frostviken.
"""

from enum import Enum

from pydantic import BaseModel, Field, computed_field

from .pieces import PieceType, can_capture
from .player import Player


class WinReason(Enum):
    """Possible reasons for game ending"""

    KING_CAPTURE = "king_capture"
    LONE_KING = "lone_king"
    MOVE_LIMIT = "move_limit"
    STALEMATE = "stalemate"
    KINGS_ONLY_DRAW = "kings_only_draw"


WIN_DESCRIPTIONS: dict[WinReason, str] = {
    WinReason.KING_CAPTURE: "King captured",
    WinReason.LONE_KING: "Opponent reduced to lone king",
    WinReason.MOVE_LIMIT: "Move limit reached - Draw",
    WinReason.STALEMATE: "No legal moves available",
    WinReason.KINGS_ONLY_DRAW: "Both players reduced to only their kings - Draw",
}


class Move(BaseModel, frozen=True):
    """
    Represents a move in sørsamisk dablo with validation

    Moves can be either regular moves (to empty space) or captures (jumping over opponent).
    Chain captures are handled by making multiple consecutive Move objects.
    """

    from_pos: tuple[float, float] = Field(description="Starting position (row, col)")
    to_pos: tuple[float, float] = Field(description="Destination position (row, col)")
    capture_pos: tuple[float, float] | None = Field(
        default=None, description="Position of captured piece (if any)"
    )

    @computed_field
    @property
    def is_capture(self) -> bool:
        """Check if this move represents a capture"""
        return self.capture_pos is not None

    @computed_field
    @property
    def distance(self) -> float:
        """Calculate the distance of this move"""
        row_diff = self.to_pos[0] - self.from_pos[0]
        col_diff = self.to_pos[1] - self.from_pos[1]
        return (row_diff**2 + col_diff**2) ** 0.5

    def __str__(self) -> str:
        """Provides a human-readable string representation of the move"""
        if self.is_capture:
            return f"{self.from_pos} → {self.to_pos} (captures {self.capture_pos})"
        return f"{self.from_pos} → {self.to_pos}"


class MovementValidator:
    """Validates moves according to Dablo rules"""

    @staticmethod
    def is_forward_move(
        from_pos: tuple[float, float], to_pos: tuple[float, float], is_player1: bool
    ) -> bool:
        """Check if a move is forward based on player direction."""
        row_diff = to_pos[0] - from_pos[0]
        return row_diff < 0 if is_player1 else row_diff > 0

    @staticmethod
    def is_adjacent(
        pos1: tuple[float, float],
        pos2: tuple[float, float],
        adjacency_graph: dict[tuple[float, float], list[tuple[float, float]]],
    ) -> bool:
        """Check if two tuple positions are adjacent using the board graph."""
        return pos2 in adjacency_graph.get(pos1, [])

    @staticmethod
    def calculate_capture_landing(
        from_pos: tuple[float, float], capture_pos: tuple[float, float]
    ) -> tuple[float, float]:
        """Calculate the landing tuple position after a capture."""

        row_diff = capture_pos[0] - from_pos[0]
        col_diff = capture_pos[1] - from_pos[1]

        if (abs(row_diff), abs(col_diff)) in {(0.5, 1.5), (1.5, 0.5)}:
            landing_r = capture_pos[0] + (0.5 if row_diff > 0 else -0.5)
            landing_c = capture_pos[1] + (0.5 if col_diff > 0 else -0.5)
        else:
            landing_r = capture_pos[0] + row_diff
            landing_c = capture_pos[1] + col_diff

        return (landing_r, landing_c)


class GameRules:
    """Encapsulates all Dablo game rules using tuple coordinates."""

    @staticmethod
    def is_valid_regular_move(
        from_pos: tuple[float, float],
        to_pos: tuple[float, float],
        piece: PieceType,
        board_state: dict[tuple[float, float], PieceType],
        adjacency_graph: dict[tuple[float, float], list[tuple[float, float]]],
    ) -> bool:
        """Validate a regular (non-capture) move."""
        # Target must be empty
        if board_state.get(to_pos) != PieceType.EMPTY:
            return False

        # Must be an adjacent, forward move
        if not MovementValidator.is_adjacent(from_pos, to_pos, adjacency_graph):
            return False

        is_player1 = piece > 0
        if not MovementValidator.is_forward_move(from_pos, to_pos, is_player1):
            return False

        return True

    @staticmethod
    def is_valid_capture_move(
        from_pos: tuple[float, float],
        capture_pos: tuple[float, float],
        landing_pos: tuple[float, float],
        attacker: PieceType,
        target: PieceType,
        board_state: dict[tuple[float, float], PieceType],
        adjacency_graph: dict[tuple[float, float], list[tuple[float, float]]],
    ) -> bool:
        """Validate a capture move."""
        # Landing position must exist and be empty
        if board_state.get(landing_pos) != PieceType.EMPTY:
            return False

        # Must be able to capture the target piece
        if not can_capture(attacker, target):
            return False

        # Capture position must be adjacent
        if not MovementValidator.is_adjacent(from_pos, capture_pos, adjacency_graph):
            return False

        # Landing position must be arithmetically correct
        expected_landing = MovementValidator.calculate_capture_landing(
            from_pos, capture_pos
        )
        if landing_pos != expected_landing:
            return False

        return True

    @staticmethod
    def check_win_condition(
        p1_pieces_count: int,
        p2_pieces_count: int,
        p1_king_exists: bool,
        p2_king_exists: bool,
        move_count: int,
        move_limit: int,
        has_valid_moves: bool,
        current_player: Player,
    ) -> tuple[Player | None, WinReason | None]:
        """
        Check if the game has ended according to sørsamisk dablo rules

        Win conditions:
        1. King capture - Capture opponent's king
        2. Lone king - Reduce opponent to only their king
        3. Move limit - Game exceeds maximum moves (draw)
        4. Stalemate - No valid moves available

        Returns:
            Tuple of (winner, reason) or (None, None) if game continues
        """

        # 1. King capture (Highest priority)
        if not p1_king_exists:
            return Player.P2, WinReason.KING_CAPTURE
        if not p2_king_exists:
            return Player.P1, WinReason.KING_CAPTURE

        # 2. King vs. King Draw (before checking for a lone king win)
        if p1_pieces_count == 1 and p2_pieces_count == 1:
            return None, WinReason.KINGS_ONLY_DRAW  # Draw

        # 3. Lone King rule - opponent reduced to only their king
        if p1_pieces_count == 1 and p2_pieces_count > 1:
            return Player.P2, WinReason.LONE_KING
        if p2_pieces_count == 1 and p1_pieces_count > 1:
            return Player.P1, WinReason.LONE_KING

        # 4. Move limit draw
        if move_count >= move_limit:
            return None, WinReason.MOVE_LIMIT  # Draw

        # 5. Stalemate (No legal moves)
        if not has_valid_moves:
            return current_player.opponent(), WinReason.STALEMATE

        return None, None
