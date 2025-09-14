"""
Game rules and move validation for sørsamisk dablo

Implements the traditional movement rules, capture mechanics, and win conditions
as documented by Anders and Hanna Nilsson from Frostviken.
"""

from enum import Enum
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from dablo.core.game import DabloGame

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


class Move:
    """A lightweight representation of a move."""

    __slots__ = ("from_pos", "to_pos", "capture_pos")

    def __init__(
        self,
        from_pos: tuple[float, float],
        to_pos: tuple[float, float],
        capture_pos: tuple[float, float] | None = None,
    ):
        self.from_pos = from_pos
        self.to_pos = to_pos
        self.capture_pos = capture_pos

    @property
    def is_capture(self) -> bool:
        """Check if this move represents a capture."""
        return self.capture_pos is not None

    def to_dict(self) -> dict:
        """Convert move to dictionary for serialization."""
        result = {
            "from_pos": {"r": self.from_pos[0], "c": self.from_pos[1]},
            "to_pos": {"r": self.to_pos[0], "c": self.to_pos[1]},
            "is_capture": self.is_capture,
        }
        if self.capture_pos:
            result["capture_pos"] = {"r": self.capture_pos[0], "c": self.capture_pos[1]}
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "Move":
        """Create move from dictionary."""
        from_pos = (data["from_pos"]["r"], data["from_pos"]["c"])
        to_pos = (data["to_pos"]["r"], data["to_pos"]["c"])
        capture_pos = None
        if "capture_pos" in data and data["capture_pos"]:
            capture_pos = (data["capture_pos"]["r"], data["capture_pos"]["c"])
        return cls(from_pos=from_pos, to_pos=to_pos, capture_pos=capture_pos)

    def __repr__(self) -> str:
        """Provides a representation of the move."""
        return (
            f"Move(from={self.from_pos}, to={self.to_pos}, capture={self.capture_pos})"
        )

    def __eq__(self, other) -> bool:
        """Check equality between moves."""
        if not isinstance(other, Move):
            return False
        return (
            self.from_pos == other.from_pos
            and self.to_pos == other.to_pos
            and self.capture_pos == other.capture_pos
        )

    def __hash__(self) -> int:
        """Hash function for Move objects."""
        return hash((self.from_pos, self.to_pos, self.capture_pos))


class GameRules:
    """Encapsulates all Dablo game rules using tuple coordinates."""

    @staticmethod
    def check_win_condition(
        game: "DabloGame",
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
        if not game.piece_tracker.get_king_position(Player.P1):
            return Player.P2, WinReason.KING_CAPTURE
        if not game.piece_tracker.get_king_position(Player.P2):
            return Player.P1, WinReason.KING_CAPTURE

        # Get piece counts
        p1_count = len(game.p1_pieces)
        p2_count = len(game.p2_pieces)

        # 2. King vs. King Draw
        if p1_count == 1 and p2_count == 1:
            return None, WinReason.KINGS_ONLY_DRAW

        # 3. Lone King rule - opponent reduced to only their king
        if p1_count == 1 and p2_count > 1:
            return Player.P2, WinReason.LONE_KING
        if p2_count == 1 and p1_count > 1:
            return Player.P1, WinReason.LONE_KING

        # 4. Move limit draw
        if game.move_count >= game.config.move_limit:
            return None, WinReason.MOVE_LIMIT  # Draw

        # 5. Stalemate (No legal moves)
        if not game.has_any_valid_move():
            return game.current_player.opponent(), WinReason.STALEMATE

        return None, None
