"""
Configuration settings for Dablo game using Pydantic models.

Contains all game rules, board setup, and game parameters with validation.
"""

from pydantic import BaseModel, Field, computed_field

from .pieces import PieceType


class DabloConfig(BaseModel):
    """Configuration for core game mechanics and rules with validation."""

    board_rows: int = Field(
        default=6, ge=3, le=10, description="Number of rows on the game board"
    )

    board_cols: int = Field(
        default=5, ge=3, le=10, description="Number of columns on the game board"
    )

    move_limit: int = Field(
        default=500, ge=50, le=2000, description="Maximum number of moves before draw"
    )

    p1_setup: dict[PieceType, list[tuple[float, float]]] = Field(
        default_factory=lambda: {
            PieceType.P1_KING: [(3.0, 4.0)],
            PieceType.P1_PRINCE: [(3.5, 3.5)],
            PieceType.P1_WARRIOR: [(5.0, c) for c in range(5)]
            + [(4.5, c + 0.5) for c in range(4)]
            + [(4.0, c) for c in range(5)],
        },
        description="Initial setup positions for Player 1 pieces",
    )

    p2_setup: dict[PieceType, list[tuple[float, float]]] = Field(
        default_factory=lambda: {
            PieceType.P2_KING: [(2.0, 0.0)],
            PieceType.P2_PRINCE: [(1.5, 0.5)],
            PieceType.P2_WARRIOR: [(0.0, c) for c in range(5)]
            + [(0.5, c + 0.5) for c in range(4)]
            + [(1.0, c) for c in range(5)],
        },
        description="Initial setup positions for Player 2 pieces",
    )

    @computed_field
    @property
    def board_size(self) -> tuple[int, int]:
        """Board size as (rows, cols) tuple for backward compatibility."""
        return (self.board_rows, self.board_cols)

    @classmethod
    def create_default(cls) -> "DabloConfig":
        """Create a default game configuration."""
        return cls()

    @classmethod
    def create_quick_game(cls) -> "DabloConfig":
        """Create configuration for quicker games."""
        return cls(move_limit=200)

    @classmethod
    def create_test_config(cls) -> "DabloConfig":
        """Create configuration optimized for testing."""
        return cls(move_limit=100)

    @classmethod
    def create_custom(
        cls, board_rows: int = 6, board_cols: int = 5, move_limit: int = 500
    ) -> "DabloConfig":
        """Create a custom configuration with specified parameters."""
        return cls(board_rows=board_rows, board_cols=board_cols, move_limit=move_limit)
