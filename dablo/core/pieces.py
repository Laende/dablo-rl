"""
Piece types and behavior for Dablo game

Defines piece types, values, symbols, and capture rules for the traditional
sørsamisk dablo variant.
"""

from enum import IntEnum

from pydantic import BaseModel, Field


class PieceType(IntEnum):
    """Piece types for the board with their numeric values"""

    EMPTY = 0
    P1_WARRIOR = 1
    P1_PRINCE = 2
    P1_KING = 3
    P2_WARRIOR = -1
    P2_PRINCE = -2
    P2_KING = -3


class PiecePosition(BaseModel):
    """Represents a piece and its position with validation"""

    piece_type: PieceType = Field(description="Type of piece")
    position: tuple[float, float] = Field(description="Position on board as (row, col)")


# Base piece values for AI evaluation (mapped by rank)
BASE_PIECE_VALUES: dict[int, float] = {
    0: 0.0,  # EMPTY
    1: 1.0,  # WARRIOR
    2: 3.0,  # PRINCE
    3: 10.0,  # KING
}

# Display symbols for each piece type
PIECE_SYMBOLS: dict[PieceType, str] = {
    PieceType.EMPTY: " · ",
    PieceType.P1_WARRIOR: " ▲ ",
    PieceType.P1_PRINCE: " ◇ ",
    PieceType.P1_KING: " ♔ ",
    PieceType.P2_WARRIOR: " ▼ ",
    PieceType.P2_PRINCE: " ◆ ",
    PieceType.P2_KING: " ♚ ",
}

# Human-readable names for each piece type
PIECE_NAMES: dict[PieceType, str] = {
    PieceType.P1_WARRIOR: "P1 Warrior (dåarohke)",
    PieceType.P1_PRINCE: "P1 Prince (gånkan elkie)",
    PieceType.P1_KING: "P1 King (gånka)",
    PieceType.P2_WARRIOR: "P2 Warrior (dåarohke)",
    PieceType.P2_PRINCE: "P2 Prince (gånkan elkie)",
    PieceType.P2_KING: "P2 King (gånka)",
    PieceType.EMPTY: "Empty",
}


def get_piece_symbol(piece: PieceType) -> str:
    """Get display symbol for a piece"""
    return PIECE_SYMBOLS.get(piece, " ? ")


def get_piece_name(piece: PieceType) -> str:
    """Get human-readable name for a piece with Sami terms"""
    return PIECE_NAMES.get(piece, "Unknown")


def get_piece_rank(piece: PieceType) -> int:
    """Get the rank of a piece for capture rules"""
    return abs(piece.value)


def get_piece_value(piece: PieceType) -> float:
    """Get strategic value of a piece for AI evaluation"""
    rank = abs(piece.value)
    return BASE_PIECE_VALUES.get(rank, 0.0)


def can_capture(attacker: PieceType, target: PieceType) -> bool:
    """
    Check if attacker piece can capture target piece according to sørsamisk dablo rules

    Capture hierarchy:
    - Warriors can capture warriors only
    - Princes can capture warriors and princes
    - Kings can capture any piece
    """
    # Cannot capture empty pieces
    if target == PieceType.EMPTY or attacker == PieceType.EMPTY:
        return False

    # Same player cannot capture each other
    if (attacker > 0) == (target > 0):
        return False

    # Use piece ranks for capture rules
    attacker_rank = abs(attacker.value)
    target_rank = abs(target.value)

    # A piece can capture another if its rank is >= target rank
    return attacker_rank >= target_rank
