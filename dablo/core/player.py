"""
Player identification and utilities for Dablo game

Provides enum-based player identification with cultural context for the
traditional sørsamisk dablo variant.
"""

from enum import IntEnum


class Player(IntEnum):
    """Player identification enum for sørsamisk dablo"""

    P1 = 1  # Player 1 - Aalmogh (spisse/pointed pieces)
    P2 = -1  # Player 2 - Daatjh (skarrjok/cut pieces)

    @classmethod
    def from_int(cls, value: int) -> "Player":
        """Convert integer to Player enum with validation"""
        if value > 0:
            return cls.P1
        elif value < 0:
            return cls.P2
        else:
            raise ValueError(
                f"Invalid player value: {value}. Must be positive (P1) or negative (P2)."
            )

    def opponent(self) -> "Player":
        """Get the opposing player"""
        return Player.P2 if self == Player.P1 else Player.P1

    @property
    def name(self) -> str:
        """Get human-readable player name with cultural context"""
        return "P1 (Aalmogh)" if self == Player.P1 else "P2 (Daatjh)"

    @property
    def symbol(self) -> str:
        """Get a simple symbol for the player"""
        return "▲" if self == Player.P1 else "▼"

    def is_positive(self) -> bool:
        """Check if this is the positive player (P1)"""
        return self.value > 0

    def is_negative(self) -> bool:
        """Check if this is the negative player (P2)"""
        return self.value < 0


# Player display names for UI
PLAYER_NAMES: dict[Player, str] = {
    Player.P1: "Player 1 (Aalmogh)",
    Player.P2: "Player 2 (Daatjh)",
}

# Cultural descriptions
PLAYER_DESCRIPTIONS: dict[Player, str] = {
    Player.P1: "Aalmogh - Traditional term for 'the people' with pointed (tjuppek) pieces",
    Player.P2: "Daatjh - Traditional term for 'non-people' with cut (skarrjok) pieces",
}
