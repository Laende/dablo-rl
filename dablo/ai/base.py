"""
Base NPC player class for Dablo game

Provides the foundation for AI opponents with strategic evaluation capabilities.
All NPC implementations inherit from this base class.
"""

from ..core.game import DabloGame
from ..core.player import Player
from ..core.rules import Move


class NPCPlayer:
    """Base class for NPC AI players"""

    def __init__(self, player_id: Player, difficulty: str = "medium"):
        if player_id is None:
            raise ValueError("player_id cannot be None")

        if not isinstance(player_id, Player):
            raise TypeError(f"player_id must be a Player enum, got {type(player_id)}")

        if difficulty is None:
            difficulty = "medium"

        if not isinstance(difficulty, str):
            raise TypeError(f"difficulty must be a string, got {type(difficulty)}")

        self.player_id = player_id
        self.difficulty = difficulty

    def get_move(self, game: DabloGame) -> Move | None:
        """Get the best move for current game state"""
        raise NotImplementedError
