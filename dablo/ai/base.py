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
        self.player_id = player_id
        self.difficulty = difficulty

    def get_move(self, game: DabloGame) -> Move | None:
        """Get the best move for current game state"""
        raise NotImplementedError
