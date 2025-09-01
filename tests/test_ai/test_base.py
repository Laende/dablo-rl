"""Tests for AI base NPC player class"""

import pytest

from dablo.ai.base import NPCPlayer
from dablo.core.player import Player
from dablo.core.rules import Move


@pytest.mark.unit
class TestNPCPlayer:
    """Test base NPCPlayer class"""

    def test_npc_player_initialization(self):
        """Test NPCPlayer can be initialized correctly"""
        npc = NPCPlayer(Player.P1)

        assert npc.player_id == Player.P1
        assert npc.difficulty == "medium"  # Default difficulty

    def test_npc_player_custom_difficulty(self):
        """Test NPCPlayer with custom difficulty"""
        difficulties = ["easy", "medium", "hard", "expert"]

        for difficulty in difficulties:
            npc = NPCPlayer(Player.P2, difficulty=difficulty)
            assert npc.player_id == Player.P2
            assert npc.difficulty == difficulty

    def test_npc_player_both_players(self):
        """Test NPCPlayer works with both player types"""
        npc_p1 = NPCPlayer(Player.P1)
        npc_p2 = NPCPlayer(Player.P2)

        assert npc_p1.player_id == Player.P1
        assert npc_p2.player_id == Player.P2
        assert npc_p1.player_id != npc_p2.player_id

    def test_npc_get_move_not_implemented(self, game):
        """Test that base NPCPlayer raises NotImplementedError for get_move"""
        npc = NPCPlayer(Player.P1)

        with pytest.raises(NotImplementedError):
            npc.get_move(game)

    def test_npc_player_invalid_difficulty_type(self):
        """Test NPCPlayer handles invalid difficulty"""
        with pytest.raises(TypeError):
            NPCPlayer(Player.P1, 123)


@pytest.mark.integration
class TestNPCPlayerIntegration:
    """Integration tests for NPCPlayer with game"""

    def test_npc_player_type_compatibility(self):
        """Test that NPCPlayer is compatible with game expectations"""
        npc = NPCPlayer(Player.P1)

        # Should have correct player_id type
        assert isinstance(npc.player_id, Player)
        assert npc.player_id in [Player.P1, Player.P2]

        # Should have get_move method (even if not implemented)
        assert hasattr(npc, "get_move")
        assert callable(npc.get_move)

    def test_npc_player_as_base_class(self):
        """Test NPCPlayer as a base class for inheritance"""

        class TestNPC(NPCPlayer):
            def get_move(self, game):
                # Simple implementation for testing
                moves = game.get_all_valid_moves()
                return moves[0] if moves else None

        npc = TestNPC(Player.P1, "hard")

        # Should inherit base properties
        assert npc.player_id == Player.P1
        assert npc.difficulty == "hard"

        # Should have working get_move
        assert hasattr(npc, "get_move")
        assert callable(npc.get_move)

    def test_npc_inheritance_pattern(self, game):
        """Test that NPCPlayer follows proper inheritance pattern"""

        class ConcreteNPC(NPCPlayer):
            def __init__(self, player_id: Player, difficulty: str = "medium"):
                super().__init__(player_id, difficulty)
                self.move_count = 0

            def get_move(self, game):
                self.move_count += 1
                moves = game.get_all_valid_moves()
                return moves[0] if moves else None

        npc = ConcreteNPC(Player.P2, "easy")

        # Test inheritance works
        assert isinstance(npc, NPCPlayer)
        assert npc.player_id == Player.P2
        assert npc.difficulty == "easy"
        assert npc.move_count == 0

        # Test concrete implementation works
        move = npc.get_move(game)
        assert npc.move_count == 1

        if move:
            assert isinstance(move, Move)


@pytest.mark.unit
class TestNPCPlayerEdgeCases:
    """Test edge cases and error handling for NPCPlayer"""

    def test_npc_player_with_none_player_id(self):
        """Test NPCPlayer handles None player_id"""
        with pytest.raises((TypeError, AttributeError, ValueError)):
            NPCPlayer(None)

    def test_npc_player_with_invalid_player_type(self):
        """Test NPCPlayer with invalid player type"""
        with pytest.raises((TypeError, AttributeError)):
            NPCPlayer("invalid_player")

    def test_npc_player_difficulty_types(self):
        """Test NPCPlayer handles different difficulty input types"""
        # Should work with strings
        npc_str = NPCPlayer(Player.P1, "hard")
        assert npc_str.difficulty == "hard"

        # Test with None (should use default or handle gracefully)
        npc_none = NPCPlayer(Player.P1, None)
        # Behavior depends on implementation - could be None or default
        assert npc_none.difficulty is None or isinstance(npc_none.difficulty, str)

    def test_npc_player_string_representation(self):
        """Test string representation of NPCPlayer"""
        npc = NPCPlayer(Player.P1, "expert")

        # Should be convertible to string
        npc_str = str(npc)
        assert isinstance(npc_str, str)

        # Should contain relevant information
        assert "NPC" in npc_str or "Player" in npc_str
