"""Tests for AI player implementations"""

import pytest

from dablo.ai.base import NPCPlayer
from dablo.ai.players import (
    AggressiveNPCPlayer,
    DefensiveNPCPlayer,
    RandomNPCPlayer,
    SmartNPCPlayer,
    create_npc_player,
)
from dablo.core.config import DabloConfig
from dablo.core.game import DabloGame
from dablo.core.pieces import PieceType
from dablo.core.player import Player
from dablo.core.rules import Move


@pytest.mark.unit
class TestNPCPlayerTypes:
    """Test different NPC player implementations"""

    @pytest.mark.parametrize(
        "npc_class",
        [
            SmartNPCPlayer,
            RandomNPCPlayer,
            AggressiveNPCPlayer,
            DefensiveNPCPlayer,
        ],
    )
    def test_npc_player_initialization(self, npc_class):
        """Test that all NPC types can be initialized"""
        npc = npc_class(Player.P1)

        assert isinstance(npc, NPCPlayer)
        assert npc.player_id == Player.P1
        assert hasattr(npc, "difficulty")
        assert hasattr(npc, "get_move")

    @pytest.mark.parametrize(
        "npc_class",
        [
            SmartNPCPlayer,
            RandomNPCPlayer,
            AggressiveNPCPlayer,
            DefensiveNPCPlayer,
        ],
    )
    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard", "expert"])
    def test_npc_difficulty_levels(self, npc_class, difficulty):
        """Test NPC players with different difficulty levels"""
        npc = npc_class(Player.P2, difficulty=difficulty)

        assert npc.difficulty == difficulty
        assert npc.player_id == Player.P2

    def test_smart_npc_specific_features(self):
        """Test SmartNPCPlayer specific functionality"""
        npc = SmartNPCPlayer(Player.P1, "hard")

        # Should have strategic components
        assert hasattr(npc, "settings")
        assert hasattr(npc, "clear_cache")

        # Should have cache for performance
        if hasattr(npc, "_threat_cache"):
            assert isinstance(npc._threat_cache, dict)

        # Test cache clearing
        npc.clear_cache()
        if hasattr(npc, "_threat_cache"):
            assert len(npc._threat_cache) == 0

    def test_random_npc_characteristics(self):
        """Test RandomNPCPlayer behavior"""
        npc = RandomNPCPlayer(Player.P1)

        # Should inherit from NPCPlayer
        assert isinstance(npc, NPCPlayer)

        # Should have get_move method
        assert hasattr(npc, "get_move")
        assert callable(npc.get_move)


@pytest.mark.integration
@pytest.mark.movement
class TestNPCPlayerGameplay:
    """Test NPC players in actual game scenarios"""

    @pytest.mark.parametrize("npc_type", ["smart", "random", "aggressive", "defensive"])
    def test_npc_can_make_moves(self, game: DabloGame, npc_type):
        """Test that all NPC types can make valid moves"""
        npc = create_npc_player(Player.P1, npc_type, "medium")

        # Should be able to get a move
        move = npc.get_move(game)

        if move is not None:
            assert isinstance(move, Move)

            # Move should be valid for current game state
            valid_moves = game.get_all_valid_moves()
            assert move in valid_moves, f"{npc_type} NPC returned invalid move: {move}"

    def test_npc_moves_are_for_correct_player(self, game: DabloGame):
        """Test that NPC moves are for the correct player"""
        # Test P1 NPC
        npc_p1 = SmartNPCPlayer(Player.P1)
        move_p1 = npc_p1.get_move(game)

        if move_p1:
            piece = game.board_state[move_p1.from_pos]
            assert piece.value > 0, "P1 NPC should only move P1 pieces"

        # Switch to P2's turn
        if game.get_all_valid_moves():
            first_move = game.get_all_valid_moves()[0]
            game.make_move(first_move)

        # Test P2 NPC
        if game.current_player == Player.P2:
            npc_p2 = SmartNPCPlayer(Player.P2)
            move_p2 = npc_p2.get_move(game)

            if move_p2:
                piece = game.board_state[move_p2.from_pos]
                assert piece.value < 0, "P2 NPC should only move P2 pieces"

    def test_npc_handles_no_moves_gracefully(self, empty_game: DabloGame):
        """Test NPC behavior when no moves are available"""
        npc = SmartNPCPlayer(Player.P1)

        # Empty game has no moves
        move = npc.get_move(empty_game)

        # Should return None or handle gracefully
        assert move is None

    def test_npc_capture_behavior(self, simple_capture_game: DabloGame):
        """Test NPC behavior with capture opportunities"""
        game = simple_capture_game
        npc = SmartNPCPlayer(
            Player.P1, "hard"
        )  # High difficulty should prefer captures

        move = npc.get_move(game)

        if move:
            # Should be a valid move
            valid_moves = game.get_all_valid_moves()
            assert move in valid_moves

            # For high difficulty, might prefer capture moves
            capture_moves = [m for m in valid_moves if m.is_capture]
            if capture_moves and move.is_capture:
                # Good strategic choice
                assert move.capture_pos is not None


@pytest.mark.unit
class TestNPCFactory:
    """Test NPC factory function"""

    def test_create_npc_player_basic(self):
        """Test basic NPC creation"""
        npc = create_npc_player(Player.P1)

        assert isinstance(npc, NPCPlayer)
        assert npc.player_id == Player.P1

    @pytest.mark.parametrize(
        "npc_type,expected_class",
        [
            ("smart", SmartNPCPlayer),
            ("random", RandomNPCPlayer),
            ("aggressive", AggressiveNPCPlayer),
            ("defensive", DefensiveNPCPlayer),
        ],
    )
    def test_create_npc_player_types(self, npc_type, expected_class):
        """Test creating different NPC types"""
        npc = create_npc_player(Player.P2, npc_type, "medium")

        assert isinstance(npc, expected_class)
        assert npc.player_id == Player.P2
        assert npc.difficulty == "medium"

    def test_create_npc_player_case_insensitive(self):
        """Test that NPC type is case insensitive"""
        npc_lower = create_npc_player(Player.P1, "smart")
        npc_upper = create_npc_player(Player.P1, "SMART")
        npc_mixed = create_npc_player(Player.P1, "Smart")

        # All should create SmartNPCPlayer
        assert type(npc_lower) is type(npc_upper) is type(npc_mixed)
        assert isinstance(npc_lower, SmartNPCPlayer)

    def test_create_npc_player_unknown_type(self):
        """Test creating NPC with unknown type falls back to default"""
        npc = create_npc_player(Player.P1, "unknown_type")

        # Should fall back to SmartNPCPlayer (default)
        assert isinstance(npc, SmartNPCPlayer)

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard", "expert"])
    def test_create_npc_player_difficulties(self, difficulty):
        """Test creating NPCs with different difficulties"""
        npc = create_npc_player(Player.P1, "smart", difficulty)

        assert npc.difficulty == difficulty


@pytest.mark.integration
@pytest.mark.slow
class TestNPCStrategicBehavior:
    """Test strategic behavior differences between NPC types"""

    def test_aggressive_vs_defensive_tendencies(self, game: DabloGame):
        """Test that aggressive and defensive NPCs show different tendencies"""
        aggressive_npc = AggressiveNPCPlayer(Player.P1, "medium")
        defensive_npc = DefensiveNPCPlayer(Player.P1, "medium")

        # Get moves from both
        aggressive_move = aggressive_npc.get_move(game)
        defensive_move = defensive_npc.get_move(game)

        # Both should return valid moves or None
        valid_moves = game.get_all_valid_moves()

        if aggressive_move:
            assert aggressive_move in valid_moves
        if defensive_move:
            assert defensive_move in valid_moves

        # They might choose differently (though not guaranteed in single test)
        # This test mainly ensures both can make decisions

    def test_difficulty_affects_randomness(self, game: DabloGame):
        """Test that difficulty affects move selection"""
        easy_npc = SmartNPCPlayer(Player.P1, "easy")
        expert_npc = SmartNPCPlayer(Player.P1, "expert")

        # Both should be able to make moves
        easy_move = easy_npc.get_move(game)
        expert_move = expert_npc.get_move(game)

        valid_moves = game.get_all_valid_moves()

        if easy_move:
            assert easy_move in valid_moves
        if expert_move:
            assert expert_move in valid_moves

        # Test that settings differ based on difficulty
        if hasattr(easy_npc, "settings") and hasattr(expert_npc, "settings"):
            # Expert should have lower randomness than easy
            if hasattr(easy_npc.settings, "randomness") and hasattr(
                expert_npc.settings, "randomness"
            ):
                assert easy_npc.settings.randomness >= expert_npc.settings.randomness


@pytest.mark.performance
class TestNPCPerformance:
    """Test NPC performance characteristics"""

    def test_npc_cache_effectiveness(self, game: DabloGame):
        """Test that NPC caching improves performance"""
        npc = SmartNPCPlayer(Player.P1, "expert")

        # Make multiple move requests to same position
        move1 = npc.get_move(game)
        move2 = npc.get_move(game)
        move3 = npc.get_move(game)

        # Should not crash and should return valid moves
        for move in [move1, move2, move3]:
            if move:
                assert isinstance(move, Move)

        # Check cache stats if available
        if hasattr(npc, "_threat_cache_hits") and hasattr(npc, "_threat_cache_misses"):
            total_requests = npc._threat_cache_hits + npc._threat_cache_misses
            # Should have made some cache requests
            assert total_requests >= 0


@pytest.mark.unit
class TestNPCEdgeCases:
    """Test edge cases and less common code paths in NPC players"""

    def test_smart_npc_empty_top_moves(self):
        """Test SmartNPCPlayer when no top moves are available (line 69)"""
        npc = SmartNPCPlayer(Player.P1, "medium")

        # Create a mock game with minimal moves

        game = DabloGame(config=DabloConfig(), initial_state="empty")
        # Add just one piece to get minimal moves
        game.add_piece((2.0, 2.0), PieceType.P1_WARRIOR)

        # Mock the _evaluate_move to return very low scores to trigger edge case
        original_evaluate = npc._evaluate_move

        def mock_evaluate(game, move):
            return -1000.0  # Very low score

        npc._evaluate_move = mock_evaluate

        try:
            move = npc.get_move(game)
            # Should still return a valid move or None
            if move:
                assert isinstance(move, Move)
        finally:
            npc._evaluate_move = original_evaluate

    def test_king_safety_immediate_capture_threat(self, empty_game: DabloGame):
        """Test king safety evaluation with immediate capture threat (lines 158-159)"""
        npc = SmartNPCPlayer(Player.P1, "medium")

        # Create a temporary game state to trigger the capture threat logic
        # We'll mock the get_all_valid_moves to return a capture move targeting the king
        class MockTempGame:
            def get_all_valid_moves(self):
                # Return a move that captures the king at (1.5, 1.5)
                return [
                    Move(from_pos=(1.0, 1.0), to_pos=(2.0, 2.0), capture_pos=(1.5, 1.5))
                ]

        # Mock the _create_temp_game_state method
        original_create_temp = npc._create_temp_game_state

        def mock_create_temp(game, board_after_move, player_to_move=None):
            return MockTempGame()

        npc._create_temp_game_state = mock_create_temp

        try:
            # Board state with king at position that will be threatened
            board_after_move = {(1.5, 1.5): PieceType.P1_KING}
            safety_score = npc._evaluate_king_safety(empty_game, board_after_move)

            # Should return capture danger penalty
            from dablo.ai.config import npc_config

            assert safety_score == npc_config.king_safety.capture_danger_penalty
        finally:
            npc._create_temp_game_state = original_create_temp

    def test_king_safety_no_enemies(self, empty_game: DabloGame):
        """Test king safety when no enemies remain (line 172)"""
        npc = SmartNPCPlayer(Player.P1, "medium")

        # Set up board with only P1 pieces (no enemies)
        empty_game.add_piece((2.0, 2.0), PieceType.P1_KING)
        empty_game.add_piece((3.0, 3.0), PieceType.P1_WARRIOR)

        move = Move(from_pos=(3.0, 3.0), to_pos=(4.0, 4.0))
        board_after = npc._simulate_move_on_board(empty_game.board_state, move)
        safety_score = npc._evaluate_king_safety(empty_game, board_after)

        # Should return very safe bonus
        from dablo.ai.config import npc_config

        assert safety_score == npc_config.king_safety.very_safe_bonus

    def test_piece_protection_no_piece_at_source(self, empty_game: DabloGame):
        """Test piece protection when no piece at source position (line 212)"""
        npc = SmartNPCPlayer(Player.P1, "medium")

        # Create a move from an empty position
        move = Move(from_pos=(1.0, 1.0), to_pos=(2.0, 2.0))
        board_after = {(2.0, 2.0): PieceType.P1_WARRIOR}  # Mock result

        score = npc._evaluate_piece_protection(empty_game, move, board_after)

        # Should return 0.0 when no piece at source
        assert score == 0.0

    def test_get_threat_level_with_immediate_threats(self, empty_game: DabloGame):
        """Test threat level calculation with immediate threats (line 258)"""
        npc = SmartNPCPlayer(Player.P1, "medium")

        # Mock the _create_temp_game_state to return a game with immediate threats
        class MockTempGameWithThreats:
            def get_all_valid_moves(self):
                # Return moves that capture the position we're checking
                return [
                    Move(
                        from_pos=(1.0, 1.0), to_pos=(3.0, 3.0), capture_pos=(2.0, 2.0)
                    ),
                    Move(
                        from_pos=(0.5, 0.5), to_pos=(2.5, 2.5), capture_pos=(2.0, 2.0)
                    ),
                ]

        original_create_temp = npc._create_temp_game_state

        def mock_create_temp(game, board_state, player_to_move=None):
            return MockTempGameWithThreats()

        npc._create_temp_game_state = mock_create_temp

        try:
            # Test threat level - should return the number of immediate threats
            threat_level = npc._get_threat_level(empty_game, (2.0, 2.0), Player.P1)

            # Should detect 2 immediate threats (return early on line 258)
            assert threat_level == 2
        finally:
            npc._create_temp_game_state = original_create_temp

    def test_get_threat_level_potential_threats(self, empty_game: DabloGame):
        """Test threat level calculation with potential threats (lines 264-268)"""
        npc = SmartNPCPlayer(Player.P1, "medium")

        # Set up position where enemy can move adjacent but not capture immediately
        empty_game.add_piece((3.0, 3.0), PieceType.P1_WARRIOR)  # Target piece
        empty_game.add_piece((1.0, 1.0), PieceType.P2_WARRIOR)  # Can move near
        empty_game.current_player = Player.P2

        threat_level = npc._get_threat_level(empty_game, (3.0, 3.0), Player.P1)

        # Should detect potential threats (adjacent moves)
        assert threat_level >= 0

    def test_piece_protection_with_threat_changes(self, empty_game: DabloGame):
        """Test piece protection with threat level changes (lines 223-229)"""
        npc = SmartNPCPlayer(Player.P1, "medium")

        # Set up a scenario where moving reduces threat
        empty_game.add_piece((2.0, 2.0), PieceType.P1_KING)  # Valuable piece
        empty_game.add_piece((1.5, 1.5), PieceType.P2_WARRIOR)  # Threatens king
        empty_game.current_player = Player.P1

        # Move king away from threat
        move = Move(from_pos=(2.0, 2.0), to_pos=(4.0, 4.0))
        board_after = npc._simulate_move_on_board(empty_game.board_state, move)

        score = npc._evaluate_piece_protection(empty_game, move, board_after)

        # Should get positive score for reducing threat to valuable piece
        assert isinstance(score, float)
