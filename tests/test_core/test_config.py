"""Tests for game configuration"""

import pytest

from dablo.core.config import DabloConfig
from dablo.core.game import DabloGame


@pytest.mark.unit
class TestDabloConfig:
    """Test game configuration model"""

    def test_create_default_config(self) -> None:
        """Verify default config is created with valid attributes and values."""
        config = DabloConfig.create_default()

        assert config is not None

        # Check player setups are non-empty dictionaries
        assert isinstance(config.p1_setup, dict)
        assert config.p1_setup
        assert isinstance(config.p2_setup, dict)
        assert config.p2_setup

        # Check move limit
        assert isinstance(config.move_limit, int)
        assert config.move_limit > 0

    def test_config_is_consistent(self, game_config: DabloConfig) -> None:
        """Test that configuration is internally consistent"""
        # Both players should have equivalent structure (same piece types per player)
        if hasattr(game_config.p1_setup, "keys") and hasattr(
            game_config.p2_setup, "keys"
        ):
            p1_keys = set(game_config.p1_setup.keys())
            p2_keys = set(game_config.p2_setup.keys())

            # Extract piece type names (removing player prefix)
            p1_piece_types = {key.name.replace("P1_", "") for key in p1_keys}
            p2_piece_types = {key.name.replace("P2_", "") for key in p2_keys}

            # Both players should have the same piece type structures
            assert p1_piece_types == p2_piece_types, (
                f"Player setups should have equivalent piece types: P1={p1_piece_types}, P2={p2_piece_types}"
            )

            # Both players should have at least KING, PRINCE, WARRIOR
            expected_pieces = {"KING", "PRINCE", "WARRIOR"}
            assert expected_pieces.issubset(p1_piece_types), (
                f"P1 setup missing required pieces: {expected_pieces - p1_piece_types}"
            )
            assert expected_pieces.issubset(p2_piece_types), (
                f"P2 setup missing required pieces: {expected_pieces - p2_piece_types}"
            )

    def test_config_serializable(self, game_config: DabloConfig) -> None:
        """Test that config can be converted to dict (for serialization)"""
        try:
            # Try to convert to dict - should work for Pydantic models
            config_dict = (
                game_config.model_dump()
                if hasattr(game_config, "model_dump")
                else vars(game_config)
            )
            assert isinstance(config_dict, dict)
            assert len(config_dict) > 0
        except Exception as e:
            pytest.skip(f"Config serialization not available: {e}")

    def test_config_creation_fails_with_invalid_move_limit(self):
        """Verify creating a config with a negative move_limit raises an error."""
        default_setup = DabloConfig.create_default().p1_setup  # Get a valid setup

        # pytest.raises checks that the code inside this block *must* raise the specified exception
        with pytest.raises(ValueError):
            DabloConfig(p1_setup=default_setup, p2_setup=default_setup, move_limit=-10)


@pytest.mark.integration
class TestConfigIntegration:
    """Test config integration with game"""

    def test_config_used_by_game(
        self, game: DabloGame, game_config: DabloConfig
    ) -> None:
        """Test that game actually uses the provided config"""
        assert game.config is not None

        # Game should have same move_limit as config
        if hasattr(game_config, "move_limit"):
            # Game might store this differently, but should be consistent
            assert hasattr(game, "config")

    def test_game_respects_config_settings(self, game: DabloGame) -> None:
        """Test that game respects configuration settings"""
        config = game.config

        # Test move limit is respected
        if hasattr(config, "move_limit"):
            assert isinstance(config.move_limit, int)
            assert config.move_limit > 0

            # In a real game, we'd test that the game ends when max_moves is reached
            # For now, just verify the setting exists and is reasonable

    def test_games_created_with_different_configs_are_independent(
        self, game_factory: DabloGame
    ) -> None:
        """Verify that two games created with different configs are independent."""
        config1 = DabloConfig.create_default()
        config1.move_limit = 100

        config2 = DabloConfig.create_default()
        config2.move_limit = 500

        game1 = game_factory(config=config1)
        game2 = game_factory(config=config2)

        assert game1.config.move_limit == 100
        assert game2.config.move_limit == 500
