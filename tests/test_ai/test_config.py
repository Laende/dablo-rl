"""Tests for AI configuration models"""

from typing import Any

from pydantic import ValidationError
import pytest

from dablo.ai.config import (
    DifficultySettings,
    EvaluationWeights,
    KingSafetyConfig,
    npc_config,
)


@pytest.mark.unit
class TestDifficultySettings:
    """Test DifficultySettings model"""

    def test_difficulty_settings_creation(self) -> None:
        """Test that DifficultySettings can be created with valid values"""
        settings = DifficultySettings(
            capture_preference=1.5,
            king_safety=2.0,
            forward_progress=0.7,
            randomness=0.3,
        )

        assert settings.capture_preference == 1.5
        assert settings.king_safety == 2.0
        assert settings.forward_progress == 0.7
        assert settings.randomness == 0.3

    def test_difficulty_settings_validation(self) -> None:
        """Test DifficultySettings field validation"""
        # Test valid boundary values
        valid_settings = DifficultySettings(
            capture_preference=0.0,  # Min valid
            king_safety=3.0,  # Max valid
            forward_progress=1.0,  # Max valid
            randomness=0.0,  # Min valid
        )

        assert valid_settings.capture_preference == 0.0
        assert valid_settings.king_safety == 3.0

    def test_difficulty_settings_invalid_values(self) -> None:
        """Test that invalid values raise ValidationError"""
        # Test capture_preference out of range
        with pytest.raises(ValidationError):
            DifficultySettings(capture_preference=-0.1)

        with pytest.raises(ValidationError):
            DifficultySettings(capture_preference=2.1)

        # Test king_safety out of range
        with pytest.raises(ValidationError):
            DifficultySettings(king_safety=-0.1)

        with pytest.raises(ValidationError):
            DifficultySettings(king_safety=3.1)

        # Test forward_progress out of range
        with pytest.raises(ValidationError):
            DifficultySettings(forward_progress=-0.1)

        with pytest.raises(ValidationError):
            DifficultySettings(forward_progress=1.1)

        # Test randomness out of range
        with pytest.raises(ValidationError):
            DifficultySettings(randomness=-0.1)

        with pytest.raises(ValidationError):
            DifficultySettings(randomness=1.1)

    def test_difficulty_settings_serialization(self) -> None:
        """Test DifficultySettings serialization"""
        settings = DifficultySettings(
            capture_preference=1.2,
            king_safety=1.8,
            forward_progress=0.6,
            randomness=0.4,
        )

        # Should be serializable to dict
        settings_dict: dict[str, Any] = settings.model_dump()

        assert isinstance(settings_dict, dict)
        assert settings_dict["capture_preference"] == 1.2
        assert settings_dict["king_safety"] == 1.8
        assert settings_dict["forward_progress"] == 0.6
        assert settings_dict["randomness"] == 0.4

        # Should be deserializable from dict
        reconstructed = DifficultySettings(**settings_dict)
        assert reconstructed == settings


@pytest.mark.unit
class TestEvaluationWeights:
    """Test EvaluationWeights model"""

    def test_evaluation_weights_defaults(self) -> None:
        """Test EvaluationWeights with default values"""
        weights = EvaluationWeights()

        # Test default values exist and are reasonable
        assert weights.chain_capture_bonus >= 0.0
        assert weights.center_control >= 0.0
        assert weights.piece_protection >= 0.0
        assert weights.threat_creation >= 0.0
        assert weights.max_threat_value >= 0.0

    def test_evaluation_weights_custom_values(self) -> None:
        """Test EvaluationWeights with custom values"""
        weights = EvaluationWeights(
            chain_capture_bonus=10.0,
            center_control=0.5,
            piece_protection=0.6,
            threat_creation=0.8,
            max_threat_value=3.0,
        )

        assert weights.chain_capture_bonus == 10.0
        assert weights.center_control == 0.5
        assert weights.piece_protection == 0.6
        assert weights.threat_creation == 0.8
        assert weights.max_threat_value == 3.0

    def test_evaluation_weights_validation(self) -> None:
        """Test EvaluationWeights validation constraints"""
        # All values should be non-negative
        with pytest.raises(ValidationError):
            EvaluationWeights(chain_capture_bonus=-1.0)

        with pytest.raises(ValidationError):
            EvaluationWeights(center_control=-0.1)

        with pytest.raises(ValidationError):
            EvaluationWeights(piece_protection=-0.1)

        with pytest.raises(ValidationError):
            EvaluationWeights(threat_creation=-0.1)

        with pytest.raises(ValidationError):
            EvaluationWeights(max_threat_value=-0.1)


@pytest.mark.unit
class TestKingSafetyConfig:
    """Test KingSafetyConfig model"""

    def test_king_safety_config_creation(self) -> None:
        """Test KingSafetyConfig creation"""
        config = KingSafetyConfig(
            capture_danger_penalty=-3.0, immediate_threat_penalty=-25.0
        )

        assert config.capture_danger_penalty == -3.0
        assert config.immediate_threat_penalty == -25.0

    def test_king_safety_config_penalty_validation(self) -> None:
        """Test that penalties must be non-positive"""
        # Valid negative values
        config = KingSafetyConfig(
            capture_danger_penalty=-10.0, immediate_threat_penalty=-100.0
        )

        assert config.capture_danger_penalty <= 0.0
        assert config.immediate_threat_penalty <= 0.0

        # Invalid positive values should raise ValidationError
        with pytest.raises(ValidationError):
            KingSafetyConfig(capture_danger_penalty=1.0)

        with pytest.raises(ValidationError):
            KingSafetyConfig(immediate_threat_penalty=1.0)


@pytest.mark.integration
class TestNPCConfigIntegration:
    """Test global NPC configuration integration"""

    def test_npc_config_exists(self) -> None:
        """Test that global npc_config exists and is accessible"""
        assert npc_config is not None

        # Should have methods for getting difficulty settings
        if hasattr(npc_config, "get_difficulty_settings"):
            assert callable(npc_config.get_difficulty_settings)

    def test_npc_config_difficulty_settings(self) -> None:
        """Test that npc_config provides difficulty settings"""
        if hasattr(npc_config, "get_difficulty_settings"):
            difficulties = ["easy", "medium", "hard", "expert"]

            for difficulty in difficulties:
                settings = npc_config.get_difficulty_settings(difficulty)

                # Should return some kind of settings object
                assert settings is not None

                # Should have expected attributes for difficulty settings
                if hasattr(settings, "randomness"):
                    assert isinstance(settings.randomness, (int, float))
                    assert 0.0 <= settings.randomness <= 1.0
