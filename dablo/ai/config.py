"""
AI configuration models for Dablo NPCs using Pydantic

Provides type-safe configuration for NPC behavior, evaluation weights, and game parameters.
"""

from pydantic import BaseModel, Field


class DifficultySettings(BaseModel):
    """Configuration for AI difficulty levels"""

    capture_preference: float = Field(
        ge=0.0, le=2.0, description="Weight for capture moves"
    )
    king_safety: float = Field(ge=0.0, le=3.0, description="Weight for king protection")
    forward_progress: float = Field(
        ge=0.0, le=1.0, description="Weight for advancing pieces"
    )
    randomness: float = Field(ge=0.0, le=1.0, description="Probability of random moves")


class EvaluationWeights(BaseModel):
    """Weights for move evaluation heuristics"""

    chain_capture_bonus: float = Field(
        default=5.0, ge=0.0, description="Bonus for chain captures"
    )
    center_control: float = Field(
        default=0.3, ge=0.0, description="Weight for center control"
    )
    piece_protection: float = Field(
        default=0.4, ge=0.0, description="Weight for piece protection"
    )
    threat_creation: float = Field(
        default=0.5, ge=0.0, description="Weight for creating threats"
    )
    max_threat_value: float = Field(
        default=2.0, ge=0.0, description="Maximum threat evaluation score"
    )


class KingSafetyConfig(BaseModel):
    """Configuration for king safety evaluation"""

    capture_danger_penalty: float = Field(
        default=-5.0, le=0.0, description="Penalty for king in capture danger"
    )
    immediate_threat_penalty: float = Field(
        default=-50.0, le=0.0, description="Penalty for missing king"
    )
    close_distance_penalty: float = Field(
        default=-4.0, le=0.0, description="Penalty for close enemy pieces"
    )
    medium_distance_penalty: float = Field(
        default=-2.0, le=0.0, description="Penalty for medium distance threats"
    )
    safe_distance_bonus: float = Field(
        default=0.2, ge=0.0, description="Bonus for safe positioning"
    )
    very_safe_bonus: float = Field(
        default=0.5, ge=0.0, description="Bonus for very safe positioning"
    )

    # Distance thresholds
    immediate_danger_threshold: float = Field(
        default=1.1, gt=0.0, description="Immediate danger distance"
    )
    close_danger_threshold: float = Field(
        default=1.6, gt=0.0, description="Close danger distance"
    )
    medium_safety_threshold: float = Field(
        default=2.5, gt=0.0, description="Medium safety distance"
    )


class CenterControlConfig(BaseModel):
    """Configuration for center control evaluation"""

    center_rows: set[float] = Field(
        default={2.0, 2.5, 3.0}, description="Center row positions"
    )
    center_cols: set[float] = Field(
        default={1.5, 2.0, 2.5}, description="Center column positions"
    )
    row_bonus: float = Field(
        default=0.3, ge=0.0, description="Bonus for center row control"
    )
    col_bonus: float = Field(
        default=0.3, ge=0.0, description="Bonus for center column control"
    )


class MoveSelectionConfig(BaseModel):
    """Configuration for move selection logic"""

    top_moves_count: int = Field(
        default=3, ge=1, le=10, description="Number of top moves to consider"
    )
    selection_weights: list[int] = Field(
        default=[3, 2, 1], description="Weights for top move selection"
    )
    forward_progress_multiplier: float = Field(
        default=0.5, ge=0.0, description="Multiplier for forward progress"
    )
    min_piece_value_for_protection: float = Field(
        default=2.0, ge=0.0, description="Minimum piece value to evaluate protection"
    )


class PerformanceTestConfig(BaseModel):
    """Configuration for NPC performance testing"""

    default_move_limit: int = Field(
        default=150, ge=50, le=1000, description="Default move limit for test games"
    )
    default_game_count: int = Field(
        default=67, ge=1, le=1000, description="Default number of test games"
    )
    test_matchups: list[tuple[str, str, str, str]] = Field(
        default=[
            ("smart", "random", "hard", "medium"),
            ("aggressive", "defensive", "hard", "medium"),
            ("smart", "aggressive", "hard", "hard"),
            ("smart", "defensive", "hard", "medium"),
            # Additional cross-type scenarios
            ("aggressive", "random", "hard", "medium"),
            ("defensive", "random", "medium", "medium"),
            ("aggressive", "smart", "hard", "hard"),
            ("defensive", "smart", "medium", "hard"),
            # Difficulty variation tests
            ("smart", "smart", "easy", "hard"),
            ("aggressive", "aggressive", "medium", "hard"),
            ("defensive", "defensive", "easy", "medium"),
            # Style mirror matches
            ("random", "random", "medium", "medium"),
            ("smart", "smart", "hard", "hard"),
            ("aggressive", "aggressive", "medium", "medium"),
            ("defensive", "defensive", "easy", "easy"),
        ],
        description="Default test matchups (npc1_type, npc2_type, npc1_diff, npc2_diff)",
    )


class NPCConfig(BaseModel):
    """Main NPC configuration containing all settings"""

    # Difficulty presets - tuned for better balance
    easy: DifficultySettings = DifficultySettings(
        capture_preference=0.6, king_safety=0.2, forward_progress=0.3, randomness=0.4
    )

    medium: DifficultySettings = DifficultySettings(
        capture_preference=0.8, king_safety=0.5, forward_progress=0.5, randomness=0.25
    )

    hard: DifficultySettings = DifficultySettings(
        capture_preference=1.0, king_safety=0.7, forward_progress=0.8, randomness=0.08
    )

    # Specialized styles - rebalanced based on performance data
    aggressive: DifficultySettings = DifficultySettings(
        capture_preference=1.5, king_safety=0.3, forward_progress=1.0, randomness=0.03
    )

    defensive: DifficultySettings = DifficultySettings(
        capture_preference=0.3, king_safety=1.5, forward_progress=0.2, randomness=0.18
    )

    # Evaluation configuration - tuned for more decisive gameplay
    evaluation: EvaluationWeights = EvaluationWeights(
        chain_capture_bonus=6.0,  # Increased to encourage multi-captures
        center_control=0.2,  # Reduced to focus more on captures
        piece_protection=0.3,  # Slightly reduced to be less conservative
        threat_creation=0.7,  # Increased to encourage aggressive positioning
        max_threat_value=3.0,  # Increased cap for stronger threat evaluation
    )
    king_safety: KingSafetyConfig = KingSafetyConfig()
    center_control: CenterControlConfig = CenterControlConfig()
    move_selection: MoveSelectionConfig = MoveSelectionConfig()
    performance_test: PerformanceTestConfig = PerformanceTestConfig()

    def get_difficulty_settings(self, difficulty: str) -> DifficultySettings:
        """Get difficulty settings by name"""
        difficulty_map = {
            "easy": self.easy,
            "medium": self.medium,
            "hard": self.hard,
            "aggressive": self.aggressive,
            "defensive": self.defensive,
        }
        return difficulty_map.get(difficulty, self.medium)


# Global configuration instance
npc_config = NPCConfig()
