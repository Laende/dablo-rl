"""
Configuration classes for Dablo RL training.

Provides structured configuration management for all training parameters,
reward settings, and environment options.
"""

import json
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field
import yaml

from ..core.config import DabloConfig


class RewardConfig(BaseModel):
    """Configuration for reward calculation parameters."""

    material_weight: float = Field(
        default=1.0, ge=0.0, description="Weight for material advantage reward"
    )
    positional_weight: float = Field(
        default=0.1, ge=0.0, description="Weight for positional bonus reward"
    )
    capture_bonus: float = Field(
        default=5.0, ge=0.0, description="Extra bonus reward for capturing pieces"
    )
    win_reward: float = Field(default=100.0, description="Reward for winning the game")
    loss_penalty: float = Field(
        default=-100.0, description="Penalty for losing the game"
    )
    draw_reward: float = Field(default=0.0, description="Reward for draw games")
    step_penalty: float = Field(
        default=-0.01, description="Small penalty per step to encourage decisive play"
    )


class TrainingConfig(BaseModel):
    """Configuration for MaskablePPO training parameters."""

    total_timesteps: int = Field(
        default=100000, ge=1000, description="Total training timesteps"
    )
    learning_rate: float = Field(
        default=3e-4, gt=0.0, description="Learning rate for PPO optimizer"
    )
    n_steps: int = Field(
        default=2048, ge=1, description="Number of steps per environment per update"
    )
    batch_size: int = Field(default=64, ge=1, description="Batch size for training")
    n_epochs: int = Field(default=10, ge=1, description="Number of epochs per update")
    n_envs: int = Field(default=4, ge=1, description="Number of parallel environments")
    policy: str = Field(
        default="MultiInputPolicy", description="Policy type for MaskablePPO"
    )


class EvaluationConfig(BaseModel):
    """Configuration for model evaluation parameters."""

    eval_freq: int = Field(
        default=10000,
        ge=1000,
        description="Evaluation frequency during training (in timesteps)",
    )
    n_eval_episodes: int = Field(
        default=10,
        ge=1,
        description="Number of episodes for evaluation during training",
    )
    final_eval: bool = Field(
        default=True, description="Run final evaluation after training"
    )
    final_eval_episodes: int = Field(
        default=50, ge=1, description="Number of episodes for final evaluation"
    )
    deterministic: bool = Field(
        default=True, description="Use deterministic policy for evaluation"
    )


class CheckpointConfig(BaseModel):
    """Configuration for model saving and checkpointing."""

    save_freq: int = Field(
        default=50000, ge=1000, description="Model saving frequency (in timesteps)"
    )
    model_path: str = Field(default="./models", description="Directory to save models")
    tensorboard_log: str = Field(
        default="./tensorboard_logs", description="Directory for tensorboard logs"
    )


class SelfPlayConfig(BaseModel):
    """Configuration for self-play training parameters."""
    
    enabled: bool = Field(
        default=False,
        description="Enable self-play training"
    )
    adaptive: bool = Field(
        default=True,
        description="Use adaptive opponent selection (vs fixed opponent)"
    )
    opponent_model_path: Optional[str] = Field(
        default=None,
        description="Path to fixed opponent model (only used if adaptive=False)"
    )
    selection_strategy: str = Field(
        default="mixed",
        description="Opponent selection strategy: best, random, mixed, curriculum"
    )
    opponent_randomness: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Probability of opponent making random moves"
    )
    refresh_pool_frequency: int = Field(
        default=100,
        ge=10,
        description="Episodes between opponent pool refreshes"
    )


class DabloRLConfig(BaseModel):
    """Main configuration class for Dablo RL training."""

    model_config = ConfigDict(extra="forbid")

    # Core configurations
    game_config: DabloConfig = Field(
        default_factory=DabloConfig.create_default,
        description="Dablo game configuration",
    )
    reward_config: RewardConfig = Field(
        default_factory=RewardConfig, description="Reward calculation parameters"
    )
    training_config: TrainingConfig = Field(
        default_factory=TrainingConfig, description="MaskablePPO training parameters"
    )
    evaluation_config: EvaluationConfig = Field(
        default_factory=EvaluationConfig, description="Model evaluation parameters"
    )
    checkpoint_config: CheckpointConfig = Field(
        default_factory=CheckpointConfig, description="Model saving and checkpointing"
    )
    selfplay_config: SelfPlayConfig = Field(
        default_factory=SelfPlayConfig, description="Self-play training configuration"
    )

    # Additional options
    verbose: int = Field(
        default=1, ge=0, le=2, description="Verbosity level (0=silent, 1=info, 2=debug)"
    )
    render_eval: bool = Field(
        default=False, description="Render games during evaluation"
    )
    seed: int | None = Field(
        default=None, description="Random seed for reproducibility"
    )

    @classmethod
    def create_quick_training(cls) -> "DabloRLConfig":
        """Create configuration for quick training/testing."""
        return cls(
            game_config=DabloConfig.create_test_config(),
            training_config=TrainingConfig(total_timesteps=10000, n_envs=2),
            evaluation_config=EvaluationConfig(
                eval_freq=2000, n_eval_episodes=5, final_eval_episodes=10
            ),
        )

    @classmethod
    def create_production_training(cls) -> "DabloRLConfig":
        """Create configuration for production training."""
        return cls(
            training_config=TrainingConfig(
                total_timesteps=1000000, n_envs=8, learning_rate=3e-4
            ),
            evaluation_config=EvaluationConfig(
                eval_freq=25000, n_eval_episodes=20, final_eval_episodes=100
            ),
            checkpoint_config=CheckpointConfig(save_freq=100000),
        )
    
    @classmethod
    def create_selfplay_training(cls, strategy: str = "mixed") -> "DabloRLConfig":
        """Create configuration for self-play training against existing models."""
        return cls(
            training_config=TrainingConfig(
                total_timesteps=2000000, n_envs=8, learning_rate=3e-4
            ),
            evaluation_config=EvaluationConfig(
                eval_freq=50000, n_eval_episodes=20, final_eval_episodes=100
            ),
            checkpoint_config=CheckpointConfig(save_freq=100000),
            selfplay_config=SelfPlayConfig(
                enabled=True,
                adaptive=True,
                selection_strategy=strategy
            ),
        )

    @classmethod
    def load_from_file(cls, config_path: str) -> "DabloRLConfig":
        """Load configuration from JSON or YAML file."""
        config_path = Path(config_path)

        with open(config_path) as f:
            if config_path.suffix.lower() in [".yml", ".yaml"]:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        return cls(**data)

    def save_to_file(self, config_path: str) -> None:
        """Save configuration to JSON or YAML file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        data = self.model_dump()

        with open(config_path, "w") as f:
            if config_path.suffix.lower() in [".yml", ".yaml"]:
                yaml.dump(data, f, default_flow_style=False, indent=2)
            else:
                json.dump(data, f, indent=2)

    def to_reward_dict(self) -> dict[str, Any]:
        """Convert reward config to dictionary for reward calculator."""
        return self.reward_config.model_dump()

    def to_training_dict(self) -> dict[str, Any]:
        """Convert training config to dictionary for model creation."""
        return self.training_config.model_dump(exclude={"n_envs", "total_timesteps"})

    def to_env_dict(self) -> dict[str, Any]:
        """Convert to environment configuration dictionary."""
        return {
            "config": self.game_config,
            "render_mode": "human" if self.render_eval else None,
        }
