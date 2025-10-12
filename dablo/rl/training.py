"""
Training utilities and self-play logic for Dablo RL.

Provides training workflows, callbacks, and evaluation utilities
specifically designed for MaskablePPO training with self-play.
"""

import os
import time

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from .environment import DabloRLEnvironment
from .selfplay_env import DabloSelfPlayEnvironment
from .adaptive_selfplay_env import DabloAdaptiveSelfPlayEnvironment


class DabloSelfPlayManager:
    """Manages self-play training for Dablo RL agents."""

    def __init__(
        self,
        env_config: dict | None = None,
        reward_config: dict | None = None,
        model_save_path: str = "models",
        opponent_model_path: str | None = None,
        adaptive_selfplay: bool = False,
        selfplay_strategy: str = "mixed",
    ):
        """
        Initialize self-play manager.

        Args:
            env_config: Environment configuration parameters
            reward_config: Reward calculation parameters
            model_save_path: Directory to save trained models
            opponent_model_path: Path to opponent model for self-play (None for regular training)
            adaptive_selfplay: Use adaptive opponent selection from model pool
            selfplay_strategy: Strategy for opponent selection ("best", "random", "mixed", "curriculum")
        """
        self.env_config = env_config or {}
        self.reward_config = reward_config or {}
        self.model_save_path = model_save_path
        self.opponent_model_path = opponent_model_path
        self.adaptive_selfplay = adaptive_selfplay
        self.selfplay_strategy = selfplay_strategy

        # Create save directory
        os.makedirs(model_save_path, exist_ok=True)

        # Training history
        self.training_history = []

    def create_environment(self, render_mode: str | None = None, monitor_wrapper: bool = True):
        """Create a single Dablo RL environment (self-play or regular)."""
        env_config = dict(self.env_config)
        if render_mode is not None:
            env_config['render_mode'] = render_mode
        
        # Create appropriate environment type
        if self.adaptive_selfplay:
            # Adaptive self-play against model pool
            env = DabloAdaptiveSelfPlayEnvironment(
                model_directory=self.model_save_path,
                selection_strategy=self.selfplay_strategy,
                reward_config=self.reward_config,
                **env_config
            )
        elif self.opponent_model_path:
            # Fixed opponent self-play
            env = DabloSelfPlayEnvironment(
                opponent_model_path=self.opponent_model_path,
                reward_config=self.reward_config,
                **env_config
            )
        else:
            # Regular training (agent plays both sides)
            env = DabloRLEnvironment(
                reward_config=self.reward_config,
                **env_config
            )
        
        # Wrap with Monitor for better episode tracking
        if monitor_wrapper:
            log_dir = os.path.join(self.model_save_path, "monitor_logs")
            os.makedirs(log_dir, exist_ok=True)
            
            # Use Monitor without specific info_keywords to avoid KeyError issues
            # Monitor will still log episode rewards and lengths
            env = Monitor(env, log_dir)
        
        return env

    def create_vec_environment(self, n_envs: int = 1) -> VecEnv:
        """Create vectorized environment for parallel training."""

        def make_env():
            return self.create_environment()

        envs = [make_env for _ in range(n_envs)]
        return DummyVecEnv(envs)

    def create_model(
        self,
        env: VecEnv,
        policy: str = "MultiInputPolicy",
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        verbose: int = 1,
        **kwargs,
    ) -> MaskablePPO:
        """
        Create MaskablePPO model for training.

        Args:
            env: Vectorized environment
            policy: Policy type (must be "MultiInputPolicy" for Dict observations)
            learning_rate: Learning rate for optimization
            n_steps: Number of steps per environment per update
            batch_size: Batch size for training
            n_epochs: Number of epochs per update
            verbose: Verbosity level
            **kwargs: Additional MaskablePPO parameters
        """
        model = MaskablePPO(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            verbose=verbose,
            tensorboard_log="./tensorboard_logs/",
            **kwargs,
        )

        return model

    def train_model(
        self,
        model: MaskablePPO,
        total_timesteps: int = 100000,
        eval_freq: int = 10000,
        n_eval_episodes: int = 10,
        save_freq: int = 50000,
        callback: BaseCallback | None = None,
    ) -> MaskablePPO:
        """
        Train model with evaluation and checkpointing.

        Args:
            model: MaskablePPO model to train
            total_timesteps: Total training timesteps
            eval_freq: Frequency of evaluation (in timesteps)
            n_eval_episodes: Number of episodes for evaluation
            save_freq: Frequency of model saving (in timesteps)
            callback: Additional callback for training
        """
        # Create evaluation environment
        eval_env = self.create_vec_environment(n_envs=1)

        # Create evaluation callback
        eval_callback = MaskableEvalCallback(
            eval_env,
            best_model_save_path=os.path.join(self.model_save_path, "best_model"),
            log_path=os.path.join(self.model_save_path, "evaluations"),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            verbose=1,
        )

        # Create checkpoint callback
        checkpoint_callback = DabloCheckpointCallback(
            save_freq=save_freq, save_path=self.model_save_path
        )

        # Combine callbacks
        callbacks = [eval_callback, checkpoint_callback]
        if callback:
            callbacks.append(callback)

        # Start training
        print(f"Starting training for {total_timesteps} timesteps...")
        start_time = time.time()

        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name="dablo_maskable_ppo",
        )

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        # Save final model
        final_model_path = os.path.join(self.model_save_path, "final_model")
        model.save(final_model_path)
        print(f"Final model saved to {final_model_path}")

        # Record training session
        self.training_history.append(
            {
                "timestamp": time.time(),
                "total_timesteps": total_timesteps,
                "training_time": training_time,
                "model_path": final_model_path,
            }
        )

        return model

    def evaluate_model(
        self,
        model: BaseAlgorithm,
        n_eval_episodes: int = 100,
        deterministic: bool = True,
        render: bool = False,
    ) -> dict[str, float]:
        """
        Evaluate trained model performance.

        Args:
            model: Trained model to evaluate
            n_eval_episodes: Number of episodes for evaluation
            deterministic: Whether to use deterministic policy
            render: Whether to render during evaluation
        """
        eval_env = self.create_environment(
            render_mode="human" if render else None, 
            monitor_wrapper=True  # Enable Monitor for evaluation too
        )

        # Evaluate using maskable policy evaluation
        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            return_episode_rewards=False,
        )

        eval_env.close()

        results = {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "n_episodes": n_eval_episodes,
        }

        print("Evaluation Results:")
        print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"  Episodes: {n_eval_episodes}")

        return results

    def load_model(self, model_path: str) -> MaskablePPO:
        """Load a trained model from file."""
        print(f"Loading model from {model_path}")
        return MaskablePPO.load(model_path)


class DabloCheckpointCallback(BaseCallback):
    """Callback for saving model checkpoints during training."""

    def __init__(
        self,
        save_freq: int = 50000,
        save_path: str = "models",
        name_prefix: str = "checkpoint",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            checkpoint_path = os.path.join(
                self.save_path, f"{self.name_prefix}_{self.n_calls}_steps"
            )
            self.model.save(checkpoint_path)

            if self.verbose >= 1:
                print(f"Checkpoint saved at step {self.n_calls}: {checkpoint_path}")

        return True


class DabloTrainingMonitor(BaseCallback):
    """Callback for monitoring Dablo-specific training metrics."""

    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.game_outcomes = []

    def _on_step(self) -> bool:
        # Check if episode finished
        if len(self.locals.get("dones", [])) > 0 and any(self.locals["dones"]):
            # Get episode info
            infos = self.locals.get("infos", [])

            for i, done in enumerate(self.locals["dones"]):
                if done and i < len(infos):
                    info = infos[i]

                    # Record episode statistics
                    if "episode" in info:
                        episode_info = info["episode"]
                        self.episode_rewards.append(episode_info["r"])
                        self.episode_lengths.append(episode_info["l"])

                    # Record game-specific outcomes
                    if "winner" in info:
                        self.game_outcomes.append(info["winner"])

        # Log statistics periodically
        if self.n_calls % 10000 == 0 and self.episode_rewards:
            recent_rewards = self.episode_rewards[-100:]  # Last 100 episodes
            recent_lengths = self.episode_lengths[-100:]

            self.logger.record("train/mean_episode_reward", np.mean(recent_rewards))
            self.logger.record("train/mean_episode_length", np.mean(recent_lengths))

            if self.game_outcomes:
                recent_outcomes = self.game_outcomes[-100:]
                p1_wins = sum(
                    1 for outcome in recent_outcomes if outcome == "P1 (Aalmogh)"
                )
                p2_wins = sum(
                    1 for outcome in recent_outcomes if outcome == "P2 (Daatjh)"
                )
                draws = len(recent_outcomes) - p1_wins - p2_wins

                self.logger.record("train/p1_win_rate", p1_wins / len(recent_outcomes))
                self.logger.record("train/p2_win_rate", p2_wins / len(recent_outcomes))
                self.logger.record("train/draw_rate", draws / len(recent_outcomes))

        return True
