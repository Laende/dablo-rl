"""
Gymnasium environment wrapper for Dablo game.

Main RL environment implementing the Gymnasium interface with action masking
support for training MaskablePPO agents.
"""

from typing import Any

import gymnasium as gym
import numpy as np

from ..core.config import DabloConfig
from ..core.game import DabloGame
from ..core.player import Player
from ..core.rules import Move
from .masking import DabloActionMasker
from .rewards import DabloRewardCalculator
from .spaces import DabloActionSpace, DabloObservationSpace


class DabloRLEnvironment(gym.Env):
    """
    Gymnasium environment for Dablo game with action masking support.

    Designed for training with MaskablePPO from sb3-contrib.
    """

    metadata = {"render_modes": ["human", "text"], "render_fps": 1}

    def __init__(
        self,
        config: DabloConfig | None = None,
        render_mode: str | None = None,
        reward_config: dict[str, float] | None = None,
    ):
        """
        Initialize Dablo RL environment.

        Args:
            config: Game configuration (uses default if None)
            render_mode: Rendering mode ("human", "text", or None)
            reward_config: Reward calculation parameters
        """
        super().__init__()

        # Game configuration
        self.config = config or DabloConfig.create_default()
        self.render_mode = render_mode

        # Create initial game to get board structure
        temp_game = DabloGame(self.config)
        self.positions = temp_game.positions

        # Initialize RL components
        self.action_space_manager = DabloActionSpace(self.positions)
        self.observation_space_manager = DabloObservationSpace(
            self.positions, self.config.move_limit
        )
        self.action_masker = DabloActionMasker(self.action_space_manager)

        # Setup reward calculator
        reward_config = reward_config or {}
        self.reward_calculator = DabloRewardCalculator(**reward_config)

        # Gymnasium spaces
        self.action_space = self.action_space_manager.gym_space
        self.observation_space = self.observation_space_manager.gym_space

        # Environment state
        self.game: DabloGame | None = None
        self.current_player = Player.P1
        self._last_observation = None
        self._episode_step_count = 0
        self._episode_count = 0  # Track episodes for alternating starts

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Create new game
        self.game = DabloGame(self.config)
        self.current_player = Player.P1
        self._episode_step_count = 0

        # Generate initial observation
        observation = self._get_observation()
        self._last_observation = observation

        # Info dictionary
        info = {
            "valid_moves": len(self.game.get_all_valid_moves()),
            "current_player": self.current_player.name,
            "game_over": False,
        }

        return observation, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action as [from_idx, to_idx] array

        Returns:
            observation: Next state observation
            reward: Reward for the action
            terminated: Whether episode ended (game over)
            truncated: Whether episode was truncated (move limit)
            info: Additional information
        """
        if self.game is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Convert action to game move (now single discrete action)
        action_idx = int(action)
        from_idx, to_idx = self.action_space_manager.get_action_from_index(action_idx)
        from_pos, to_pos = self.action_space_manager.decode_action((from_idx, to_idx))

        # Validate action is legal
        if not self._is_action_valid(action):
            # Invalid action penalty
            reward = -10.0
            terminated = True
            truncated = False
            observation = self._get_observation()
            info = {
                "invalid_action": True,
                "action": action.tolist(),
                "from_pos": from_pos,
                "to_pos": to_pos,
            }
            return observation, reward, terminated, truncated, info

        # Store previous state for reward calculation
        previous_player = self.current_player
        previous_board_state = dict(self.game.board_state)  # Copy current state

        # Execute move in game
        move = Move(from_pos, to_pos)
        success, result = self.game.make_move(move)

        if not success:
            # This shouldn't happen if action masking is working correctly
            reward = -10.0
            terminated = True
            truncated = False
            observation = self._get_observation()
            info = {"move_failed": True, "result": result}
            return observation, reward, terminated, truncated, info

        # Update step count
        self._episode_step_count += 1

        # Calculate reward for the player who just moved
        reward = self.reward_calculator.get_reward(
            self.game, previous_player, previous_board_state
        )

        # Check terminal conditions
        terminated = self.game.game_over
        truncated = (
            self._episode_step_count >= self.config.move_limit and not terminated
        )

        # Generate next observation
        observation = self._get_observation()
        self._last_observation = observation

        # Create info dictionary
        info = {
            "valid_moves": len(self.game.get_all_valid_moves())
            if not terminated
            else 0,
            "current_player": self.game.current_player.name
            if not terminated
            else "None",
            "game_over": terminated,
            "winner": self.game.winner.name if self.game.winner else None,
            "win_reason": self.game.win_reason.value if self.game.win_reason else None,
            "move_count": self.game.move_count,
            "material_p1": self.reward_calculator.calculate_material_value(
                self.game.board_state, Player.P1
            ),
            "material_p2": self.reward_calculator.calculate_material_value(
                self.game.board_state, Player.P2
            ),
        }

        return observation, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """
        Get action mask for current state.

        Required method for MaskablePPO integration.
        For Discrete action space, return flat boolean mask.
        """
        if self.game is None:
            # Return all-false mask if game not initialized
            return np.zeros(self.action_masker.total_actions, dtype=bool)

        return self.action_masker.get_action_mask(self.game)

    def render(self) -> str | None:
        """Render the current state."""
        if self.game is None:
            return None

        if self.render_mode == "human" or self.render_mode == "text":
            # Simple text rendering
            output = []
            output.append("=== Dablo Game State ===")
            output.append(f"Move: {self.game.move_count}")
            output.append(f"Current Player: {self.game.current_player.name}")
            output.append(f"Game Over: {self.game.game_over}")

            if self.game.game_over:
                winner_name = self.game.winner.name if self.game.winner else "Draw"
                output.append(f"Result: {winner_name}")
                if self.game.win_reason:
                    output.append(f"Reason: {self.game.win_reason.value}")

            # Piece counts
            p1_count = len(self.game.p1_pieces)
            p2_count = len(self.game.p2_pieces)
            output.append(f"Pieces - P1: {p1_count}, P2: {p2_count}")

            # Valid moves
            if not self.game.game_over:
                valid_moves = len(self.game.get_all_valid_moves())
                output.append(f"Valid moves: {valid_moves}")

            text_output = "\n".join(output)

            if self.render_mode == "human":
                print(text_output)

            return text_output

        return None

    def close(self):
        """Clean up environment resources."""
        self.game = None
        self._last_observation = None

    def _get_observation(self) -> dict[str, np.ndarray]:
        """Generate observation from current game state."""
        if self.game is None:
            raise RuntimeError("Game not initialized")

        return self.observation_space_manager.create_observation(
            self.game.board_state,
            self.game.current_player,
            self.game.move_count,
            self.game.capture_sequence,
        )

    def _is_action_valid(self, action: np.ndarray) -> bool:
        """Check if action is valid in current state."""
        if self.game is None:
            return False

        action_idx = int(action)
        from_idx, to_idx = self.action_space_manager.get_action_from_index(action_idx)
        action_tuple = (from_idx, to_idx)
        return self.action_masker.validate_action(self.game, action_tuple)

    def get_wrapper_attr(self, name: str) -> Any:
        """Get attribute from environment (for sb3 compatibility)."""
        return getattr(self, name)

    def get_action_meanings(self) -> list[str]:
        """Get human-readable action meanings (for debugging)."""
        meanings = []
        for from_idx in range(self.action_space_manager.n_positions):
            for to_idx in range(self.action_space_manager.n_positions):
                from_pos = self.action_space_manager.idx_to_pos[from_idx]
                to_pos = self.action_space_manager.idx_to_pos[to_idx]
                meanings.append(f"Move from {from_pos} to {to_pos}")
        return meanings
