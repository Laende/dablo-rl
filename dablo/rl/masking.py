"""
Action masking implementation for Dablo RL environment.

Critical component for MaskablePPO training that converts valid game moves
into boolean masks for the action space.
"""

import numpy as np

from ..core.game import DabloGame
from .spaces import DabloActionSpace


class DabloActionMasker:
    """Handles action masking for valid moves in Dablo game."""

    def __init__(self, action_space: DabloActionSpace):
        """Initialize with action space for encoding moves."""
        self.action_space = action_space
        self.n_positions = action_space.n_positions
        self.total_actions = self.n_positions * self.n_positions

    def get_action_mask(self, game: DabloGame) -> np.ndarray:
        """
        Generate boolean mask for valid actions.

        Returns array of shape (2500,) with True for valid actions.
        Critical: 99.5% of actions are typically invalid.
        """
        # Initialize all actions as invalid
        mask = np.zeros(self.total_actions, dtype=bool)

        # Get valid moves from game
        valid_moves = game.get_all_valid_moves()

        # Mark valid actions as True
        for move in valid_moves:
            try:
                from_idx, to_idx = self.action_space.encode_move(
                    move.from_pos, move.to_pos
                )
                action_idx = self.action_space.get_action_index((from_idx, to_idx))
                mask[action_idx] = True
            except KeyError as e:
                # This shouldn't happen with properly configured games
                print(
                    f"Warning: Invalid position in move {move.from_pos} -> {move.to_pos}: {e}"
                )
                continue

        return mask

    def decode_masked_action(
        self, action_index: int
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Convert flat action index back to board positions."""
        from_idx, to_idx = self.action_space.get_action_from_index(action_index)
        return self.action_space.decode_action((from_idx, to_idx))

    def validate_action(self, game: DabloGame, action: tuple[int, int]) -> bool:
        """Check if action is valid in current game state."""
        mask = self.get_action_mask(game)
        action_idx = self.action_space.get_action_index(action)
        return mask[action_idx]

    def get_valid_actions_count(self, game: DabloGame) -> int:
        """Count valid actions (for monitoring/debugging)."""
        return len(game.get_all_valid_moves())

    def get_action_efficiency(self, game: DabloGame) -> float:
        """Calculate percentage of actions that are valid."""
        valid_count = self.get_valid_actions_count(game)
        return valid_count / self.total_actions
