"""
Position mapping and encoding utilities for Dablo RL environment.

Provides bidirectional mapping between board positions and action indices,
plus encoding utilities for converting game state to observation vectors.
"""

from gymnasium import spaces
import numpy as np

from ..core.pieces import PieceType
from ..core.player import Player


class DabloActionSpace:
    """Manages action space encoding for Dablo moves."""

    def __init__(self, positions: list[tuple[float, float]]):
        """Initialize with sorted board positions from game."""
        self.positions = positions
        self.n_positions = len(positions)

        # Create bidirectional position mappings
        self.pos_to_idx = {pos: idx for idx, pos in enumerate(positions)}
        self.idx_to_pos = {idx: pos for idx, pos in enumerate(positions)}

        # Gymnasium action space: Discrete for flattened (from*n_pos + to) actions
        self.gym_space = spaces.Discrete(self.n_positions * self.n_positions)

    def encode_move(
        self, from_pos: tuple[float, float], to_pos: tuple[float, float]
    ) -> tuple[int, int]:
        """Convert board positions to action indices."""
        from_idx = self.pos_to_idx[from_pos]
        to_idx = self.pos_to_idx[to_pos]
        return (from_idx, to_idx)

    def decode_action(
        self, action: tuple[int, int]
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Convert action indices back to board positions."""
        from_idx, to_idx = action
        from_pos = self.idx_to_pos[from_idx]
        to_pos = self.idx_to_pos[to_idx]
        return (from_pos, to_pos)

    def get_action_index(self, action: tuple[int, int]) -> int:
        """Convert (from_idx, to_idx) to single action index for masking."""
        from_idx, to_idx = action
        return from_idx * self.n_positions + to_idx

    def get_action_from_index(self, action_idx: int) -> tuple[int, int]:
        """Convert single action index back to (from_idx, to_idx)."""
        from_idx = action_idx // self.n_positions
        to_idx = action_idx % self.n_positions
        return (from_idx, to_idx)


class DabloObservationSpace:
    """Manages observation space encoding for Dablo game state."""

    def __init__(self, positions: list[tuple[float, float]], move_limit: int = 500):
        """Initialize observation space with board size and game parameters."""
        self.positions = positions
        self.n_positions = len(positions)
        self.move_limit = move_limit

        # Piece type encoding for observations
        self.piece_encoding = {
            PieceType.EMPTY: 0,
            PieceType.P1_WARRIOR: 1,
            PieceType.P1_PRINCE: 2,
            PieceType.P1_KING: 3,
            PieceType.P2_WARRIOR: 4,
            PieceType.P2_PRINCE: 5,
            PieceType.P2_KING: 6,
        }

        # Gymnasium observation space
        self.gym_space = spaces.Dict(
            {
                "board": spaces.Box(0, 6, shape=(self.n_positions,), dtype=np.int8),
                "current_player": spaces.Box(0, 1, shape=(1,), dtype=np.int8),
                "move_count": spaces.Box(0, move_limit, shape=(1,), dtype=np.int16),
                "capture_sequence": spaces.Box(0, 1, shape=(1,), dtype=np.int8),
            }
        )

    def encode_board_state(
        self, board_state: dict[tuple[float, float], PieceType]
    ) -> np.ndarray:
        """Convert board state to numerical vector."""
        board_vector = np.zeros(self.n_positions, dtype=np.int8)

        for i, pos in enumerate(self.positions):
            piece_type = board_state.get(pos, PieceType.EMPTY)
            board_vector[i] = self.piece_encoding[piece_type]

        return board_vector

    def encode_player(self, player: Player) -> int:
        """Convert player enum to observation index."""
        return 0 if player == Player.P1 else 1

    def create_observation(
        self,
        board_state: dict[tuple[float, float], PieceType],
        current_player: Player,
        move_count: int,
        capture_sequence: bool,
    ) -> dict[str, np.ndarray]:
        """Create complete observation dictionary."""
        return {
            "board": self.encode_board_state(board_state),
            "current_player": np.array(
                [self.encode_player(current_player)], dtype=np.int8
            ),
            "move_count": np.array([move_count], dtype=np.int16),
            "capture_sequence": np.array([int(capture_sequence)], dtype=np.int8),
        }

    def decode_board_state(
        self, board_vector: np.ndarray
    ) -> dict[tuple[float, float], PieceType]:
        """Convert numerical vector back to board state (for debugging)."""
        # Reverse encoding lookup
        idx_to_piece = {idx: piece for piece, idx in self.piece_encoding.items()}

        board_state = {}
        for i, piece_idx in enumerate(board_vector):
            if piece_idx > 0:  # Skip empty positions
                piece_type = idx_to_piece[piece_idx]
                board_state[self.positions[i]] = piece_type

        return board_state
