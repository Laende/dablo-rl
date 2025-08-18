"""
Utility functions and classes for Dablo game operations

Provides helper functions for common game operations like piece counting,
position analysis, board state queries, and game statistics.
"""

from functools import lru_cache
from typing import Any

from .pieces import PieceType


class GameUtils:
    """General utility functions for game operations"""

    @staticmethod
    def count_pieces(
        board_state: dict[tuple[float, float], PieceType],
    ) -> dict[PieceType, int]:
        """
        Count all pieces on the board by type.
        """
        counts = {}
        for piece in board_state.values():
            counts[piece] = counts.get(piece, 0) + 1
        return counts

    @staticmethod
    @lru_cache(maxsize=1)  # The center positions never change
    def get_center_positions() -> list[tuple[float, float]]:
        """Get center board positions for control analysis"""
        return [
            (2.5, 2.0),
            (2.5, 1.5),
            (2.5, 2.5),
            (2.0, 2.0),
            (3.0, 2.0),
            (2.5, 1.0),
            (2.5, 3.0),
        ]

    @staticmethod
    def calculate_center_control(
        board_state: dict[tuple[float, float], PieceType],
    ) -> tuple[int, int]:
        """
        Calculate center control for both players (p1_control, p2_control).
        """
        center_positions = GameUtils.get_center_positions()
        p1_control = 0
        p2_control = 0

        for pos in center_positions:
            piece = board_state.get(pos, PieceType.EMPTY)

            if piece.value > 0:
                p1_control += 1
            elif piece.value < 0:
                p2_control += 1

        return p1_control, p2_control

    @staticmethod
    def analyze_board_features(
        board_state: dict[tuple[float, float], PieceType],
        move_count: int,
        p1_king_pos: tuple[float, float] | None = None,
        p2_king_pos: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        """
        Get advanced features for reward calculation. This logic was moved from DabloGame.
        """
        # Count all pieces by type and player
        piece_counts_by_type = GameUtils.count_pieces(board_state)

        p1_warriors = piece_counts_by_type.get(PieceType.P1_WARRIOR, 0)
        p1_princes = piece_counts_by_type.get(PieceType.P1_PRINCE, 0)
        p1_kings = piece_counts_by_type.get(PieceType.P1_KING, 0)

        p2_warriors = piece_counts_by_type.get(PieceType.P2_WARRIOR, 0)
        p2_princes = piece_counts_by_type.get(PieceType.P2_PRINCE, 0)
        p2_kings = piece_counts_by_type.get(PieceType.P2_KING, 0)

        p1_total_pieces = p1_warriors + p1_princes + p1_kings
        p2_total_pieces = p2_warriors + p2_princes + p2_kings

        # Calculate center control
        p1_center_control, p2_center_control = GameUtils.calculate_center_control(
            board_state
        )

        features = {
            "p1_pieces": p1_total_pieces,
            "p2_pieces": p2_total_pieces,
            "p1_warriors": p1_warriors,
            "p1_princes": p1_princes,
            "p1_kings": p1_kings,
            "p2_warriors": p2_warriors,
            "p2_princes": p2_princes,
            "p2_kings": p2_kings,
            "piece_difference": p1_total_pieces - p2_total_pieces,
            "p1_king_pos": p1_king_pos,
            "p2_king_pos": p2_king_pos,
            "p1_center_control": p1_center_control,
            "p2_center_control": p2_center_control,
            "move_count": move_count,
        }
        return features
