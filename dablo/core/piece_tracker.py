"""
Event-driven piece tracking for optimal performance.

This module provides the PieceTracker class that efficiently manages
piece positions and state using an event-driven approach.
"""

from .pieces import PieceType
from .player import Player


class PieceTracker:
    """Event-driven piece tracking for optimal performance."""

    def __init__(self, positions: list[tuple[float, float]]):
        """Initialize tracker with all board positions."""
        self.positions_dict: dict[Player, set[tuple[float, float]]] = {
            Player.P1: set(),
            Player.P2: set(),
        }
        self.kings: dict[Player, tuple[float, float] | None] = {
            Player.P1: None,
            Player.P2: None,
        }
        self.board: dict[tuple[float, float], PieceType] = dict.fromkeys(
            positions, PieceType.EMPTY
        )

    def add_piece(self, pos: tuple[float, float], piece_type: PieceType) -> bool:
        """Add a piece at position. Returns False if position invalid or occupied."""

        if pos not in self.board:
            return False

        if self.board[pos] != PieceType.EMPTY:
            return False

        self.board[pos] = piece_type

        if piece_type.value > 0:  # Player 1 piece
            self.positions_dict[Player.P1].add(pos)
            if piece_type == PieceType.P1_KING:
                self.kings[Player.P1] = pos
        elif piece_type.value < 0:  # Player 2 piece
            self.positions_dict[Player.P2].add(pos)
            if piece_type == PieceType.P2_KING:
                self.kings[Player.P2] = pos

        return True

    def remove_piece(self, pos: tuple[float, float]) -> PieceType:
        """Remove piece at position. Returns the piece that was removed."""
        piece = self.board.get(pos, PieceType.EMPTY)
        self.board[pos] = PieceType.EMPTY

        # Remove from both player sets
        self.positions_dict[Player.P1].discard(pos)
        self.positions_dict[Player.P2].discard(pos)

        # Clear king position if this was a king
        if self.kings[Player.P1] == pos:
            self.kings[Player.P1] = None
        if self.kings[Player.P2] == pos:
            self.kings[Player.P2] = None

        return piece

    def clear_board(self):
        """Clear all pieces from the board."""
        for pos in self.board:
            self.board[pos] = PieceType.EMPTY
        self.positions_dict[Player.P1].clear()
        self.positions_dict[Player.P2].clear()
        self.kings[Player.P1] = None
        self.kings[Player.P2] = None

    def clear_all(self):
        """Clear all pieces from the board. (Alias for clear_board)"""
        self.clear_board()

    def set_board_state(self, board_state: dict[tuple[float, float], PieceType]):
        """Set entire board state from a dictionary."""
        self.clear_board()
        for pos, piece_type in board_state.items():
            if piece_type != PieceType.EMPTY:
                self.add_piece(pos, piece_type)

    def get_board_state(self) -> dict[tuple[float, float], PieceType]:
        """Get complete board state as dictionary."""
        return self.board.copy()

    def get_player_pieces(self, player: Player) -> set[tuple[float, float]]:
        """Get all pieces for a player."""
        return self.positions_dict[player]

    def get_king_position(self, player: Player) -> tuple[float, float] | None:
        """Get king position for player."""
        return self.kings[player]

    def get_piece_at(self, pos: tuple[float, float]) -> PieceType:
        """Get piece type at position."""
        return self.board.get(pos, PieceType.EMPTY)
