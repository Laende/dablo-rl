"""
Core Dablo game logic

Contains the main game state, board management, and move execution.
"""

import random
from typing import Any

from .config import DabloConfig
from .pieces import PieceType, get_piece_symbol
from .player import Player
from .rules import GameRules, Move, MovementValidator


class DabloGame:
    """Core Dablo game logic for RL environment."""

    def __init__(
        self, config: DabloConfig | None = None, initial_state: str = "default"
    ):
        self.config = config or DabloConfig.create_default()

        self.board_state: dict[tuple[float, float], PieceType] = {}
        self.nodes: dict[tuple[float, float], list[tuple[float, float]]] = (
            self._create_board_graph()
        )
        self.positions: list[tuple[float, float]] = sorted(self.nodes.keys())

        self.current_player = Player.P1
        self.game_over = False
        self.winner = None
        self.win_reason = None
        self.capture_sequence = False
        self.capturing_piece: tuple[float, float] | None = None
        self.move_count = 0
        self.last_captured_piece = None

        # For reward calculation
        self.initial_piece_counts = None

        self.p1_pieces: set[tuple[float, float]] = set()
        self.p2_pieces: set[tuple[float, float]] = set()

        if not initial_state == "empty":
            self._setup_initial_pieces()

    def _create_board_graph(
        self,
    ) -> dict[tuple[float, float], list[tuple[float, float]]]:
        """Creates the board graph using tuple coordinates."""
        all_nodes: set[tuple[float, float]] = set()
        rows, cols = self.config.board_rows, self.config.board_cols

        for r_int in range(rows):
            for c_int in range(cols):
                r, c = float(r_int), float(c_int)
                all_nodes.add((r, c))
                if r_int < rows - 1 and c_int < cols - 1:
                    all_nodes.add((r + 0.5, c + 0.5))

        graph: dict[tuple[float, float], list[tuple[float, float]]] = {
            pos: [] for pos in all_nodes
        }

        for r, c in all_nodes:
            current_pos = (r, c)
            if r == int(r) and c == int(c):  # Primary Node
                potential_neighbors = [
                    (r - 1.0, c),
                    (r + 1.0, c),
                    (r, c - 1.0),
                    (r, c + 1.0),
                    (r - 0.5, c - 0.5),
                    (r - 0.5, c + 0.5),
                    (r + 0.5, c - 0.5),
                    (r + 0.5, c + 0.5),
                ]
            else:  # Secondary Node
                potential_neighbors = [
                    (r - 0.5, c - 0.5),
                    (r - 0.5, c + 0.5),
                    (r + 0.5, c - 0.5),
                    (r + 0.5, c + 0.5),
                ]

            for neighbor in potential_neighbors:
                if neighbor in all_nodes:
                    graph[current_pos].append(neighbor)

        return graph

    def _setup_initial_pieces(self):
        """Set up the initial game position from the config."""
        # Initialize empty board
        self.board_state = dict.fromkeys(self.nodes, PieceType.EMPTY)
        self.p1_pieces.clear()
        self.p2_pieces.clear()

        # Add P1 pieces using the proper method
        for piece_type, positions in self.config.p1_setup.items():
            for pos in positions:
                self.add_piece(pos, piece_type)

        # Add P2 pieces using the proper method
        for piece_type, positions in self.config.p2_setup.items():
            for pos in positions:
                self.add_piece(pos, piece_type)

    def add_piece(self, pos: tuple[float, float], piece_type: PieceType) -> bool:
        """
        Add a piece to the board at the specified position.

        Args:
            pos: Position to place the piece
            piece_type: Type of piece to place

        Returns:
            True if piece was successfully added, False if position is invalid or occupied
        """
        # Validate position exists on board
        if pos not in self.board_state:
            return False

        # Check if position is already occupied
        if self.board_state[pos] != PieceType.EMPTY:
            return False

        # Add piece to board state
        self.board_state[pos] = piece_type

        # Update piece tracking sets
        if piece_type.value > 0:  # Player 1 piece
            self.p1_pieces.add(pos)
        elif piece_type.value < 0:  # Player 2 piece
            self.p2_pieces.add(pos)
        # Note: EMPTY pieces don't need to be tracked

        return True

    def remove_piece(self, pos: tuple[float, float]) -> PieceType:
        """
        Remove a piece from the board at the specified position.

        Args:
            pos: Position to remove piece from

        Returns:
            The piece type that was removed, or PieceType.EMPTY if no piece was there
        """
        # Get the piece type before removing
        removed_piece = self.board_state.get(pos, PieceType.EMPTY)

        # Remove from board state
        self.board_state[pos] = PieceType.EMPTY

        # Update piece tracking sets
        if removed_piece.value > 0:  # Was Player 1 piece
            self.p1_pieces.discard(pos)
        elif removed_piece.value < 0:  # Was Player 2 piece
            self.p2_pieces.discard(pos)

        return removed_piece

    def clear_board(self):
        """Clear all pieces from the board."""
        for pos in self.board_state:
            self.board_state[pos] = PieceType.EMPTY
        self.p1_pieces.clear()
        self.p2_pieces.clear()

    def setup_custom_position(self, pieces: dict[tuple[float, float], PieceType]):
        """
        Set up a custom board position.

        Args:
            pieces: Dictionary mapping positions to piece types
        """
        # Clear the board first
        self.clear_board()

        # Add each piece
        for pos, piece_type in pieces.items():
            if piece_type != PieceType.EMPTY:
                self.add_piece(pos, piece_type)

    def get_valid_moves(self, pos: tuple[float, float]) -> list[Move]:
        """Get all valid moves for a piece at given position"""
        piece: PieceType | None = self.board_state.get(pos)
        if not piece or (piece.value > 0) != (self.current_player > 0):
            return []

        all_possible_moves = []
        for neighbor_pos in self.nodes.get(pos, []):
            target_piece = self.board_state.get(neighbor_pos, PieceType.EMPTY)

            # Regular move to empty position
            if target_piece == PieceType.EMPTY:
                if GameRules.is_valid_regular_move(
                    pos, neighbor_pos, piece, self.board_state, self.nodes
                ):
                    all_possible_moves.append(Move(from_pos=pos, to_pos=neighbor_pos))

            # Capture move
            else:
                is_opponent = piece.value * target_piece.value < 0
                if is_opponent:
                    # Calculate landing position
                    landing_pos = MovementValidator.calculate_capture_landing(
                        pos, neighbor_pos
                    )

                    if GameRules.is_valid_capture_move(
                        pos,
                        neighbor_pos,
                        landing_pos,
                        piece,
                        target_piece,
                        self.board_state,
                        self.nodes,
                    ):
                        all_possible_moves.append(
                            Move(
                                from_pos=pos,
                                to_pos=landing_pos,
                                capture_pos=neighbor_pos,
                            )
                        )

        return all_possible_moves

    def get_all_valid_moves(self) -> list[Move]:
        """Get all valid moves for current player"""
        # If in capture sequence, only the capturing piece can move
        if self.capture_sequence and self.capturing_piece:
            return self.get_valid_moves(self.capturing_piece)

        moves = []
        player_pieces = (
            self.p1_pieces if self.current_player == Player.P1 else self.p2_pieces
        )
        for pos in player_pieces:
            moves.extend(self.get_valid_moves(pos))
        return moves

    def get_king_position(self, player: Player) -> tuple[float, float] | None:
        """Get king position for specified player"""
        king_type = PieceType.P1_KING if player == Player.P1 else PieceType.P2_KING
        player_pieces = self.p1_pieces if player == Player.P1 else self.p2_pieces

        for pos in player_pieces:
            if self.board_state[pos] == king_type:
                return pos
        return None

    def make_move(self, move: Move) -> tuple[bool, dict[str, Any]]:
        """Execute a move and return success status and move info"""
        if self.game_over:
            return False, {"error": "Game is over"}

        # Store move info for rewards
        move_info = {"is_capture": False, "chain_capture_available": False}

        # Execute the move using proper piece management methods
        moving_piece = self.remove_piece(move.from_pos)
        move_info["piece_moved"] = moving_piece
        self.add_piece(move.to_pos, moving_piece)
        self.move_count += 1

        # Handle capture
        if move.is_capture:
            captured_piece = self.remove_piece(move.capture_pos)
            self.last_captured_piece = captured_piece
            move_info.update({"is_capture": True, "captured_piece": captured_piece})

            # Check for chain captures (only captures from the new position are valid)
            additional_captures = [
                m for m in self.get_valid_moves(move.to_pos) if m.is_capture
            ]

            # Prevent moving back to the starting square in the same turn
            additional_captures = [
                m for m in additional_captures if m.to_pos != move.from_pos
            ]

            if additional_captures:
                self.capture_sequence = True
                self.capturing_piece = move.to_pos
                move_info["chain_capture_available"] = True
                return True, move_info

        # End turn
        self.capture_sequence = False
        self.capturing_piece = None
        self.current_player = self.current_player.opponent()
        self._check_game_state()

        return True, move_info

    def _check_game_state(self):
        """Check if game has ended and determine winner"""

        # Generate the list of moves ONCE.
        has_valid_moves = bool(self.get_all_valid_moves())

        p1_king_pos = self.get_king_position(Player.P1)
        p2_king_pos = self.get_king_position(Player.P2)

        winner, reason = GameRules.check_win_condition(
            p1_pieces_count=len(self.p1_pieces),
            p2_pieces_count=len(self.p2_pieces),
            p1_king_exists=p1_king_pos is not None,
            p2_king_exists=p2_king_pos is not None,
            move_count=self.move_count,
            move_limit=self.config.move_limit,
            has_valid_moves=has_valid_moves,
            current_player=self.current_player,
        )

        if winner:
            self.end_game(winner, reason)

    def end_game(self, winner: Player, reason: str):
        """Manually end the game with specified winner and reason"""
        self.game_over = True
        self.winner = winner
        self.win_reason = reason

    def make_random_move(self) -> tuple[bool, dict[str, Any]]:
        """Make a random valid move (for AI opponent)"""
        valid_moves = self.get_all_valid_moves()
        if not valid_moves:
            return False, {"error": "No valid moves"}

        return self.make_move(random.choice(valid_moves))

    def render_ascii(self) -> str:
        """Render the current game state as ASCII string"""
        lines = [
            f"--- Move: {self.move_count} | Player: {self.current_player.name} ---"
        ]

        if self.capture_sequence:
            lines.append(
                f"!! Chain capture required for piece at {self.capturing_piece} !!"
            )

        rows_to_render: dict[float, list[tuple[float, float]]] = {}
        for r, c in self.positions:
            if r not in rows_to_render:
                rows_to_render[r] = []
            rows_to_render[r].append((r, c))

        for r in sorted(rows_to_render.keys()):
            row_str = f"{r: >3.1f} |"
            if r != int(r):
                row_str += "  "  # Indent secondary rows

            # Sort columns to ensure consistent order
            sorted_cols = sorted(rows_to_render[r], key=lambda pos: pos[1])
            for pos in sorted_cols:
                piece = self.board_state.get(pos)
                row_str += get_piece_symbol(piece) if piece is not None else "   "
            lines.append(row_str)

        if self.game_over:
            if self.winner:
                lines.append(f"\n🎉 Game Over! Winner: {self.winner.name}")
            else:
                lines.append("\n🎉 Game Over! Result: Draw")

        lines.append("-" * 45)
        return "\n".join(lines)
