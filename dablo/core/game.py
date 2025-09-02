"""
Core Dablo game logic

Contains the main game state, board management, and move execution.
"""

import random
from typing import Any

from .config import DabloConfig
from .piece_tracker import PieceTracker
from .pieces import PieceType, can_capture, get_piece_symbol
from .player import Player
from .rules import GameRules, Move


class DabloGame:
    """Core Dablo game logic for RL environment."""

    # Class-level cache for board graphs and positions
    _board_cache: dict[
        tuple[int, int],
        tuple[
            dict[tuple[float, float], list[tuple[float, float]]],
            list[tuple[float, float]],
        ],
    ] = {}

    # Cache for piece setups to avoid repeated config parsing
    _setup_cache: dict[str, list[tuple[tuple[float, float], PieceType]]] = {}

    def __init__(
        self, config: DabloConfig | None = None, initial_state: str = "default"
    ):
        self.config = config or DabloConfig.create_default()

        # Get cached board graph and positions
        self.nodes, self.positions = self._get_cached_board_data()

        # Event-driven piece tracking for optimal performance
        self.piece_tracker = PieceTracker(self.positions)

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

        if not initial_state == "empty":
            self._setup_initial_pieces()

    # Backward compatibility properties
    @property
    def board_state(self) -> dict[tuple[float, float], PieceType]:
        """Backward compatibility: access to board state."""
        return self.piece_tracker.board

    @board_state.setter
    def board_state(self, board_state: dict[tuple[float, float], PieceType]):
        """Backward compatibility: set board state."""
        self.piece_tracker.set_board_state(board_state)

    @property
    def p1_pieces(self) -> set[tuple[float, float]]:
        """Backward compatibility: Player 1 pieces."""
        return self.piece_tracker.get_player_pieces(Player.P1)

    @property
    def p2_pieces(self) -> set[tuple[float, float]]:
        """Backward compatibility: Player 2 pieces."""
        return self.piece_tracker.get_player_pieces(Player.P2)

    def _get_cached_board_data(
        self,
    ) -> tuple[
        dict[tuple[float, float], list[tuple[float, float]]], list[tuple[float, float]]
    ]:
        """Get cached board graph and sorted positions."""
        cache_key = (self.config.board_rows, self.config.board_cols)

        if cache_key not in self._board_cache:
            # Create board graph and sorted positions once
            nodes = self._create_board_graph()
            positions = sorted(nodes.keys())
            self._board_cache[cache_key] = (nodes, positions)

        return self._board_cache[cache_key]

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
        """Set up the initial game position from the config with optimized batch operations."""
        self.piece_tracker.clear_all()

        # Generate cache key from config
        config_key = self._get_config_key()

        # Check if we have cached piece setup for this config
        if config_key not in self._setup_cache:
            # Batch add all pieces to minimize function call overhead
            pieces_to_add = []

            # Collect all P1 pieces
            for piece_type, positions in self.config.p1_setup.items():
                for pos in positions:
                    pieces_to_add.append((pos, piece_type))

            # Collect all P2 pieces
            for piece_type, positions in self.config.p2_setup.items():
                for pos in positions:
                    pieces_to_add.append((pos, piece_type))

            # Cache the setup for future use
            self._setup_cache[config_key] = pieces_to_add
        else:
            pieces_to_add = self._setup_cache[config_key]

        # Add all pieces in one batch operation
        self._batch_add_pieces(pieces_to_add)

    def _get_config_key(self) -> str:
        """Generate a unique key for the current config setup."""
        # Create a hash of the config setup for caching
        config_data = []
        for piece_type, positions in self.config.p1_setup.items():
            config_data.append((piece_type.value, tuple(sorted(positions))))
        for piece_type, positions in self.config.p2_setup.items():
            config_data.append((piece_type.value, tuple(sorted(positions))))

        return str(hash(tuple(config_data)))

    def _batch_add_pieces(self, pieces: list[tuple[tuple[float, float], PieceType]]):
        """Add multiple pieces efficiently in batch operation."""
        for pos, piece_type in pieces:
            # Direct access to tracker internals for speed
            self.piece_tracker.board[pos] = piece_type

            if piece_type.value > 0:  # Player 1 piece
                self.piece_tracker.positions_dict[Player.P1].add(pos)
                if piece_type == PieceType.P1_KING:
                    self.piece_tracker.kings[Player.P1] = pos
            elif piece_type.value < 0:  # Player 2 piece
                self.piece_tracker.positions_dict[Player.P2].add(pos)
                if piece_type == PieceType.P2_KING:
                    self.piece_tracker.kings[Player.P2] = pos

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
        if pos not in self.piece_tracker.board:
            return False

        return self.piece_tracker.add_piece(pos, piece_type)

    def remove_piece(self, pos: tuple[float, float]) -> PieceType:
        """
        Remove a piece from the board at the specified position.

        Args:
            pos: Position to remove piece from

        Returns:
            The piece type that was removed, or PieceType.EMPTY if no piece was there
        """
        return self.piece_tracker.remove_piece(pos)

    def clear_board(self):
        """Clear all pieces from the board."""
        self.piece_tracker.clear_all()

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
        piece: PieceType | None = self.piece_tracker.get_piece_at(pos)
        if not piece or (piece.value > 0) != (self.current_player > 0):
            return []

        valid_moves = []
        is_p1 = self.current_player == Player.P1

        for neighbor_pos in self.nodes.get(pos, []):
            target_piece = self.piece_tracker.get_piece_at(neighbor_pos)

            if target_piece == PieceType.EMPTY:
                row_diff = neighbor_pos[0] - pos[0]
                is_forward = row_diff < 0 if is_p1 else row_diff > 0

                if is_forward:
                    valid_moves.append(Move(from_pos=pos, to_pos=neighbor_pos))
                continue  # Move to the next neighbor

            is_opponent = piece.value * target_piece.value < 0
            if is_opponent:
                if not can_capture(piece, target_piece):
                    continue

                row_diff = neighbor_pos[0] - pos[0]
                col_diff = neighbor_pos[1] - pos[1]
                landing_pos = (neighbor_pos[0] + row_diff, neighbor_pos[1] + col_diff)

                # Check if landing position exists on board and is empty
                if (
                    landing_pos in self.piece_tracker.board
                    and self.piece_tracker.get_piece_at(landing_pos) == PieceType.EMPTY
                ):
                    valid_moves.append(
                        Move(
                            from_pos=pos,
                            to_pos=landing_pos,
                            capture_pos=neighbor_pos,
                        )
                    )

        return valid_moves

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
        return self.piece_tracker.get_king_position(player)

    def make_move(
        self, move: Move, _is_validated: bool = False
    ) -> tuple[bool, dict[str, Any]]:
        """Execute a move and return success status and move info"""
        if self.game_over:
            return False, {"error": "Game is over"}

        if not _is_validated:
            valid_moves = self.get_all_valid_moves()
            if move not in valid_moves:
                return False, {"error": "Invalid move", "move": move}

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
        self._end_turn_and_check_for_winner()

        return True, move_info

    def has_any_valid_move(self) -> bool:
        """Efficiently check if any valid move exists for the current player."""
        if self.capture_sequence and self.capturing_piece:
            # Check for the first capture from the required piece
            for move in self.get_valid_moves(self.capturing_piece):
                if move.is_capture:
                    return True
            return False

        player_pieces = (
            self.p1_pieces if self.current_player == Player.P1 else self.p2_pieces
        )
        for pos in player_pieces:
            # If get_valid_moves finds even one move for this piece, we're done
            if self.get_valid_moves(pos):
                return True

        return False

    def _end_turn_and_check_for_winner(self):
        """Check if game has ended and determine winner"""
        winner, reason = GameRules.check_win_condition(game=self)

        if winner:
            self.end_game(winner, reason)

    def end_game(self, winner: Player, reason: str):
        """Manually end the game with specified winner and reason"""
        self.game_over = True
        self.winner = winner
        self.win_reason = reason

    def make_random_move(self) -> tuple[bool, dict[str, Any]]:
        """Make a random valid move (for AI opponent) - optimized to avoid generating all moves"""
        if self.capture_sequence and self.capturing_piece:
            # During capture sequence, check the capturing piece for chain captures
            capturing_piece_moves = self.get_valid_moves(self.capturing_piece)
            chain_captures = [move for move in capturing_piece_moves if move.is_capture]

            if chain_captures:
                return self.make_move(random.choice(chain_captures), _is_validated=True)
            # If no chain captures, fall back to all moves (since captures are optional)

        player_pieces = list(self.piece_tracker.get_player_pieces(self.current_player))
        if not player_pieces:
            return False, {"error": "No pieces available"}

        # Shuffle pieces to try them in random order
        random.shuffle(player_pieces)

        for pos in player_pieces:
            # Generate moves for this piece only
            piece_moves = self.get_valid_moves(pos)

            # If this piece has valid moves, pick one randomly and execute it
            if piece_moves:
                return self.make_move(random.choice(piece_moves), _is_validated=True)

        # If we get here, no pieces have valid moves
        return False, {"error": "No valid moves"}

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
                piece = self.piece_tracker.get_piece_at(pos)
                row_str += get_piece_symbol(piece) if piece is not None else "   "
            lines.append(row_str)

        if self.game_over:
            if self.winner:
                lines.append(f"\n🎉 Game Over! Winner: {self.winner.name}")
            else:
                lines.append("\n🎉 Game Over! Result: Draw")

        lines.append("-" * 45)
        return "\n".join(lines)
