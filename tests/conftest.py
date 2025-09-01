"""Pytest configuration and fixtures for Dablo tests

Provides clean, focused fixtures for testing core game functionality.
"""

import pytest

from dablo.core.config import DabloConfig
from dablo.core.game import DabloGame
from dablo.core.pieces import PieceType
from dablo.core.player import Player
from dablo.core.rules import Move


# Core Fixtures
@pytest.fixture(scope="session")
def game_config() -> DabloConfig:
    """Shared game configuration for all tests"""
    return DabloConfig.create_default()


@pytest.fixture
def game(game_config: DabloConfig) -> DabloGame:
    """Fresh game instance with default setup"""
    return DabloGame(config=game_config)


@pytest.fixture
def empty_game(game_config: DabloConfig) -> DabloGame:
    """Game with empty board for custom setups"""
    return DabloGame(config=game_config, initial_state="empty")


# Game State Fixtures
@pytest.fixture
def simple_capture_game(empty_game: DabloGame) -> DabloGame:
    """Simple capture scenario - P1 warrior can capture P2 warrior"""
    empty_game.add_piece((2.0, 2.0), PieceType.P1_WARRIOR)
    empty_game.add_piece((1.5, 2.5), PieceType.P2_WARRIOR)
    return empty_game


@pytest.fixture
def endgame_scenario(empty_game: DabloGame) -> DabloGame:
    """Endgame with few pieces remaining"""
    pieces = {
        (1.0, 1.0): PieceType.P1_KING,
        (1.5, 1.5): PieceType.P1_WARRIOR,
        (3.0, 3.0): PieceType.P2_KING,
        (3.5, 3.5): PieceType.P2_WARRIOR,
    }

    for pos, piece in pieces.items():
        empty_game.add_piece(pos, piece)

    return empty_game


# Parametrized Fixtures
@pytest.fixture(
    params=[
        PieceType.P1_WARRIOR,
        PieceType.P1_PRINCE,
        PieceType.P1_KING,
        PieceType.P2_WARRIOR,
        PieceType.P2_PRINCE,
        PieceType.P2_KING,
    ]
)
def piece_type(request):
    """All piece types for comprehensive testing"""
    return request.param


@pytest.fixture(params=[Player.P1, Player.P2])
def player(request):
    """Both players for testing"""
    return request.param


# Test Utilities
class GameTestUtils:
    """Utility functions for game testing"""

    @staticmethod
    def count_pieces(game: DabloGame, piece_type: PieceType = None) -> int:
        """Count pieces of a specific type, or all non-empty pieces"""
        if piece_type is not None:
            return sum(1 for p in game.board_state.values() if p == piece_type)
        return sum(1 for p in game.board_state.values() if p != PieceType.EMPTY)

    @staticmethod
    def find_pieces(
        game: DabloGame, piece_type: PieceType
    ) -> list[tuple[float, float]]:
        """Find all positions containing a specific piece type"""
        return [pos for pos, piece in game.board_state.items() if piece == piece_type]

    @staticmethod
    def player_piece_count(game: DabloGame, player: Player) -> int:
        """Count total pieces for a player"""
        if player == Player.P1:
            return sum(1 for p in game.board_state.values() if p > 0)
        return sum(1 for p in game.board_state.values() if p < 0)

    @staticmethod
    def is_valid_board_state(game: DabloGame) -> bool:
        """Validate that board state is consistent"""
        # Check all positions exist in graph
        for pos in game.board_state.keys():
            if pos not in game.nodes:
                return False

        # Check piece sets match board state
        p1_on_board = {pos for pos, piece in game.board_state.items() if piece > 0}
        p2_on_board = {pos for pos, piece in game.board_state.items() if piece < 0}

        return p1_on_board == game.p1_pieces and p2_on_board == game.p2_pieces

    @staticmethod
    def create_test_position(
        game: DabloGame, pieces_dict: dict[tuple[float, float], PieceType]
    ):
        """Helper to create test positions using proper game API"""
        for pos, piece in pieces_dict.items():
            if pos in game.board_state and game.board_state[pos] == PieceType.EMPTY:
                game.add_piece(pos, piece)


@pytest.fixture
def game_utils():
    """Access to game testing utilities"""
    return GameTestUtils


# Test Assertions
def assert_valid_move(move: Move, game: DabloGame, player: Player):
    """Assert that a move is valid for the current game state"""
    # Basic move validation
    assert move.from_pos in game.board_state, (
        f"Source position {move.from_pos} not on board"
    )
    assert move.to_pos in game.board_state, (
        f"Target position {move.to_pos} not on board"
    )

    # Check piece ownership
    piece = game.board_state[move.from_pos]
    if player == Player.P1:
        assert piece > 0, f"Player 1 cannot move piece {piece} at {move.from_pos}"
    else:
        assert piece < 0, f"Player 2 cannot move piece {piece} at {move.from_pos}"

    # Target should be empty or valid capture
    target_piece = game.board_state[move.to_pos]
    if move.is_capture:
        assert move.capture_pos is not None, "Capture move missing capture_pos"
        assert target_piece == PieceType.EMPTY, (
            "Capture target position should be empty"
        )
    else:
        assert target_piece == PieceType.EMPTY, "Regular move target should be empty"


def assert_game_state_consistent(game: DabloGame):
    """Assert that game state is internally consistent"""
    # Board should have all positions
    assert len(game.board_state) == len(game.positions), "Board size mismatch"

    # Piece sets should match board state
    p1_on_board = {pos for pos, piece in game.board_state.items() if piece > 0}
    p2_on_board = {pos for pos, piece in game.board_state.items() if piece < 0}

    assert p1_on_board == game.p1_pieces, "P1 piece set inconsistent with board"
    assert p2_on_board == game.p2_pieces, "P2 piece set inconsistent with board"


# Modern pytest fixtures for better test isolation
@pytest.fixture
def fresh_config() -> DabloConfig:
    """Factory fixture for creating fresh configs"""

    def _create_config(**kwargs):
        return DabloConfig(**kwargs) if kwargs else DabloConfig.create_default()

    return _create_config


@pytest.fixture
def game_factory(fresh_config: DabloConfig) -> DabloGame:
    """Factory fixture for creating games with different configurations"""

    def _create_game(config=None, initial_state="default"):
        return DabloGame(config=config or fresh_config(), initial_state=initial_state)

    return _create_game


# Performance monitoring fixture
@pytest.fixture(autouse=True)
def monitor_test_performance(request):
    """Auto-use fixture to monitor test performance"""
    import time

    start_time = time.time()

    yield

    duration = time.time() - start_time
    if duration > 1.0:  # Log slow tests
        print(f"\nSLOW TEST: {request.node.name} took {duration:.2f}s")


# Error context fixture for better debugging
@pytest.fixture
def capture_game_state():
    """Capture game state for debugging failed tests"""

    def _capture(game: DabloGame, description: str = ""):
        return {
            "description": description,
            "current_player": game.current_player,
            "move_count": game.move_count,
            "game_over": game.game_over,
            "winner": game.winner,
            "p1_piece_count": len(game.p1_pieces),
            "p2_piece_count": len(game.p2_pieces),
            "capture_sequence": game.capture_sequence,
        }

    return _capture


@pytest.fixture
def optional_chain_capture_scenario(empty_game: DabloGame) -> DabloGame:
    """
    Sets up a scenario where a P1 piece makes a capture and then has a choice.
    - P1 Warrior at (2.0, 2.0) captures P2 Warrior at (2.5, 2.5), landing at (3.0, 3.0).
    - From (3.0, 3.0), it can either:
        1. Capture another P2 Warrior at (3.5, 3.5).
        2. Make a simple, non-capturing move to an adjacent empty square like (2.5, 3.5).
    """
    pieces = {
        (2.0, 2.0): PieceType.P1_WARRIOR,
        (2.5, 2.5): PieceType.P2_WARRIOR,
        (3.5, 3.5): PieceType.P2_WARRIOR,
        # Add another P1 piece elsewhere to confirm it can't be moved during the sequence.
        (0.0, 0.0): PieceType.P1_WARRIOR,
    }
    for pos, piece in pieces.items():
        empty_game.add_piece(pos, piece)

    return empty_game
