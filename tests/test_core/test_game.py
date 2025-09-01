"""Tests for core DabloGame functionality"""

import pytest

from dablo.core.game import DabloGame
from dablo.core.pieces import PieceType
from dablo.core.player import Player
from dablo.core.rules import Move, WinReason
from tests.conftest import assert_game_state_consistent


@pytest.mark.unit
@pytest.mark.game_mechanics
class TestDabloGameInit:
    """Test game initialization"""

    def test_fresh_game_has_correct_initial_state(self, game: DabloGame):
        """Test that a new game starts with correct state"""
        assert game.current_player == Player.P1
        assert not game.game_over
        assert game.winner is None
        assert game.win_reason is None
        assert game.move_count == 0
        assert not game.capture_sequence
        assert game.capturing_piece is None

    def test_empty_game_has_no_pieces(self, empty_game: DabloGame):
        """Test that empty game has no pieces on board"""
        piece_count = sum(
            1 for p in empty_game.board_state.values() if p != PieceType.EMPTY
        )
        assert piece_count == 0
        assert len(empty_game.p1_pieces) == 0
        assert len(empty_game.p2_pieces) == 0

    def test_board_graph_creation(self, game: DabloGame):
        """Test that board graph is created correctly"""
        assert len(game.positions) == 50  # Dablo has 50 positions
        assert len(game.nodes) == 50
        assert len(game.board_state) == 50

        # All positions should be in nodes
        for pos in game.positions:
            assert pos in game.nodes
            assert pos in game.board_state

    def test_initial_piece_setup(self, game: DabloGame):
        """Test that pieces are set up correctly initially"""
        p1_count = sum(1 for p in game.board_state.values() if p > 0)
        p2_count = sum(1 for p in game.board_state.values() if p < 0)

        assert p1_count > 0, "P1 should have pieces"
        assert p2_count > 0, "P2 should have pieces"
        assert p1_count == p2_count, "Both players should have same number of pieces"

        assert_game_state_consistent(game)


@pytest.mark.unit
@pytest.mark.game_mechanics
class TestGameState:
    """Test game state management"""

    def test_player_switching(self, game: DabloGame):
        """Test that players switch correctly"""
        assert game.current_player == Player.P1

        # Get a valid move and make it
        moves = game.get_all_valid_moves()
        if moves:
            move = moves[0]
            game.make_move(move)
            assert game.current_player == Player.P2

    def test_piece_tracking_consistency(self, game: DabloGame, game_utils):
        """Test that piece sets stay consistent with board state"""
        assert_game_state_consistent(game)

        # Count pieces both ways
        p1_board_count = game_utils.player_piece_count(game, Player.P1)
        p2_board_count = game_utils.player_piece_count(game, Player.P2)

        assert len(game.p1_pieces) == p1_board_count
        assert len(game.p2_pieces) == p2_board_count

    def test_move_count_increments(self, game: DabloGame):
        """Test that move count increments correctly"""
        initial_count = game.move_count
        moves = game.get_all_valid_moves()

        if moves:
            game.make_move(moves[0])
            assert game.move_count == initial_count + 1


@pytest.mark.unit
@pytest.mark.movement
class TestMoveGeneration:
    """Test move generation"""

    def test_get_all_valid_moves_returns_list(self, game: DabloGame):
        """Test that move generation returns a list"""
        moves = game.get_all_valid_moves()
        assert isinstance(moves, list)
        assert len(moves) > 0, "Initial game should have valid moves"

    def test_moves_are_valid_for_current_player(self, game: DabloGame):
        """Test that all generated moves are valid for current player"""
        moves = game.get_all_valid_moves()

        for move in moves:
            piece = game.board_state[move.from_pos]
            if game.current_player == Player.P1:
                assert piece > 0, f"P1 move should use P1 piece, got {piece}"
            else:
                assert piece < 0, f"P2 move should use P2 piece, got {piece}"

    def test_no_moves_to_occupied_positions(self, game: DabloGame):
        """Test that moves don't target occupied positions (except captures)"""
        moves = game.get_all_valid_moves()

        for move in moves:
            target_piece = game.board_state[move.to_pos]
            if not move.is_capture:
                assert target_piece == PieceType.EMPTY, (
                    f"Non-capture move to occupied position: {move}"
                )

    def test_empty_game_has_no_moves(self, empty_game: DabloGame):
        """Test that empty game has no valid moves"""
        moves = empty_game.get_all_valid_moves()
        assert len(moves) == 0


@pytest.mark.integration
@pytest.mark.movement
class TestMoveExecution:
    """Test move execution"""

    def test_valid_move_execution(self, game: DabloGame):
        """Test that valid moves can be executed"""
        moves = game.get_all_valid_moves()
        if not moves:
            pytest.skip("No moves available")

        move = moves[0]
        initial_player = game.current_player

        success, result = game.make_move(move)

        assert success, f"Valid move should succeed: {result}"
        assert game.current_player != initial_player, "Player should switch after move"
        assert_game_state_consistent(game)

    def test_invalid_move_fails(self, game: DabloGame):
        """Test that invalid moves are rejected"""
        # Create an invalid move (empty to empty)
        empty_positions = [
            pos for pos, piece in game.board_state.items() if piece == PieceType.EMPTY
        ]
        if len(empty_positions) < 2:
            pytest.skip("Need empty positions for test")

        invalid_move = Move(from_pos=empty_positions[0], to_pos=empty_positions[1])
        success, result = game.make_move(invalid_move)

        assert not success, "Invalid move should fail"
        assert isinstance(result, dict), "Should return error info"

    def test_move_updates_board_state(self, simple_capture_game: DabloGame):
        """Test that moves properly update board state"""
        game = simple_capture_game

        # Find a regular move (non-capture)
        moves = [m for m in game.get_all_valid_moves() if not m.is_capture]
        if not moves:
            pytest.skip("No regular moves available")

        move = moves[0]
        original_piece = game.board_state[move.from_pos]

        game.make_move(move)

        # Check piece moved correctly
        assert game.board_state[move.from_pos] == PieceType.EMPTY
        assert game.board_state[move.to_pos] == original_piece

    def test_capture_move_execution(self, simple_capture_game: DabloGame):
        """Test that capture moves work correctly"""
        game = simple_capture_game

        # Find capture moves
        capture_moves = [m for m in game.get_all_valid_moves() if m.is_capture]
        if not capture_moves:
            pytest.skip("No capture moves available in test scenario")

        move = capture_moves[0]
        attacker_piece = game.board_state[move.from_pos]
        captured_piece = game.board_state[move.capture_pos]

        success, result = game.make_move(move)

        assert success, "Capture move should succeed"
        assert game.board_state[move.from_pos] == PieceType.EMPTY
        assert (
            game.board_state[move.capture_pos] == PieceType.EMPTY
        )  # Captured piece removed
        assert game.board_state[move.to_pos] == attacker_piece  # Attacker moved

        assert_game_state_consistent(game)

    @pytest.mark.capture
    def test_optional_chain_capture_sequence(
        self, optional_chain_capture_scenario: DabloGame
    ):
        """
        Tests that after a capture, the player can choose between another
        capture or a regular move with the SAME piece.
        """
        game = optional_chain_capture_scenario

        # Define the first capture move
        first_capture = Move(
            from_pos=(2.0, 2.0), to_pos=(3.0, 3.0), capture_pos=(2.5, 2.5)
        )

        # Execute the first capture
        success, info = game.make_move(first_capture)

        assert success, "The first capture should be successful."

        # 1. Verify the turn has NOT ended
        assert game.current_player == Player.P1, (
            "Player should not change during a chain sequence."
        )
        assert game.capture_sequence is True, "Game should be in a capture sequence."
        assert game.capturing_piece == (3.0, 3.0), (
            "Capturing piece's new position should be tracked."
        )

        # 2. Get the available moves AFTER the first capture
        available_moves = game.get_all_valid_moves()

        # 3. Verify the player is correctly restricted
        # All available moves MUST originate from the piece that just captured.
        for move in available_moves:
            assert move.from_pos == (3.0, 3.0), "Only the capturing piece can be moved."

        # 4. Verify the player has the correct CHOICES
        # The list of moves should contain BOTH the next capture AND any simple moves.
        expected_capture_move = Move(
            from_pos=(3.0, 3.0), to_pos=(4.0, 4.0), capture_pos=(3.5, 3.5)
        )
        expected_simple_move = Move(
            from_pos=(3.0, 3.0), to_pos=(2.5, 3.5)
        )  # Example simple move

        assert any(move == expected_capture_move for move in available_moves), (
            "The second capture move should be an option."
        )
        assert any(move == expected_simple_move for move in available_moves), (
            "A simple move should also be an option."
        )


@pytest.mark.unit
@pytest.mark.endgame
class TestGameEnding:
    """Test game ending conditions"""

    def test_game_not_over_initially(self, game: DabloGame):
        """Test that game is not over initially"""
        assert not game.game_over
        assert game.winner is None
        assert game.win_reason is None

    def test_lone_king_detection(self, endgame_scenario: DabloGame):
        """Test detection of lone king condition"""
        game = endgame_scenario

        # Test that we can detect king positions
        p1_king_pos = game.get_king_position(Player.P1)
        p2_king_pos = game.get_king_position(Player.P2)

        assert p1_king_pos is not None, "P1 should have a king"
        assert p2_king_pos is not None, "P2 should have a king"

        # Verify kings are correct pieces
        assert game.board_state[p1_king_pos] == PieceType.P1_KING
        assert game.board_state[p2_king_pos] == PieceType.P2_KING

    def test_win_by_annihilation(self, empty_game: DabloGame):
        """Test that P1 wins if P2 has no pieces left."""
        empty_game.add_piece((1.0, 1.0), PieceType.P1_KING)
        empty_game.add_piece((1.5, 1.5), PieceType.P2_WARRIOR)  # P2's last piece

        capture_move = Move(
            from_pos=(1.0, 1.0), to_pos=(2.0, 2.0), capture_pos=(1.5, 1.5)
        )

        # Assume the move is valid for this test setup
        empty_game.make_move(capture_move)

        assert empty_game.game_over is True
        assert empty_game.winner == Player.P1
        assert empty_game.win_reason == WinReason.KING_CAPTURE

    @pytest.mark.endgame
    def test_win_by_stalemate_player_blocked(self, empty_game: DabloGame):
        """Test P1 wins if all P2 pieces are completely blocked by P1 pieces."""
        # We'll place a single P2 warrior at (0.5, 0.5) and surround it.
        # Its neighbors are (0.0, 0.0), (0.0, 1.0), (1.0, 0.0), and (1.0, 1.0).
        pieces = {
            # P2 pieces - both king and warrior blocked
            (0.0, 0.0): PieceType.P2_KING,  # P2 king at corner
            (0.5, 0.5): PieceType.P2_WARRIOR,  # P2 warrior blocked
            # P1 blocking pieces - surround P2 pieces completely
            (1.0, 0.0): PieceType.P1_WARRIOR,
            (0.0, 1.0): PieceType.P1_WARRIOR,
            (1.0, 1.0): PieceType.P1_WARRIOR,
            (1.5, 0.5): PieceType.P1_WARRIOR,
            (0.5, 1.5): PieceType.P1_WARRIOR,
            (1.5, 1.5): PieceType.P1_WARRIOR,
            (2.0, 0.0): PieceType.P1_WARRIOR,
            (0.0, 2.0): PieceType.P1_WARRIOR,
            # P1 king far away
            (3.0, 4.0): PieceType.P1_KING,
        }
        empty_game.setup_custom_position(pieces)

        # Set the turn to P2, who now has no legal moves.
        empty_game.current_player = Player.P2

        # First, confirm that the game logic finds no valid moves for P2.
        assert not empty_game.get_all_valid_moves()
        # Manually trigger the end-of-turn check to evaluate the game state.
        empty_game._end_turn_and_check_for_winner()

        print(empty_game.winner, empty_game.win_reason)
        assert empty_game.game_over is True, "The game should be over."
        assert empty_game.winner == Player.P1, "P1 should be the winner."
        assert empty_game.win_reason == WinReason.STALEMATE, (
            "Win reason should be stalemate."
        )

    @pytest.mark.endgame
    def test_win_by_stalemate_pawns_at_edge(self, empty_game: DabloGame):
        """Test P1 wins if P2's only warrior is at the end of the board."""
        # Assuming P2 moves from higher row numbers towards row 0.
        # A P2 warrior on row 0 has no more forward moves.
        pieces = {
            # The stuck P2 pawns
            (5.0, 0.0): PieceType.P2_WARRIOR,
            (5.0, 1.0): PieceType.P2_KING,
            (3.0, 4.0): PieceType.P1_KING,
            (4.0, 4.0): PieceType.P1_WARRIOR,
        }
        empty_game.setup_custom_position(pieces)
        # Set the turn to P2.
        empty_game.current_player = Player.P2

        # Verify that the game finds no moves for the stuck warrior.
        assert not empty_game.get_all_valid_moves()

        # Trigger the end-of-turn check.
        empty_game._end_turn_and_check_for_winner()

        assert empty_game.game_over is True, "The game should be over."
        assert empty_game.winner == Player.P1, "P1 should be the winner."
        assert empty_game.win_reason == WinReason.STALEMATE, (
            "Win reason should be stalemate."
        )


@pytest.mark.integration
@pytest.mark.game_mechanics
class TestGameIntegration:
    """Integration tests for complete game functionality"""

    def test_complete_game_flow(self, game: DabloGame):
        """Test a complete game flow with multiple moves"""
        max_moves = 20  # Prevent infinite loops
        moves_made = 0

        while not game.game_over and moves_made < max_moves:
            moves = game.get_all_valid_moves()
            if not moves:
                break

            # Make a random move
            move = moves[0]  # Use first available move
            success, result = game.make_move(move)

            if success:
                moves_made += 1
                assert_game_state_consistent(game)
            else:
                break

        assert moves_made > 0, "Should have made at least one move"

    def test_board_state_always_consistent(self, game: DabloGame):
        """Test that board state remains consistent throughout game"""
        moves_to_test = 5
        moves_made = 0

        while not game.game_over and moves_made < moves_to_test:
            assert_game_state_consistent(game)

            moves = game.get_all_valid_moves()
            if not moves:
                break

            game.make_move(moves[0])
            moves_made += 1

        assert_game_state_consistent(game)
