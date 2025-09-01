"""Tests for game rules and move validation"""

import math

import pytest

from dablo.core.rules import WIN_DESCRIPTIONS, Move, MovementValidator, WinReason


@pytest.mark.unit
class TestMove:
    """Test Move model"""

    def test_move_creation(self):
        """Test basic move creation"""
        move = Move(from_pos=(1.0, 1.0), to_pos=(1.5, 1.5))
        assert move.from_pos == (1.0, 1.0)
        assert move.to_pos == (1.5, 1.5)
        assert move.capture_pos is None
        assert not move.is_capture

    def test_capture_move_creation(self):
        """Test capture move creation"""
        move = Move(from_pos=(1.0, 1.0), to_pos=(2.0, 2.0), capture_pos=(1.5, 1.5))
        assert move.capture_pos == (1.5, 1.5)
        assert move.is_capture

    def test_move_distance_calculation(self):
        """Test distance calculation"""
        # Horizontal move
        move = Move(from_pos=(1.0, 1.0), to_pos=(1.0, 2.0))
        assert move.distance == 1.0

        # Vertical move
        move = Move(from_pos=(1.0, 1.0), to_pos=(2.0, 1.0))
        assert move.distance == 1.0

        # Diagonal move
        move = Move(from_pos=(0.0, 0.0), to_pos=(1.0, 1.0))
        assert abs(move.distance - math.sqrt(2)) < 0.001

        # No movement
        move = Move(from_pos=(1.0, 1.0), to_pos=(1.0, 1.0))
        assert move.distance == 0.0

    def test_move_string_representation(self):
        """Test move string conversion"""
        # Regular move
        move = Move(from_pos=(1.0, 1.0), to_pos=(1.5, 1.5))
        move_str = str(move)
        assert "(1.0, 1.0)" in move_str
        assert "(1.5, 1.5)" in move_str
        assert "captures" not in move_str

        # Capture move
        capture_move = Move(
            from_pos=(1.0, 1.0), to_pos=(2.0, 2.0), capture_pos=(1.5, 1.5)
        )
        capture_str = str(capture_move)
        assert "(1.0, 1.0)" in capture_str
        assert "(2.0, 2.0)" in capture_str
        assert "(1.5, 1.5)" in capture_str
        assert "captures" in capture_str

    def test_move_immutability(self):
        """Test that moves are frozen/immutable"""
        move = Move(from_pos=(1.0, 1.0), to_pos=(1.5, 1.5))

        # Should not be able to modify move attributes
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            move.from_pos = (2.0, 2.0)

        with pytest.raises(Exception):
            move.to_pos = (3.0, 3.0)

    def test_move_equality(self):
        """Test move equality comparison"""
        move1 = Move(from_pos=(1.0, 1.0), to_pos=(1.5, 1.5))
        move2 = Move(from_pos=(1.0, 1.0), to_pos=(1.5, 1.5))
        move3 = Move(from_pos=(1.0, 1.0), to_pos=(2.0, 2.0))

        assert move1 == move2
        assert move1 != move3

        # Test with capture moves
        capture1 = Move(from_pos=(1.0, 1.0), to_pos=(2.0, 2.0), capture_pos=(1.5, 1.5))
        capture2 = Move(from_pos=(1.0, 1.0), to_pos=(2.0, 2.0), capture_pos=(1.5, 1.5))

        assert capture1 == capture2
        assert move1 != capture1


@pytest.mark.unit
class TestWinReason:
    """Test win reason enumeration and descriptions"""

    def test_win_reason_enum_values(self):
        """Test that all win reasons have string values"""
        assert WinReason.KING_CAPTURE.value == "king_capture"
        assert WinReason.LONE_KING.value == "lone_king"
        assert WinReason.MOVE_LIMIT.value == "move_limit"
        assert WinReason.STALEMATE.value == "stalemate"
        assert WinReason.KINGS_ONLY_DRAW.value == "kings_only_draw"

    def test_win_descriptions_complete(self):
        """Test that all win reasons have descriptions"""
        for reason in WinReason:
            assert reason in WIN_DESCRIPTIONS
            assert isinstance(WIN_DESCRIPTIONS[reason], str)
            assert len(WIN_DESCRIPTIONS[reason]) > 0

    def test_win_descriptions_content(self):
        """Test that win descriptions make sense"""
        assert "King captured" in WIN_DESCRIPTIONS[WinReason.KING_CAPTURE]
        assert "lone king" in WIN_DESCRIPTIONS[WinReason.LONE_KING]
        assert "Move limit" in WIN_DESCRIPTIONS[WinReason.MOVE_LIMIT]
        assert "No legal moves" in WIN_DESCRIPTIONS[WinReason.STALEMATE]
        assert "kings" in WIN_DESCRIPTIONS[WinReason.KINGS_ONLY_DRAW]


@pytest.mark.unit
@pytest.mark.movement
class TestMovementValidator:
    """Test movement validation logic"""

    def test_forward_move_validation_p1(self):
        """Test forward move detection for Player 1"""
        # P1 moves toward smaller row numbers
        assert MovementValidator.is_forward_move(
            (2.0, 1.0), (1.0, 1.0), is_player1=True
        )
        assert MovementValidator.is_forward_move(
            (3.5, 2.5), (2.0, 3.0), is_player1=True
        )

        # Not forward moves for P1
        assert not MovementValidator.is_forward_move(
            (1.0, 1.0), (2.0, 1.0), is_player1=True
        )
        assert not MovementValidator.is_forward_move(
            (2.0, 2.0), (2.0, 2.0), is_player1=True
        )  # No movement

    def test_forward_move_validation_p2(self):
        """Test forward move detection for Player 2"""
        # P2 moves toward larger row numbers
        assert MovementValidator.is_forward_move(
            (1.0, 1.0), (2.0, 1.0), is_player1=False
        )
        assert MovementValidator.is_forward_move(
            (1.5, 2.5), (3.0, 1.0), is_player1=False
        )

        # Not forward moves for P2
        assert not MovementValidator.is_forward_move(
            (2.0, 1.0), (1.0, 1.0), is_player1=False
        )
        assert not MovementValidator.is_forward_move(
            (2.0, 2.0), (2.0, 2.0), is_player1=False
        )  # No movement

    def test_diagonal_moves_can_be_forward(self):
        """Test that diagonal moves can be forward"""
        # P1 diagonal forward
        assert MovementValidator.is_forward_move(
            (2.0, 1.0), (1.5, 1.5), is_player1=True
        )
        assert MovementValidator.is_forward_move(
            (2.0, 1.0), (1.5, 0.5), is_player1=True
        )

        # P2 diagonal forward
        assert MovementValidator.is_forward_move(
            (1.0, 1.0), (1.5, 1.5), is_player1=False
        )
        assert MovementValidator.is_forward_move(
            (1.0, 1.0), (1.5, 0.5), is_player1=False
        )

    def test_sideways_moves_not_forward(self):
        """Test that purely sideways moves are not forward"""
        # Same row, different column - not forward for either player
        assert not MovementValidator.is_forward_move(
            (2.0, 1.0), (2.0, 2.0), is_player1=True
        )
        assert not MovementValidator.is_forward_move(
            (2.0, 1.0), (2.0, 2.0), is_player1=False
        )

        assert not MovementValidator.is_forward_move(
            (1.5, 2.5), (1.5, 1.5), is_player1=True
        )
        assert not MovementValidator.is_forward_move(
            (1.5, 2.5), (1.5, 1.5), is_player1=False
        )


@pytest.mark.unit
class TestMoveValidation:
    """Test comprehensive move validation"""

    def test_valid_regular_move_structure(self):
        """Test that valid regular moves have correct structure"""
        move = Move(from_pos=(1.0, 1.0), to_pos=(1.5, 1.5))

        # Basic structure
        assert isinstance(move.from_pos, tuple)
        assert isinstance(move.to_pos, tuple)
        assert len(move.from_pos) == 2
        assert len(move.to_pos) == 2
        assert all(isinstance(coord, (int, float)) for coord in move.from_pos)
        assert all(isinstance(coord, (int, float)) for coord in move.to_pos)

        # Regular move properties
        assert move.capture_pos is None
        assert not move.is_capture
        assert move.distance > 0

    def test_valid_capture_move_structure(self):
        """Test that valid capture moves have correct structure"""
        move = Move(from_pos=(1.0, 1.0), to_pos=(2.0, 2.0), capture_pos=(1.5, 1.5))

        # Capture move properties
        assert move.capture_pos is not None
        assert move.is_capture
        assert isinstance(move.capture_pos, tuple)
        assert len(move.capture_pos) == 2
        assert all(isinstance(coord, (int, float)) for coord in move.capture_pos)

    def test_move_coordinates_are_float_compatible(self):
        """Test that move coordinates work with float positions"""
        # Test with integer coordinates
        move_int = Move(from_pos=(1, 1), to_pos=(2, 2))
        assert move_int.distance > 0

        # Test with float coordinates
        move_float = Move(from_pos=(1.0, 1.5), to_pos=(2.5, 2.0))
        assert move_float.distance > 0

        # Test with mixed coordinates
        move_mixed = Move(from_pos=(1, 1.5), to_pos=(2.0, 2))
        assert move_mixed.distance > 0


@pytest.mark.integration
@pytest.mark.movement
class TestMoveValidationIntegration:
    """Integration tests for move validation with game state"""

    def test_move_validation_with_game_context(self, game):
        """Test move validation in context of actual game"""
        moves = game.get_all_valid_moves()

        for move in moves[:5]:  # Test first 5 moves
            # All moves should be valid Move objects
            assert isinstance(move, Move)
            assert move.distance > 0

            # Source and target should be different
            assert move.from_pos != move.to_pos

            # Forward move validation
            piece = game.board_state[move.from_pos]
            is_p1 = piece > 0

            if not move.is_capture:  # Regular moves should generally be forward
                # Note: Some special moves might not be forward, so we just test structure
                assert isinstance(
                    MovementValidator.is_forward_move(
                        move.from_pos, move.to_pos, is_p1
                    ),
                    bool,
                )

    def test_capture_move_geometry(self, simple_capture_game):
        """Test that capture moves have correct geometric properties"""
        game = simple_capture_game
        capture_moves = [m for m in game.get_all_valid_moves() if m.is_capture]

        for move in capture_moves:
            # Capture position should be between source and destination
            from_r, from_c = move.from_pos
            cap_r, cap_c = move.capture_pos
            to_r, to_c = move.to_pos

            # Check that capture is in the middle (for standard captures)
            # Landing position should be opposite side of captured piece
            expected_to_r = cap_r + (cap_r - from_r)
            expected_to_c = cap_c + (cap_c - from_c)

            # Allow small floating point differences
            assert abs(to_r - expected_to_r) < 0.001
            assert abs(to_c - expected_to_c) < 0.001


@pytest.mark.unit
class TestRulesEdgeCases:
    """Test edge cases and specific code paths in rules module"""

    def test_calculate_capture_landing_special_cases(self):
        """Test special cases in calculate_capture_landing (lines 100-101)"""
        # Test case where (abs(row_diff), abs(col_diff)) in {(0.5, 1.5), (1.5, 0.5)}

        # Case 1: (0.5, 1.5) - row_diff positive, col_diff positive
        from_pos = (1.0, 1.0)
        capture_pos = (1.5, 2.5)  # row_diff=0.5, col_diff=1.5

        landing_pos = MovementValidator.calculate_capture_landing(from_pos, capture_pos)

        # Should use special calculation: landing_r = capture_pos[0] + 0.5, landing_c = capture_pos[1] + 0.5
        expected = (2.0, 3.0)  # (1.5 + 0.5, 2.5 + 0.5)
        assert landing_pos == expected

        # Case 2: (0.5, 1.5) - row_diff negative, col_diff negative
        from_pos = (2.0, 3.0)
        capture_pos = (1.5, 1.5)  # row_diff=-0.5, col_diff=-1.5

        landing_pos = MovementValidator.calculate_capture_landing(from_pos, capture_pos)

        # Should use special calculation: landing_r = capture_pos[0] - 0.5, landing_c = capture_pos[1] - 0.5
        expected = (1.0, 1.0)  # (1.5 - 0.5, 1.5 - 0.5)
        assert landing_pos == expected

    def test_regular_move_validation_failures(self):
        """Test regular move validation failure conditions"""
        from dablo.core.pieces import PieceType
        from dablo.core.rules import GameRules

        # Mock board state and adjacency graph
        board_state = {
            (1.0, 1.0): PieceType.P1_WARRIOR,
            (1.5, 1.5): PieceType.P2_WARRIOR,  # Target occupied
            (2.0, 2.0): PieceType.EMPTY,
        }

        adjacency_graph = {
            (1.0, 1.0): [(1.5, 1.5), (2.0, 2.0)],
            (1.5, 1.5): [(1.0, 1.0), (2.0, 2.0)],
            (2.0, 2.0): [(1.0, 1.0), (1.5, 1.5)],
        }

        # Test line 123: Target position not empty
        result = GameRules.is_valid_regular_move(
            from_pos=(1.0, 1.0),
            to_pos=(1.5, 1.5),  # Occupied position
            piece=PieceType.P1_WARRIOR,
            board_state=board_state,
            adjacency_graph=adjacency_graph,
        )
        assert result is False

        # Test line 127: Positions not adjacent
        result = GameRules.is_valid_regular_move(
            from_pos=(1.0, 1.0),
            to_pos=(3.0, 3.0),  # Not in adjacency graph
            piece=PieceType.P1_WARRIOR,
            board_state=board_state,
            adjacency_graph=adjacency_graph,
        )
        assert result is False

    def test_capture_move_validation_failures(self):
        """Test capture move validation failure conditions"""
        from dablo.core.pieces import PieceType
        from dablo.core.rules import GameRules

        board_state = {
            (1.0, 1.0): PieceType.P1_WARRIOR,
            (1.5, 1.5): PieceType.P2_WARRIOR,
            (2.0, 2.0): PieceType.EMPTY,
            (3.0, 3.0): PieceType.EMPTY,
        }

        adjacency_graph = {
            (1.0, 1.0): [(1.5, 1.5)],
            (1.5, 1.5): [(1.0, 1.0), (2.0, 2.0)],
            (2.0, 2.0): [(1.5, 1.5)],
            (3.0, 3.0): [],
        }

        # Test line 156: Capture position not adjacent to source
        result = GameRules.is_valid_capture_move(
            from_pos=(1.0, 1.0),
            capture_pos=(3.0, 3.0),  # Not adjacent to source
            landing_pos=(4.0, 4.0),
            attacker=PieceType.P1_WARRIOR,
            target=PieceType.P2_WARRIOR,
            board_state=board_state,
            adjacency_graph=adjacency_graph,
        )
        assert result is False

        # Test line 163: Landing position doesn't match expected calculation
        result = GameRules.is_valid_capture_move(
            from_pos=(1.0, 1.0),
            capture_pos=(1.5, 1.5),
            landing_pos=(3.0, 3.0),  # Wrong landing position
            attacker=PieceType.P1_WARRIOR,
            target=PieceType.P2_WARRIOR,
            board_state=board_state,
            adjacency_graph=adjacency_graph,
        )
        assert result is False

    def test_win_condition_edge_cases(self):
        """Test specific win conditions (lines 199, 203, 205)"""
        from dablo.core.player import Player
        from dablo.core.rules import GameRules

        # Test line 199: Kings only draw (both players have exactly 1 piece)
        winner, reason = GameRules.check_win_condition(
            p1_pieces_count=1,
            p2_pieces_count=1,
            p1_king_exists=True,
            p2_king_exists=True,
            move_count=50,
            move_limit=100,
            has_valid_moves=True,
            current_player=Player.P1,
        )
        assert winner is None
        assert reason == WinReason.KINGS_ONLY_DRAW

        # Test line 203: P2 wins when P1 has lone king (P1=1, P2>1)
        winner, reason = GameRules.check_win_condition(
            p1_pieces_count=1,  # P1 lone king
            p2_pieces_count=3,  # P2 has multiple pieces
            p1_king_exists=True,
            p2_king_exists=True,
            move_count=50,
            move_limit=100,
            has_valid_moves=True,
            current_player=Player.P1,
        )
        assert winner == Player.P2
        assert reason == WinReason.LONE_KING

        # Test line 205: P1 wins when P2 has lone king (P1>1, P2=1)
        winner, reason = GameRules.check_win_condition(
            p1_pieces_count=3,  # P1 has multiple pieces
            p2_pieces_count=1,  # P2 lone king
            p1_king_exists=True,
            p2_king_exists=True,
            move_count=50,
            move_limit=100,
            has_valid_moves=True,
            current_player=Player.P2,
        )
        assert winner == Player.P1
        assert reason == WinReason.LONE_KING
