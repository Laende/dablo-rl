"""Tests for player enumeration"""

import pytest

from dablo.core.player import Player


@pytest.mark.unit
class TestPlayer:
    """Test Player enumeration"""

    def test_player_values(self):
        """Test that players have correct values"""
        assert Player.P1.value == 1
        assert Player.P2.value == -1

    def test_player_comparison(self):
        """Test player comparison operations"""
        assert Player.P1 != Player.P2
        assert Player.P1 == Player.P1
        assert Player.P2 == Player.P2

    def test_player_conversion_to_int(self):
        """Test that players can be converted to integers"""
        assert int(Player.P1) == 1
        assert int(Player.P2) == -1

    def test_player_boolean_context(self):
        """Test players in boolean contexts"""
        # Both players should be truthy
        assert Player.P1
        assert Player.P2

    def test_player_string_representation(self):
        """Test player string conversion"""
        p1_str = str(Player.P1)
        p2_str = str(Player.P2)

        assert isinstance(p1_str, str)
        assert isinstance(p2_str, str)
        assert p1_str != p2_str
        assert "P1" in p1_str or "1" in p1_str
        assert "P2" in p2_str or "2" in p2_str or "-1" in p2_str


@pytest.mark.unit
class TestPlayerIntegration:
    """Test player integration with game logic"""

    def test_player_piece_identification(self, player):
        """Test identifying pieces belonging to players"""
        from dablo.core.pieces import PieceType

        if player == Player.P1:
            # P1 pieces have positive values
            assert PieceType.P1_WARRIOR.value > 0
            assert PieceType.P1_PRINCE.value > 0
            assert PieceType.P1_KING.value > 0
        else:  # Player.P2
            # P2 pieces have negative values
            assert PieceType.P2_WARRIOR.value < 0
            assert PieceType.P2_PRINCE.value < 0
            assert PieceType.P2_KING.value < 0

    def test_player_piece_ownership(self, game, player):
        """Test that players own correct pieces in game"""
        if player == Player.P1:
            player_pieces = game.p1_pieces
            expected_sign = 1
        else:
            player_pieces = game.p2_pieces
            expected_sign = -1

        # Check that all pieces in player's set have correct sign
        for pos in player_pieces:
            piece = game.board_state[pos]
            if expected_sign > 0:
                assert piece.value > 0, (
                    f"P1 should only have positive pieces, found {piece} at {pos}"
                )
            else:
                assert piece.value < 0, (
                    f"P2 should only have negative pieces, found {piece} at {pos}"
                )

    def test_current_player_switching(self, game):
        """Test that current player switches correctly"""
        initial_player = game.current_player

        # Get and make a move
        moves = game.get_all_valid_moves()
        if moves:
            success, _ = game.make_move(moves[0])
            if success:
                # Player should have switched
                assert game.current_player != initial_player

                # Should be the other player
                if initial_player == Player.P1:
                    assert game.current_player == Player.P2
                else:
                    assert game.current_player == Player.P1
