"""Tests for piece types and capture mechanics"""

import pytest

from dablo.core.pieces import (
    BASE_PIECE_VALUES,
    PIECE_NAMES,
    PIECE_SYMBOLS,
    PieceType,
    can_capture,
    get_piece_name,
    get_piece_rank,
    get_piece_symbol,
    get_piece_value,
)


@pytest.mark.unit
class TestPieceType:
    """Test PieceType enum"""

    def test_piece_type_values(self):
        """Test that piece types have correct integer values"""
        assert PieceType.EMPTY == 0
        assert PieceType.P1_WARRIOR == 1
        assert PieceType.P1_PRINCE == 2
        assert PieceType.P1_KING == 3
        assert PieceType.P2_WARRIOR == -1
        assert PieceType.P2_PRINCE == -2
        assert PieceType.P2_KING == -3

    def test_piece_type_ordering(self):
        """Test that piece types can be compared"""
        assert PieceType.P1_KING > PieceType.P1_PRINCE > PieceType.P1_WARRIOR
        assert PieceType.P2_KING < PieceType.P2_PRINCE < PieceType.P2_WARRIOR
        assert PieceType.P1_WARRIOR > PieceType.EMPTY > PieceType.P2_WARRIOR


@pytest.mark.unit
class TestPieceHelperFunctions:
    """Test piece utility functions"""

    def test_get_piece_symbol(self, piece_type):
        """Test that all piece types have symbols"""
        symbol = get_piece_symbol(piece_type)
        assert isinstance(symbol, str)
        assert len(symbol) > 0
        assert piece_type in PIECE_SYMBOLS

    def test_get_piece_name(self, piece_type):
        """Test that all piece types have names"""
        if piece_type == PieceType.EMPTY:
            name = get_piece_name(piece_type)
            assert name == "Empty"
        else:
            name = get_piece_name(piece_type)
            assert isinstance(name, str)
            assert len(name) > 0
            assert piece_type in PIECE_NAMES

    def test_get_piece_rank(self, piece_type):
        """Test piece rank calculation"""
        rank = get_piece_rank(piece_type)
        assert isinstance(rank, int)
        assert rank >= 0
        assert rank <= 3
        assert rank == abs(piece_type.value)

    def test_get_piece_value(self, piece_type):
        """Test piece value calculation"""
        value = get_piece_value(piece_type)
        assert isinstance(value, float)
        assert value >= 0.0

        expected_value = BASE_PIECE_VALUES.get(abs(piece_type.value), 0.0)
        assert value == expected_value

    def test_unknown_piece_symbol(self):
        """Test handling of unknown piece types"""
        # Test with invalid enum value
        symbol = get_piece_symbol(99)  # Invalid piece type
        assert symbol == " ? "

    def test_unknown_piece_name(self):
        """Test handling of unknown piece types"""
        name = get_piece_name(99)  # Invalid piece type
        assert name == "Unknown"


@pytest.mark.unit
@pytest.mark.capture
class TestCaptureRules:
    """Test piece capture mechanics"""

    def test_warrior_capture_rules(self):
        """Test warrior capture capabilities"""
        # Warriors can capture warriors
        assert can_capture(PieceType.P1_WARRIOR, PieceType.P2_WARRIOR)
        assert can_capture(PieceType.P2_WARRIOR, PieceType.P1_WARRIOR)

        # Warriors cannot capture princes or kings
        assert not can_capture(PieceType.P1_WARRIOR, PieceType.P2_PRINCE)
        assert not can_capture(PieceType.P1_WARRIOR, PieceType.P2_KING)
        assert not can_capture(PieceType.P2_WARRIOR, PieceType.P1_PRINCE)
        assert not can_capture(PieceType.P2_WARRIOR, PieceType.P1_KING)

    def test_prince_capture_rules(self):
        """Test prince capture capabilities"""
        # Princes can capture warriors and princes
        assert can_capture(PieceType.P1_PRINCE, PieceType.P2_WARRIOR)
        assert can_capture(PieceType.P1_PRINCE, PieceType.P2_PRINCE)
        assert can_capture(PieceType.P2_PRINCE, PieceType.P1_WARRIOR)
        assert can_capture(PieceType.P2_PRINCE, PieceType.P1_PRINCE)

        # Princes cannot capture kings
        assert not can_capture(PieceType.P1_PRINCE, PieceType.P2_KING)
        assert not can_capture(PieceType.P2_PRINCE, PieceType.P1_KING)

    def test_king_capture_rules(self):
        """Test king capture capabilities"""
        # Kings can capture any piece
        assert can_capture(PieceType.P1_KING, PieceType.P2_WARRIOR)
        assert can_capture(PieceType.P1_KING, PieceType.P2_PRINCE)
        assert can_capture(PieceType.P1_KING, PieceType.P2_KING)
        assert can_capture(PieceType.P2_KING, PieceType.P1_WARRIOR)
        assert can_capture(PieceType.P2_KING, PieceType.P1_PRINCE)
        assert can_capture(PieceType.P2_KING, PieceType.P1_KING)

    def test_same_player_cannot_capture(self):
        """Test that pieces cannot capture their own pieces"""
        # P1 pieces cannot capture P1 pieces
        assert not can_capture(PieceType.P1_WARRIOR, PieceType.P1_WARRIOR)
        assert not can_capture(PieceType.P1_WARRIOR, PieceType.P1_PRINCE)
        assert not can_capture(PieceType.P1_WARRIOR, PieceType.P1_KING)
        assert not can_capture(PieceType.P1_PRINCE, PieceType.P1_WARRIOR)
        assert not can_capture(PieceType.P1_KING, PieceType.P1_PRINCE)

        # P2 pieces cannot capture P2 pieces
        assert not can_capture(PieceType.P2_WARRIOR, PieceType.P2_WARRIOR)
        assert not can_capture(PieceType.P2_WARRIOR, PieceType.P2_PRINCE)
        assert not can_capture(PieceType.P2_WARRIOR, PieceType.P2_KING)
        assert not can_capture(PieceType.P2_PRINCE, PieceType.P2_WARRIOR)
        assert not can_capture(PieceType.P2_KING, PieceType.P2_PRINCE)

    def test_empty_pieces_cannot_capture(self):
        """Test that empty pieces cannot capture or be captured"""
        for piece_type in [
            PieceType.P1_WARRIOR,
            PieceType.P1_PRINCE,
            PieceType.P1_KING,
            PieceType.P2_WARRIOR,
            PieceType.P2_PRINCE,
            PieceType.P2_KING,
        ]:
            # Empty cannot capture anything
            assert not can_capture(PieceType.EMPTY, piece_type)
            # Nothing can capture empty
            assert not can_capture(piece_type, PieceType.EMPTY)

        # Empty cannot capture empty
        assert not can_capture(PieceType.EMPTY, PieceType.EMPTY)

    @pytest.mark.parametrize(
        "attacker,target,expected",
        [
            (PieceType.P1_WARRIOR, PieceType.P2_WARRIOR, True),
            (PieceType.P1_WARRIOR, PieceType.P2_PRINCE, False),
            (PieceType.P1_PRINCE, PieceType.P2_WARRIOR, True),
            (PieceType.P1_PRINCE, PieceType.P2_PRINCE, True),
            (PieceType.P1_KING, PieceType.P2_KING, True),
            (PieceType.P2_KING, PieceType.P1_WARRIOR, True),
        ],
    )
    def test_capture_matrix(self, attacker, target, expected):
        """Test specific capture combinations"""
        result = can_capture(attacker, target)
        assert result == expected, f"{attacker} capturing {target} should be {expected}"


@pytest.mark.unit
class TestPieceValues:
    """Test piece value system"""

    def test_base_piece_values_exist(self):
        """Test that all ranks have values"""
        for rank in range(4):  # 0-3
            assert rank in BASE_PIECE_VALUES
            assert isinstance(BASE_PIECE_VALUES[rank], float)

    def test_piece_value_hierarchy(self):
        """Test that higher rank pieces have higher values"""
        warrior_value = get_piece_value(PieceType.P1_WARRIOR)
        prince_value = get_piece_value(PieceType.P1_PRINCE)
        king_value = get_piece_value(PieceType.P1_KING)
        empty_value = get_piece_value(PieceType.EMPTY)

        assert empty_value < warrior_value < prince_value < king_value

    def test_same_rank_same_value(self):
        """Test that pieces of same rank have same value"""
        assert get_piece_value(PieceType.P1_WARRIOR) == get_piece_value(
            PieceType.P2_WARRIOR
        )
        assert get_piece_value(PieceType.P1_PRINCE) == get_piece_value(
            PieceType.P2_PRINCE
        )
        assert get_piece_value(PieceType.P1_KING) == get_piece_value(PieceType.P2_KING)


@pytest.mark.unit
class TestPieceConstants:
    """Test piece constant dictionaries"""

    def test_piece_symbols_complete(self):
        """Test that all piece types have symbols"""
        all_pieces = [
            PieceType.EMPTY,
            PieceType.P1_WARRIOR,
            PieceType.P1_PRINCE,
            PieceType.P1_KING,
            PieceType.P2_WARRIOR,
            PieceType.P2_PRINCE,
            PieceType.P2_KING,
        ]

        for piece in all_pieces:
            assert piece in PIECE_SYMBOLS
            assert isinstance(PIECE_SYMBOLS[piece], str)
            assert len(PIECE_SYMBOLS[piece]) > 0

    def test_piece_names_complete(self):
        """Test that all non-empty piece types have names"""
        non_empty_pieces = [
            PieceType.P1_WARRIOR,
            PieceType.P1_PRINCE,
            PieceType.P1_KING,
            PieceType.P2_WARRIOR,
            PieceType.P2_PRINCE,
            PieceType.P2_KING,
            PieceType.EMPTY,
        ]

        for piece in non_empty_pieces:
            assert piece in PIECE_NAMES
            assert isinstance(PIECE_NAMES[piece], str)
            assert len(PIECE_NAMES[piece]) > 0

    def test_base_piece_values_complete(self):
        """Test that all ranks have base values"""
        for rank in range(4):  # 0 (empty) through 3 (king)
            assert rank in BASE_PIECE_VALUES
            assert isinstance(BASE_PIECE_VALUES[rank], int | float)
            assert BASE_PIECE_VALUES[rank] >= 0
