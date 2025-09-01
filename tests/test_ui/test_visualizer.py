"""Tests for game visualizer"""

from unittest.mock import patch

import matplotlib
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import pytest


# Use a non-interactive backend for testing
matplotlib.use("Agg")

from dablo.core.game import DabloGame
from dablo.core.pieces import PieceType
from dablo.core.player import Player
from dablo.ui.visualizer import DabloVisualizer, demo_visualization, quick_visualize


@pytest.mark.ui
@pytest.mark.unit
class TestDabloVisualizer:
    """Test DabloVisualizer class"""

    def test_visualizer_initialization(self):
        """Test visualizer can be initialized"""
        visualizer = DabloVisualizer()

        assert visualizer.figsize == (10, 8)
        assert isinstance(visualizer.colors, dict)
        assert isinstance(visualizer.piece_symbols, dict)
        assert isinstance(visualizer.piece_sizes, dict)

        # Test with custom figsize
        custom_visualizer = DabloVisualizer(figsize=(12, 10))
        assert custom_visualizer.figsize == (12, 10)

    def test_visualizer_has_all_piece_colors(self):
        """Test that visualizer has colors for all piece types"""
        visualizer = DabloVisualizer()

        piece_types = [
            PieceType.P1_WARRIOR,
            PieceType.P1_PRINCE,
            PieceType.P1_KING,
            PieceType.P2_WARRIOR,
            PieceType.P2_PRINCE,
            PieceType.P2_KING,
        ]

        for piece_type in piece_types:
            assert piece_type in visualizer.colors
            assert piece_type in visualizer.piece_symbols
            assert piece_type in visualizer.piece_sizes

    def test_visualize_game_basic(self, game: DabloGame):
        """Test basic game visualization"""
        visualizer = DabloVisualizer()

        fig = visualizer.visualize_game(game)

        assert fig is not None
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_visualize_game_with_title(self, game: DabloGame):
        """Test visualization with custom title"""
        visualizer = DabloVisualizer()

        fig = visualizer.visualize_game(game, title="Custom Title")

        assert fig is not None
        plt.close(fig)

    def test_visualize_game_with_highlights(self, game: DabloGame):
        """Test visualization with move highlights"""
        visualizer = DabloVisualizer()

        fig = visualizer.visualize_game(game, highlight_moves=True)

        assert fig is not None
        plt.close(fig)

    @patch("matplotlib.pyplot.savefig")
    def test_visualize_game_with_save(self, mock_savefig, game: DabloGame):
        """Test visualization with save functionality"""
        visualizer = DabloVisualizer()

        fig = visualizer.visualize_game(game, save_path="test_output.png")

        assert fig is not None
        mock_savefig.assert_called_once_with(
            "test_output.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

    def test_draw_board_grid(self, game: DabloGame):
        """Test board grid drawing"""
        visualizer = DabloVisualizer()
        fig, ax = plt.subplots()

        # Should not raise exception
        visualizer._draw_board_grid(ax, game)

        # Check that patches were added
        assert len(ax.patches) > 0
        plt.close(fig)

    def test_draw_pieces(self, game: DabloGame):
        """Test piece drawing"""
        visualizer = DabloVisualizer()
        fig, ax = plt.subplots()

        # Should not raise exception
        visualizer._draw_pieces(ax, game)

        # Check that pieces were drawn
        assert len(ax.patches) > 0
        assert len(ax.texts) > 0
        plt.close(fig)

    def test_draw_pieces_empty_board(self, empty_game: DabloGame):
        """Test piece drawing on empty board"""
        visualizer = DabloVisualizer()
        fig, ax = plt.subplots()

        # Should not raise exception even with empty board
        visualizer._draw_pieces(ax, empty_game)

        # Should have no pieces drawn
        assert len([p for p in ax.patches if isinstance(p, Circle)]) == 0
        assert len(ax.texts) == 0
        plt.close(fig)

    def test_highlight_valid_moves(self, game: DabloGame):
        """Test valid move highlighting"""
        visualizer = DabloVisualizer()
        fig, ax = plt.subplots()

        # Should not raise exception
        visualizer._highlight_valid_moves(ax, game)

        # Should add highlighting patches
        assert len(ax.patches) > 0
        plt.close(fig)

    def test_add_game_info_basic(self, game: DabloGame):
        """Test basic game info display"""
        visualizer = DabloVisualizer()
        fig, ax = plt.subplots()

        visualizer._add_game_info(ax, game, None)

        # Should add text
        assert len(ax.texts) > 0
        plt.close(fig)

    def test_add_game_info_with_title(self, game: DabloGame):
        """Test game info with custom title"""
        visualizer = DabloVisualizer()
        fig, ax = plt.subplots()

        visualizer._add_game_info(ax, game, "Custom Title")

        # Should add text
        assert len(ax.texts) > 0
        plt.close(fig)

    def test_add_game_info_game_over(self, empty_game: DabloGame):
        """Test game info when game is over"""
        visualizer = DabloVisualizer()
        fig, ax = plt.subplots()

        # Simulate game over
        empty_game.game_over = True
        empty_game.winner = Player.P1

        visualizer._add_game_info(ax, empty_game, None)

        # Should add text
        assert len(ax.texts) > 0
        plt.close(fig)

    def test_add_game_info_chain_capture(self, game: DabloGame):
        """Test game info during chain capture"""
        visualizer = DabloVisualizer()
        fig, ax = plt.subplots()

        # Simulate chain capture
        game.capture_sequence = True

        visualizer._add_game_info(ax, game, None)

        # Should add text
        assert len(ax.texts) > 0
        plt.close(fig)

    def test_style_plot(self):
        """Test plot styling"""
        visualizer = DabloVisualizer()
        fig, ax = plt.subplots()

        # Should not raise exception
        visualizer._style_plot(ax)

        # Check that styling was applied
        # Aspect is set to 'equal' but matplotlib may return 1.0
        assert ax.get_aspect() == 1.0 or ax.get_aspect() == "equal"
        assert ax.get_xlim() == (-0.5, 4.5)
        assert ax.get_ylim() == (-5.5, 0.5)
        plt.close(fig)

    def test_animate_move(self, game: DabloGame):
        """Test move animation"""
        visualizer = DabloVisualizer()

        moves = game.get_all_valid_moves()
        if moves:
            with patch("matplotlib.pyplot.show"):
                # Should not raise exception
                visualizer.animate_move(game, moves[0])


@pytest.mark.ui
@pytest.mark.unit
class TestQuickVisualize:
    """Test quick visualization functions"""

    @patch("matplotlib.pyplot.show")
    def test_quick_visualize_show_true(self, mock_show, game: DabloGame):
        """Test quick visualize with show=True"""
        fig = quick_visualize(game, show=True)

        assert fig is not None
        mock_show.assert_called_once()
        plt.close(fig)

    def test_quick_visualize_show_false(self, game: DabloGame):
        """Test quick visualize with show=False"""
        fig = quick_visualize(game, show=False)

        assert fig is not None
        plt.close(fig)

    @patch("matplotlib.pyplot.savefig")
    def test_quick_visualize_with_save(self, mock_savefig, game: DabloGame):
        """Test quick visualize with save path"""
        fig = quick_visualize(game, show=False, save_path="test.png")

        assert fig is not None
        mock_savefig.assert_called_once()
        plt.close(fig)


@pytest.mark.ui
@pytest.mark.integration
class TestVisualizerIntegration:
    """Integration tests for visualizer"""

    def test_visualize_capture_scenario(self, simple_capture_game: DabloGame):
        """Test visualizing capture scenario"""
        visualizer = DabloVisualizer()

        fig = visualizer.visualize_game(simple_capture_game, highlight_moves=True)

        assert fig is not None
        plt.close(fig)

    def test_demo_visualization(self):
        """Test demo visualization function"""
        with patch("matplotlib.pyplot.show"):
            with patch.object(DabloGame, "make_move", return_value=(True, {})):
                # Should not raise exception
                demo_visualization()

    def test_visualize_different_game_states(self):
        """Test visualizing games in different states"""
        visualizer = DabloVisualizer()

        # Test fresh game
        fresh_game = DabloGame()
        fig1 = visualizer.visualize_game(fresh_game)
        assert fig1 is not None
        plt.close(fig1)

        # Test empty game
        empty_game = DabloGame(initial_state="empty")
        fig2 = visualizer.visualize_game(empty_game)
        assert fig2 is not None
        plt.close(fig2)


@pytest.mark.ui
class TestVisualizerEdgeCases:
    """Test edge cases and error conditions"""

    def test_visualize_with_unknown_piece_symbol(self, empty_game: DabloGame):
        """Test handling of unknown piece types"""
        visualizer = DabloVisualizer()

        # Add a mock piece type that doesn't have a symbol
        empty_game.board_state[(1.0, 1.0)] = PieceType.EMPTY  # Use EMPTY as unknown

        fig, ax = plt.subplots()

        # Should handle gracefully
        visualizer._draw_pieces(ax, empty_game)
        plt.close(fig)

    def test_visualize_malformed_board_state(self, empty_game: DabloGame):
        """Test visualization with malformed board state"""
        visualizer = DabloVisualizer()

        # Create malformed board state
        empty_game.board_state = {}  # Empty board state

        fig = visualizer.visualize_game(empty_game)

        assert fig is not None
        plt.close(fig)

    def test_visualize_no_valid_moves(self, empty_game: DabloGame):
        """Test visualization when no moves available"""
        visualizer = DabloVisualizer()

        # Empty game has no valid moves
        fig = visualizer.visualize_game(empty_game, highlight_moves=True)

        assert fig is not None
        plt.close(fig)
