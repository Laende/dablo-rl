"""Tests for interactive Dablo game"""

from unittest.mock import MagicMock, Mock, patch

import matplotlib
import pytest


# Use a non-interactive backend for testing
matplotlib.use("Agg")

from dablo.core.pieces import PieceType
from dablo.core.player import Player
from dablo.core.rules import Move
from dablo.ui.interactive import InteractiveDabloGame, play_interactive_dablo


@pytest.mark.ui
@pytest.mark.unit
class TestInteractiveDabloGame:
    """Test InteractiveDabloGame class"""

    def test_initialization_vs_npc(self):
        """Test initialization with NPC opponent"""
        interactive_game = InteractiveDabloGame(vs_npc=True, npc_difficulty="medium")

        assert interactive_game.vs_npc is True
        assert interactive_game.npc_difficulty == "medium"
        assert interactive_game.game is not None
        assert interactive_game.visualizer is not None
        assert interactive_game.selected_piece is None
        assert interactive_game.valid_moves_for_piece == []
        assert interactive_game.game_history == []
        assert interactive_game.npc_player is not None

    def test_initialization_vs_human(self):
        """Test initialization with human opponent"""
        interactive_game = InteractiveDabloGame(vs_npc=False)

        assert interactive_game.vs_npc is False
        assert interactive_game.npc_player is None

    @patch("matplotlib.pyplot.show")
    @patch("builtins.print")
    def test_start_game(self, mock_print, mock_show):
        """Test starting the interactive game"""
        interactive_game = InteractiveDabloGame(vs_npc=False)

        # Mock the figure and canvas
        mock_fig = MagicMock()
        mock_canvas = MagicMock()
        mock_fig.canvas = mock_canvas

        with patch("matplotlib.pyplot.subplots", return_value=(mock_fig, MagicMock())):
            interactive_game.start_game()

        # Should print game instructions
        assert mock_print.call_count >= 3

        # Should connect event handlers
        mock_canvas.mpl_connect.assert_any_call(
            "button_press_event", interactive_game._on_click
        )
        mock_canvas.mpl_connect.assert_any_call(
            "key_press_event", interactive_game._on_key
        )

    def test_screen_to_game_pos_valid(self):
        """Test screen to game position conversion with valid coordinates"""
        interactive_game = InteractiveDabloGame(vs_npc=False)

        # Mock game positions
        interactive_game.game.positions = [(1.0, 1.0), (1.5, 1.5), (2.0, 2.0)]

        # Should find closest position
        result = interactive_game._screen_to_game_pos(1.1, -1.1)
        assert result == (1.0, 1.0)

    def test_screen_to_game_pos_none(self):
        """Test screen to game position conversion with None coordinates"""
        interactive_game = InteractiveDabloGame(vs_npc=False)

        result = interactive_game._screen_to_game_pos(None, 1.0)
        assert result is None

        result = interactive_game._screen_to_game_pos(1.0, None)
        assert result is None

    def test_screen_to_game_pos_too_far(self):
        """Test screen to game position conversion with coordinates too far from any position"""
        interactive_game = InteractiveDabloGame(vs_npc=False)

        # Mock game positions
        interactive_game.game.positions = [(1.0, 1.0), (1.5, 1.5)]

        # Should return None if too far from any position
        result = interactive_game._screen_to_game_pos(10.0, -10.0)
        assert result is None

    @patch("builtins.print")
    def test_try_select_piece_valid_p1(self, mock_print):
        """Test selecting valid P1 piece"""
        interactive_game = InteractiveDabloGame(vs_npc=False)

        # Set up game state
        pos = (2.0, 2.0)
        interactive_game.game.board_state[pos] = PieceType.P1_WARRIOR
        interactive_game.game.current_player = Player.P1

        # Mock valid moves
        mock_moves = [Move(from_pos=pos, to_pos=(3.0, 3.0))]
        with patch.object(
            interactive_game.game, "get_all_valid_moves", return_value=mock_moves
        ):
            with patch.object(interactive_game, "_update_display"):
                interactive_game._try_select_piece(pos)

        assert interactive_game.selected_piece == pos
        assert len(interactive_game.valid_moves_for_piece) == 1
        mock_print.assert_called()

    @patch("builtins.print")
    def test_try_select_piece_valid_p2_vs_human(self, mock_print):
        """Test selecting valid P2 piece when vs human"""
        interactive_game = InteractiveDabloGame(vs_npc=False)

        # Set up game state
        pos = (2.0, 2.0)
        interactive_game.game.board_state[pos] = PieceType.P2_WARRIOR
        interactive_game.game.current_player = Player.P2

        # Mock valid moves
        mock_moves = [Move(from_pos=pos, to_pos=(3.0, 3.0))]
        with patch.object(
            interactive_game.game, "get_all_valid_moves", return_value=mock_moves
        ):
            with patch.object(interactive_game, "_update_display"):
                interactive_game._try_select_piece(pos)

        assert interactive_game.selected_piece == pos
        mock_print.assert_called()

    @patch("builtins.print")
    def test_try_select_piece_p2_vs_npc(self, mock_print):
        """Test trying to select P2 piece when vs NPC"""
        interactive_game = InteractiveDabloGame(vs_npc=True)

        # Set up game state
        pos = (2.0, 2.0)
        interactive_game.game.board_state[pos] = PieceType.P2_WARRIOR
        interactive_game.game.current_player = Player.P2

        interactive_game._try_select_piece(pos)

        # Should not select piece and print warning
        assert interactive_game.selected_piece is None
        mock_print.assert_called_with("It's the NPC's turn, please wait...")

    @patch("builtins.print")
    def test_try_select_piece_invalid(self, mock_print):
        """Test trying to select invalid piece"""
        interactive_game = InteractiveDabloGame(vs_npc=False)

        # Try to select empty position
        pos = (2.0, 2.0)
        interactive_game.game.board_state[pos] = PieceType.EMPTY
        interactive_game.game.current_player = Player.P1

        interactive_game._try_select_piece(pos)

        assert interactive_game.selected_piece is None
        mock_print.assert_called_with("Cannot select that piece!")

    def test_try_make_move_no_valid_moves(self):
        """Test trying to make move when no valid moves available"""
        interactive_game = InteractiveDabloGame(vs_npc=False)
        interactive_game.valid_moves_for_piece = []

        with patch.object(interactive_game, "_deselect_piece") as mock_deselect:
            interactive_game._try_make_move((3.0, 3.0))

        mock_deselect.assert_called_once()

    def test_try_make_move_valid(self):
        """Test making valid move"""
        interactive_game = InteractiveDabloGame(vs_npc=False)

        # Set up valid move
        target_pos = (3.0, 3.0)
        move = Move(from_pos=(2.0, 2.0), to_pos=target_pos)
        interactive_game.valid_moves_for_piece = [move]

        with patch.object(interactive_game, "_execute_move") as mock_execute:
            interactive_game._try_make_move(target_pos)

        mock_execute.assert_called_once_with(move)

    @patch("builtins.print")
    def test_try_make_move_invalid(self, mock_print):
        """Test trying to make invalid move"""
        interactive_game = InteractiveDabloGame(vs_npc=False)

        # Set up move that doesn't match target position
        move = Move(from_pos=(2.0, 2.0), to_pos=(3.0, 3.0))
        interactive_game.valid_moves_for_piece = [move]

        with patch.object(interactive_game, "_deselect_piece") as mock_deselect:
            interactive_game._try_make_move((4.0, 4.0))  # Different position

        mock_print.assert_called_with("Invalid move to (4.0, 4.0)")
        mock_deselect.assert_called_once()

    @patch("builtins.print")
    def test_execute_move_success_simple(self, mock_print):
        """Test executing successful simple move"""
        interactive_game = InteractiveDabloGame(vs_npc=False)

        move = Move(from_pos=(2.0, 2.0), to_pos=(3.0, 3.0))
        move_info = {"is_capture": False, "chain_capture_available": False}

        with patch.object(interactive_game, "_save_game_state", return_value={}):
            with patch.object(
                interactive_game.game, "make_move", return_value=(True, move_info)
            ):
                with patch.object(interactive_game, "_deselect_piece"):
                    with patch.object(interactive_game, "_update_display"):
                        interactive_game._execute_move(move)

        mock_print.assert_any_call("Moving from (2.0, 2.0) to (3.0, 3.0)")

    @patch("builtins.print")
    def test_execute_move_capture(self, mock_print):
        """Test executing capture move"""
        interactive_game = InteractiveDabloGame(vs_npc=False)

        move = Move(from_pos=(2.0, 2.0), to_pos=(3.0, 3.0), capture_pos=(2.5, 2.5))
        move_info = {
            "is_capture": True,
            "captured_piece": PieceType.P2_WARRIOR,
            "chain_capture_available": False,
        }

        with patch.object(interactive_game, "_save_game_state", return_value={}):
            with patch.object(
                interactive_game.game, "make_move", return_value=(True, move_info)
            ):
                with patch.object(interactive_game, "_deselect_piece"):
                    with patch.object(interactive_game, "_update_display"):
                        interactive_game._execute_move(move)

        mock_print.assert_any_call("Captured P2_WARRIOR!")

    @patch("builtins.print")
    def test_execute_move_chain_capture(self, mock_print):
        """Test executing move with chain capture available"""
        interactive_game = InteractiveDabloGame(vs_npc=False)

        move = Move(from_pos=(2.0, 2.0), to_pos=(3.0, 3.0), capture_pos=(2.5, 2.5))
        move_info = {
            "is_capture": True,
            "captured_piece": PieceType.P2_WARRIOR,
            "chain_capture_available": True,
        }

        # Mock additional moves for chain capture
        chain_moves = [Move(from_pos=(3.0, 3.0), to_pos=(4.0, 4.0))]

        with patch.object(interactive_game, "_save_game_state", return_value={}):
            with patch.object(
                interactive_game.game, "make_move", return_value=(True, move_info)
            ):
                with patch.object(
                    interactive_game.game,
                    "get_all_valid_moves",
                    return_value=chain_moves,
                ):
                    with patch.object(interactive_game, "_update_display"):
                        interactive_game._execute_move(move)

        mock_print.assert_any_call(
            "Chain capture available! Select the piece again to continue."
        )
        assert len(interactive_game.valid_moves_for_piece) == 1

    @patch("builtins.print")
    def test_execute_move_failure(self, mock_print):
        """Test executing failed move"""
        interactive_game = InteractiveDabloGame(vs_npc=False)

        move = Move(from_pos=(2.0, 2.0), to_pos=(3.0, 3.0))

        with patch.object(interactive_game, "_save_game_state", return_value={}):
            with patch.object(
                interactive_game.game, "make_move", return_value=(False, {})
            ):
                with patch.object(interactive_game, "_deselect_piece") as mock_deselect:
                    interactive_game._execute_move(move)

        mock_print.assert_called_with("Move failed!")
        mock_deselect.assert_called_once()

    @patch("builtins.print")
    @patch("matplotlib.pyplot.pause")
    def test_make_npc_move(self, mock_pause, mock_print):
        """Test NPC making a move"""
        interactive_game = InteractiveDabloGame(vs_npc=True)

        # Mock NPC player
        mock_npc = MagicMock()
        move = Move(from_pos=(2.0, 2.0), to_pos=(3.0, 3.0))
        mock_npc.get_move.return_value = move
        interactive_game.npc_player = mock_npc

        # Mock figure canvas
        interactive_game.fig = MagicMock()

        move_info = {"is_capture": False, "chain_capture_available": False}

        with patch.object(
            interactive_game.game, "make_move", return_value=(True, move_info)
        ):
            with patch.object(interactive_game, "_update_display"):
                interactive_game._make_npc_move()

        mock_print.assert_any_call("NPC is thinking...")
        mock_print.assert_any_call("NPC moves from (2.0, 2.0) to (3.0, 3.0)")

    def test_make_npc_move_no_npc_player(self):
        """Test NPC move when no NPC player exists"""
        interactive_game = InteractiveDabloGame(vs_npc=False)
        interactive_game.npc_player = None

        # Should return early without error
        interactive_game._make_npc_move()

    def test_deselect_piece(self):
        """Test deselecting piece"""
        interactive_game = InteractiveDabloGame(vs_npc=False)

        # Set some selection state
        interactive_game.selected_piece = (2.0, 2.0)
        interactive_game.valid_moves_for_piece = [
            Move(from_pos=(2.0, 2.0), to_pos=(3.0, 3.0))
        ]

        with patch.object(interactive_game, "_update_display"):
            interactive_game._deselect_piece()

        assert interactive_game.selected_piece is None
        assert interactive_game.valid_moves_for_piece == []

    def test_save_and_load_game_state(self):
        """Test saving and loading game state"""
        interactive_game = InteractiveDabloGame(vs_npc=False)

        # Set up game state
        interactive_game.game.current_player = Player.P2
        interactive_game.game.move_count = 10
        interactive_game.game.capture_sequence = True

        # Save state
        saved_state = interactive_game._save_game_state()

        assert saved_state["current_player"] == Player.P2
        assert saved_state["move_count"] == 10
        assert saved_state["capture_sequence"] is True

        # Modify game state
        interactive_game.game.current_player = Player.P1
        interactive_game.game.move_count = 15

        # Load saved state
        interactive_game._load_game_state(saved_state)

        assert interactive_game.game.current_player == Player.P2
        assert interactive_game.game.move_count == 10

    @patch("builtins.print")
    def test_undo_move_success(self, mock_print):
        """Test successful undo move"""
        interactive_game = InteractiveDabloGame(vs_npc=False)

        # Add history
        saved_state = {"current_player": Player.P1, "move_count": 5}
        interactive_game.game_history = [saved_state]

        with patch.object(interactive_game, "_load_game_state"):
            with patch.object(interactive_game, "_deselect_piece"):
                interactive_game._undo_move()

        mock_print.assert_called_with("Move undone!")
        assert len(interactive_game.game_history) == 0

    @patch("builtins.print")
    def test_undo_move_no_history(self, mock_print):
        """Test undo move with no history"""
        interactive_game = InteractiveDabloGame(vs_npc=False)
        interactive_game.game_history = []

        interactive_game._undo_move()

        mock_print.assert_called_with("Nothing to undo!")

    @patch("builtins.print")
    def test_restart_game(self, mock_print):
        """Test restarting game"""
        interactive_game = InteractiveDabloGame(vs_npc=False)

        # Set some state
        interactive_game.selected_piece = (2.0, 2.0)
        interactive_game.game_history = [{}]

        with patch.object(interactive_game, "_update_display"):
            interactive_game._restart_game()

        assert interactive_game.selected_piece is None
        assert interactive_game.game_history == []
        mock_print.assert_called_with("Game restarted!")

    @patch("builtins.print")
    def test_show_help(self, mock_print):
        """Test showing help"""
        interactive_game = InteractiveDabloGame(vs_npc=False)

        interactive_game._show_help()

        mock_print.assert_called_once()
        # Check that help text contains key information
        help_text = mock_print.call_args[0][0]
        assert "HOW TO PLAY" in help_text
        assert "CONTROLS" in help_text

    def test_on_key_events(self):
        """Test keyboard event handling"""
        interactive_game = InteractiveDabloGame(vs_npc=False)

        # Mock event objects
        escape_event = Mock(key="escape")
        r_event = Mock(key="r")
        u_event = Mock(key="u")
        h_event = Mock(key="h")

        with patch.object(interactive_game, "_deselect_piece") as mock_deselect:
            interactive_game._on_key(escape_event)
            mock_deselect.assert_called_once()

        with patch.object(interactive_game, "_restart_game") as mock_restart:
            interactive_game._on_key(r_event)
            mock_restart.assert_called_once()

        with patch.object(interactive_game, "_undo_move") as mock_undo:
            interactive_game._on_key(u_event)
            mock_undo.assert_called_once()

        with patch.object(interactive_game, "_show_help") as mock_help:
            interactive_game._on_key(h_event)
            mock_help.assert_called_once()

    def test_on_click_outside_axes(self):
        """Test click event outside axes"""
        interactive_game = InteractiveDabloGame(vs_npc=False)
        interactive_game.ax = Mock()

        # Mock event outside axes
        event = Mock(inaxes=None)

        # Should return early without doing anything
        result = interactive_game._on_click(event)
        assert result is None

    def test_on_click_valid_position(self):
        """Test click event with valid position"""
        interactive_game = InteractiveDabloGame(vs_npc=False)
        interactive_game.ax = Mock()

        # Mock event
        event = Mock(inaxes=interactive_game.ax, xdata=1.0, ydata=-2.0)

        with patch.object(
            interactive_game, "_screen_to_game_pos", return_value=(2.0, 1.0)
        ):
            with patch.object(interactive_game, "_try_select_piece") as mock_select:
                interactive_game._on_click(event)

        mock_select.assert_called_once_with((2.0, 1.0))

    def test_on_click_with_selected_piece(self):
        """Test click event when piece is already selected"""
        interactive_game = InteractiveDabloGame(vs_npc=False)
        interactive_game.ax = Mock()
        interactive_game.selected_piece = (1.0, 1.0)

        # Mock event
        event = Mock(inaxes=interactive_game.ax, xdata=2.0, ydata=-3.0)

        with patch.object(
            interactive_game, "_screen_to_game_pos", return_value=(3.0, 2.0)
        ):
            with patch.object(interactive_game, "_try_make_move") as mock_move:
                interactive_game._on_click(event)

        mock_move.assert_called_once_with((3.0, 2.0))


@pytest.mark.ui
@pytest.mark.integration
class TestInteractiveIntegration:
    """Integration tests for interactive game"""

    @patch("matplotlib.pyplot.show")
    @patch("builtins.print")
    def test_play_interactive_dablo_vs_npc(self, mock_print, mock_show):
        """Test main play function vs NPC"""
        with patch.object(InteractiveDabloGame, "start_game") as mock_start:
            play_interactive_dablo(vs_npc=True, difficulty="easy")

        mock_start.assert_called_once()

    @patch("matplotlib.pyplot.show")
    @patch("builtins.print")
    def test_play_interactive_dablo_vs_human(self, mock_print, mock_show):
        """Test main play function vs human"""
        with patch.object(InteractiveDabloGame, "start_game") as mock_start:
            play_interactive_dablo(vs_npc=False)

        mock_start.assert_called_once()

    @patch("builtins.print")
    def test_handle_game_over_draw(self, mock_print):
        """Test handling game over with draw"""
        interactive_game = InteractiveDabloGame(vs_npc=False)
        interactive_game.game.winner = None
        interactive_game.game.move_count = 50

        interactive_game._handle_game_over()

        mock_print.assert_any_call("Game Over! It's a draw!")

    def test_update_display_with_selection(self):
        """Test display update with piece selection"""
        interactive_game = InteractiveDabloGame(vs_npc=False)

        # Mock components
        interactive_game.ax = Mock()
        interactive_game.fig = Mock()
        interactive_game.selected_piece = (2.0, 2.0)
        interactive_game.valid_moves_for_piece = [
            Move(from_pos=(2.0, 2.0), to_pos=(3.0, 3.0))
        ]

        with patch.object(interactive_game.visualizer, "_draw_board_grid"):
            with patch.object(interactive_game.visualizer, "_draw_pieces"):
                with patch.object(interactive_game.visualizer, "_style_plot"):
                    with patch.object(interactive_game, "_add_game_status"):
                        interactive_game._update_display()

        # Should have drawn selection highlight
        interactive_game.ax.add_patch.assert_called()

    def test_add_game_status_various_states(self):
        """Test adding game status in various states"""
        interactive_game = InteractiveDabloGame(vs_npc=False)
        interactive_game.ax = Mock()

        # Test normal state
        interactive_game.game.move_count = 10
        interactive_game.game.current_player = Player.P1
        interactive_game._add_game_status()
        assert interactive_game.ax.text.call_count == 2

        # Reset mock
        interactive_game.ax.reset_mock()

        # Test with chain capture
        interactive_game.game.capture_sequence = True
        interactive_game._add_game_status()
        assert interactive_game.ax.text.call_count == 2

        # Reset mock
        interactive_game.ax.reset_mock()

        # Test with selected piece
        interactive_game.selected_piece = (2.0, 2.0)
        interactive_game.game.board_state[(2.0, 2.0)] = PieceType.P1_KING
        interactive_game._add_game_status()
        assert interactive_game.ax.text.call_count == 2

        # Reset mock
        interactive_game.ax.reset_mock()

        # Test game over
        interactive_game.game.game_over = True
        interactive_game.game.winner = Player.P2
        interactive_game._add_game_status()
        assert interactive_game.ax.text.call_count == 2
