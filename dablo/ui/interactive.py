"""
Interactive Dablo game using matplotlib

Click-to-play interface for the Dablo game with mouse interaction.
"""

from matplotlib.patches import Circle
import matplotlib.pyplot as plt

from ..core.game import DabloGame
from ..core.pieces import PieceType
from ..core.player import Player
from ..core.rules import Move
from .visualizer import DabloVisualizer


class InteractiveDabloGame:
    """Interactive Dablo game with click-to-play interface"""

    def __init__(self, vs_npc: bool = True, npc_difficulty: str = "medium"):
        self.game = DabloGame()
        self.visualizer = DabloVisualizer(figsize=(12, 10))
        self.vs_npc = vs_npc
        self.npc_difficulty = npc_difficulty

        # Game state
        self.selected_piece = None
        self.valid_moves_for_piece = []
        self.game_history = []

        # Create NPC if needed
        self.npc_player = None
        if vs_npc:
            from ..ai.players import create_npc_player

            self.npc_player = create_npc_player(Player.P2, "smart", npc_difficulty)

        # UI state
        self.fig = None
        self.ax = None

    def start_game(self):
        """Start the interactive game"""
        print("🎮 Starting Interactive Dablo Game!")
        print("Click on your pieces to select them, then click destination to move.")
        print("Blue pieces are yours (P1), Red pieces are opponent (P2)")

        self.fig, self.ax = plt.subplots(figsize=self.visualizer.figsize)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._update_display()
        plt.show()

    def _on_click(self, event):
        """Handle mouse click events"""
        if event.inaxes != self.ax:
            return

        # Convert screen coordinates to game coordinates
        clicked_pos = self._screen_to_game_pos(event.xdata, event.ydata)

        if clicked_pos is None:
            return

        # Handle the click based on current state
        if self.selected_piece is None:
            self._try_select_piece(clicked_pos)
        else:
            self._try_make_move(clicked_pos)

    def _on_key(self, event):
        """Handle keyboard events"""
        if event.key == "escape":
            self._deselect_piece()
        elif event.key == "r":
            self._restart_game()
        elif event.key == "u":
            self._undo_move()
        elif event.key == "h":
            self._show_help()

    def _screen_to_game_pos(self, x, y) -> tuple[float, float] | None:
        """Convert screen coordinates to game position"""
        if x is None or y is None:
            return None

        # Convert back from display coordinates (flipped Y)
        game_row = -y
        game_col = x

        # Find closest valid position
        min_dist = float("inf")
        closest_pos = None

        for pos in self.game.positions:
            dist = ((pos[0] - game_row) ** 2 + (pos[1] - game_col) ** 2) ** 0.5
            if dist < min_dist and dist < 0.2:  # Within click tolerance
                min_dist = dist
                closest_pos = pos

        return closest_pos

    def _try_select_piece(self, pos: tuple[float, float]):
        """Try to select a piece at the given position"""
        piece = self.game.board_state.get(pos, PieceType.EMPTY)

        # Check if it's the current player's piece
        if self.game.current_player == Player.P1 and piece.value > 0:
            self.selected_piece = pos
            self.valid_moves_for_piece = [
                move for move in self.game.get_all_valid_moves() if move.from_pos == pos
            ]
            print(f"Selected {piece.name} at {pos}")
            print(f"Available moves: {len(self.valid_moves_for_piece)}")
            self._update_display()

        elif self.game.current_player == Player.P2 and piece.value < 0:
            # Only allow if playing vs human
            if not self.vs_npc:
                self.selected_piece = pos
                self.valid_moves_for_piece = [
                    move
                    for move in self.game.get_all_valid_moves()
                    if move.from_pos == pos
                ]
                print(f"Selected {piece.name} at {pos}")
                self._update_display()
            else:
                print("It's the NPC's turn, please wait...")

        else:
            print("Cannot select that piece!")

    def _try_make_move(self, pos: tuple[float, float]):
        """Try to make a move to the given position"""
        if not self.valid_moves_for_piece:
            self._deselect_piece()
            return

        # Find move that goes to this position
        target_move = None
        for move in self.valid_moves_for_piece:
            if move.to_pos == pos:
                target_move = move
                break

        if target_move:
            self._execute_move(target_move)
        else:
            print(f"Invalid move to {pos}")
            self._deselect_piece()

    def _execute_move(self, move: Move):
        """Execute a move and update game state"""
        print(f"Moving from {move.from_pos} to {move.to_pos}")

        # Save current state for undo
        self.game_history.append(self._save_game_state())

        success, move_info = self.game.make_move(move)

        if success:
            if move_info.get("is_capture"):
                print(f"Captured {move_info['captured_piece'].name}!")

            if move_info.get("chain_capture_available"):
                print("Chain capture available! Select the piece again to continue.")
                # Keep the same piece selected for chain capture
                self.valid_moves_for_piece = [
                    m
                    for m in self.game.get_all_valid_moves()
                    if m.from_pos == move.to_pos
                ]
            else:
                self._deselect_piece()

            self._update_display()

            # Check for game over
            if self.game.game_over:
                self._handle_game_over()
            elif self.vs_npc and self.game.current_player == Player.P2:
                self._make_npc_move()
        else:
            print("Move failed!")
            self._deselect_piece()

    def _make_npc_move(self):
        """Make NPC move"""
        if not self.npc_player:
            return

        print("NPC is thinking...")
        self.fig.canvas.draw()
        plt.pause(0.5)  # Brief pause for realism

        npc_move = self.npc_player.get_move(self.game)

        if npc_move:
            print(f"NPC moves from {npc_move.from_pos} to {npc_move.to_pos}")
            success, move_info = self.game.make_move(npc_move)

            if success:
                if move_info.get("is_capture"):
                    print(f"NPC captured your {move_info['captured_piece'].name}!")

                # Handle NPC chain captures
                while (
                    move_info.get("chain_capture_available") and not self.game.game_over
                ):
                    self._update_display()
                    plt.pause(0.5)

                    npc_move = self.npc_player.get_move(self.game)
                    if npc_move:
                        print(
                            f"NPC chain capture: {npc_move.from_pos} to {npc_move.to_pos}"
                        )
                        success, move_info = self.game.make_move(npc_move)
                        if not success:
                            break
                    else:
                        break

                self._update_display()

                if self.game.game_over:
                    self._handle_game_over()

    def _deselect_piece(self):
        """Deselect current piece"""
        self.selected_piece = None
        self.valid_moves_for_piece = []
        self._update_display()

    def _update_display(self):
        """Update the visual display"""
        self.ax.clear()

        # Draw base game
        self.visualizer._draw_board_grid(self.ax, self.game)
        self.visualizer._draw_pieces(self.ax, self.game)

        # Highlight selected piece
        if self.selected_piece:
            x, y = self.selected_piece[1], -self.selected_piece[0]
            circle = Circle((x, y), 0.2, fill=False, ec="yellow", linewidth=4, zorder=5)
            self.ax.add_patch(circle)

            # Show valid moves for selected piece
            for move in self.valid_moves_for_piece:
                x, y = move.to_pos[1], -move.to_pos[0]
                if move.is_capture:
                    circle = Circle((x, y), 0.1, color="red", alpha=0.8, zorder=4)
                else:
                    circle = Circle((x, y), 0.08, color="lime", alpha=0.8, zorder=4)
                self.ax.add_patch(circle)

        # Add game info
        self._add_game_status()

        # Style
        self.visualizer._style_plot(self.ax)

        self.fig.canvas.draw()

    def _add_game_status(self):
        """Add game status information"""
        status_text = (
            f"Move {self.game.move_count} - {self.game.current_player.name}'s turn"
        )

        if self.game.capture_sequence:
            status_text += "\n🔗 Chain capture in progress!"

        if self.selected_piece:
            piece = self.game.board_state[self.selected_piece]
            status_text += f"\n📍 Selected: {piece.name}"

        if self.game.game_over:
            winner_text = self.game.winner.name if self.game.winner else "Draw"
            status_text += f"\n🏆 Game Over - {winner_text} wins!"

        # Position text in top-right corner to avoid blocking game
        self.ax.text(
            0.98,
            0.98,
            status_text,
            transform=self.ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        )

        # Add compact controls in bottom-right
        controls = "ESC:Deselect R:Restart U:Undo H:Help"
        self.ax.text(
            0.98,
            0.02,
            controls,
            transform=self.ax.transAxes,
            fontsize=8,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox={"boxstyle": "round", "facecolor": "lightgray", "alpha": 0.8},
        )

    def _handle_game_over(self):
        """Handle game over state"""
        print("\n" + "=" * 50)
        if self.game.winner:
            print(f"🏆 Game Over! {self.game.winner.name} wins!")
        else:
            print("🤝 Game Over! It's a draw!")
        print(f"Total moves: {self.game.move_count}")
        print("=" * 50)

    def _restart_game(self):
        """Restart the game"""
        self.game = DabloGame()
        self.selected_piece = None
        self.valid_moves_for_piece = []
        self.game_history = []
        print("🔄 Game restarted!")
        self._update_display()

    def _undo_move(self):
        """Undo last move"""
        if self.game_history:
            self._load_game_state(self.game_history.pop())
            self._deselect_piece()
            print("↶ Move undone!")
        else:
            print("Nothing to undo!")

    def _save_game_state(self) -> dict:
        """Save current game state"""
        return {
            "board_state": self.game.board_state.copy(),
            "current_player": self.game.current_player,
            "move_count": self.game.move_count,
            "capture_sequence": self.game.capture_sequence,
            "capturing_piece": self.game.capturing_piece,
            "game_over": self.game.game_over,
            "winner": self.game.winner,
        }

    def _load_game_state(self, state: dict):
        """Load saved game state"""
        self.game.board_state = state["board_state"]
        self.game.current_player = state["current_player"]
        self.game.move_count = state["move_count"]
        self.game.capture_sequence = state["capture_sequence"]
        self.game.capturing_piece = state["capturing_piece"]
        self.game.game_over = state["game_over"]
        self.game.winner = state["winner"]

    def _show_help(self):
        """Show help information"""
        help_text = """
🎮 DABLO INTERACTIVE GAME HELP

HOW TO PLAY:
• Click on your pieces (blue) to select them
• Click on highlighted destinations to move
• Yellow outline = selected piece
• Green dots = valid moves
• Red dots = capture moves

SPECIAL MOVES:
• Chain captures: After capturing, you must continue with the same piece
• The game will automatically highlight available chain captures

CONTROLS:
• ESC - Deselect current piece
• R - Restart game
• U - Undo last move
• H - Show this help

GOAL:
• Capture the opponent's king to win!
• Or capture all opponent pieces except the king

Good luck! 🍀
        """
        print(help_text)


def play_interactive_dablo(vs_npc: bool = True, difficulty: str = "medium"):
    """
    Start an interactive Dablo game

    Args:
        vs_npc: Whether to play against NPC (True) or human (False)
        difficulty: NPC difficulty ("easy", "medium", "hard")
    """
    game = InteractiveDabloGame(vs_npc=vs_npc, npc_difficulty=difficulty)
    game.start_game()


if __name__ == "__main__":
    print("🎲 Starting Interactive Dablo Game!")
    play_interactive_dablo(vs_npc=True, difficulty="medium")
