"""
Simple matplotlib-based game visualizer for Dablo

Provides a clean, easy-to-use visualization of the game board
that can be used for debugging, analysis, or demonstrations.
"""

from matplotlib.patches import Circle
import matplotlib.pyplot as plt

from ..core.game import DabloGame
from ..core.pieces import PieceType


class DabloVisualizer:
    """Visual game board using matplotlib"""

    def __init__(self, figsize=(10, 8)):
        self.figsize = figsize
        self.colors = {
            # Player 1 (Aalmogh) - Blue tones
            PieceType.P1_WARRIOR: "#4A90E2",
            PieceType.P1_PRINCE: "#2E5BBA",
            PieceType.P1_KING: "#1A4480",
            # Player 2 (Daatjh) - Red tones
            PieceType.P2_WARRIOR: "#E24A4A",
            PieceType.P2_PRINCE: "#BA2E2E",
            PieceType.P2_KING: "#801A1A",
            # Board
            PieceType.EMPTY: "#F0F0F0",
            "primary_node": "#FFFFFF",
            "secondary_node": "#E8E8E8",
            "grid": "#CCCCCC",
        }

        self.piece_symbols = {
            PieceType.P1_WARRIOR: "♟",
            PieceType.P1_PRINCE: "♝",
            PieceType.P1_KING: "♔",
            PieceType.P2_WARRIOR: "♙",
            PieceType.P2_PRINCE: "♗",
            PieceType.P2_KING: "♚",
        }

        self.piece_sizes = {
            PieceType.P1_WARRIOR: 100,
            PieceType.P1_PRINCE: 150,
            PieceType.P1_KING: 200,
            PieceType.P2_WARRIOR: 100,
            PieceType.P2_PRINCE: 150,
            PieceType.P2_KING: 200,
        }

    def visualize_game(
        self,
        game: DabloGame,
        title: str | None = None,
        highlight_moves: bool = False,
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Create a visual representation of the game board

        Args:
            game: DabloGame instance to visualize
            title: Optional title for the plot
            highlight_moves: Whether to highlight valid moves
            save_path: Optional path to save the image

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Draw board grid
        self._draw_board_grid(ax, game)

        # Draw pieces
        self._draw_pieces(ax, game)

        # Highlight valid moves if requested
        if highlight_moves:
            self._highlight_valid_moves(ax, game)

        # Add game info
        self._add_game_info(ax, game, title)

        # Style the plot
        self._style_plot(ax)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def _draw_board_grid(self, ax, game: DabloGame):
        """Draw the board grid and nodes"""
        # Draw connections between nodes
        for pos, neighbors in game.nodes.items():
            x1, y1 = pos[1], -pos[0]  # Flip Y for display
            for neighbor in neighbors:
                x2, y2 = neighbor[1], -neighbor[0]
                ax.plot([x1, x2], [y1, y2], "gray", alpha=0.3, linewidth=1)

        # Draw nodes
        for pos in game.positions:
            x, y = pos[1], -pos[0]  # Flip Y for display
            if pos[0] == int(pos[0]) and pos[1] == int(pos[1]):
                # Primary node
                circle = Circle(
                    (x, y),
                    0.15,
                    color=self.colors["primary_node"],
                    ec=self.colors["grid"],
                    linewidth=2,
                    zorder=1,
                )
            else:
                # Secondary node
                circle = Circle(
                    (x, y),
                    0.1,
                    color=self.colors["secondary_node"],
                    ec=self.colors["grid"],
                    linewidth=1,
                    zorder=1,
                )
            ax.add_patch(circle)

    def _draw_pieces(self, ax, game: DabloGame):
        """Draw all pieces on the board"""
        for pos, piece in game.board_state.items():
            if piece != PieceType.EMPTY:
                x, y = pos[1], -pos[0]  # Flip Y for display

                # Draw piece as colored circle
                circle = Circle(
                    (x, y),
                    0.12,
                    color=self.colors[piece],
                    ec="black",
                    linewidth=2,
                    zorder=3,
                )
                ax.add_patch(circle)

                # Add piece symbol
                symbol = self.piece_symbols.get(piece, "?")
                ax.text(
                    x,
                    y,
                    symbol,
                    ha="center",
                    va="center",
                    fontsize=16,
                    fontweight="bold",
                    color="white",
                    zorder=4,
                )

    def _highlight_valid_moves(self, ax, game: DabloGame):
        """Highlight valid moves for current player"""
        valid_moves = game.get_all_valid_moves()

        # Highlight pieces that can move
        movable_positions = {move.from_pos for move in valid_moves}
        for pos in movable_positions:
            x, y = pos[1], -pos[0]
            circle = Circle((x, y), 0.18, fill=False, ec="lime", linewidth=3, zorder=2)
            ax.add_patch(circle)

        # Show possible destinations
        for move in valid_moves:
            x, y = move.to_pos[1], -move.to_pos[0]
            if move.is_capture:
                # Capture moves in red
                circle = Circle((x, y), 0.08, color="red", alpha=0.7, zorder=2)
            else:
                # Regular moves in green
                circle = Circle((x, y), 0.06, color="lime", alpha=0.7, zorder=2)
            ax.add_patch(circle)

    def _add_game_info(self, ax, game: DabloGame, title: str | None):
        """Add game information text"""
        if not title:
            title = f"Dablo Game - Move {game.move_count}"

        # Current player info
        current_player_name = game.current_player.name

        info_text = f"{title}\nCurrent Player: {current_player_name}"

        if game.capture_sequence:
            info_text += "\nChain capture in progress!"

        if game.game_over:
            winner_text = game.winner.name if game.winner else "Draw"
            info_text += f"\nGame Over - Winner: {winner_text}"

        ax.text(
            0.02,
            0.98,
            info_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    def _style_plot(self, ax):
        """Apply styling to the plot"""
        ax.set_aspect("equal")
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(-5.5, 0.5)
        ax.set_title("Dablo Game Board", fontsize=16, fontweight="bold")

        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])

        # Add subtle background
        ax.set_facecolor("#FAFAFA")

    def animate_move(self, game: DabloGame, move, duration: float = 1.0):
        """
        Simple animation of a move (requires matplotlib animation)
        This is a placeholder for future animation features
        """
        # For now, just show before and after
        self.visualize_game(game, title="Before Move")
        plt.show()

        # Make the move
        game.make_move(move)

        self.visualize_game(game, title="After Move")
        plt.show()


def quick_visualize(game: DabloGame, show: bool = True, save_path: str | None = None):
    """
    Quick visualization function for easy use

    Args:
        game: DabloGame to visualize
        show: Whether to display the plot
        save_path: Optional path to save image
    """
    visualizer = DabloVisualizer()
    fig = visualizer.visualize_game(game, save_path=save_path)

    if show:
        plt.show()

    return fig


# Example usage functions
def demo_visualization():
    """Demo the visualization with a fresh game"""
    game = DabloGame()

    # Show initial position
    quick_visualize(game, title="Initial Position")

    # Make a few moves and show
    moves = game.get_all_valid_moves()
    if moves:
        game.make_move(moves[0])
        quick_visualize(game, title="After First Move")


if __name__ == "__main__":
    demo_visualization()
