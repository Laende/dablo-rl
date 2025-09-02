#!/usr/bin/env python3
"""
Performance test for DabloGame optimizations.
Tests the performance improvements of the PieceTracker implementation.
"""

from collections.abc import Callable
import statistics
import time
from typing import Any

from dablo.core.config import DabloConfig
from dablo.core.game import DabloGame
from dablo.core.pieces import PieceType
from dablo.core.player import Player


class PerformanceTester:
    """Test performance of game operations."""

    def __init__(self):
        self.config = DabloConfig.create_default()
        self.results = {}

    def time_function(self, func: Callable, *args, **kwargs) -> tuple[Any, float]:
        """Time a function call and return result and execution time."""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return result, end - start

    def run_multiple_times(
        self, func: Callable, iterations: int = 1000, *args, **kwargs
    ) -> dict:
        """Run a function multiple times and collect timing statistics."""
        times = []

        for _ in range(iterations):
            _, execution_time = self.time_function(func, *args, **kwargs)
            times.append(execution_time)

        return {
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "min": min(times),
            "max": max(times),
            "stddev": statistics.stdev(times) if len(times) > 1 else 0,
            "iterations": iterations,
        }

    def test_game_initialization(self, iterations: int = 100):
        """Test game initialization performance."""
        print("Testing game initialization...")

        def create_game():
            return DabloGame(self.config)

        stats = self.run_multiple_times(create_game, iterations)
        self.results["initialization"] = stats

        print(f"Game initialization (n={iterations}):")
        print(f"  Mean: {stats['mean']:.6f}s")
        print(f"  Median: {stats['median']:.6f}s")
        print(f"  Min: {stats['min']:.6f}s, Max: {stats['max']:.6f}s")

    def test_king_position_lookup(self, iterations: int = 10000):
        """Test king position lookup performance."""
        print("\nTesting king position lookup...")

        game = DabloGame(self.config)

        def lookup_kings():
            p1_king = game.get_king_position(Player.P1)
            p2_king = game.get_king_position(Player.P2)
            return p1_king, p2_king

        stats = self.run_multiple_times(lookup_kings, iterations)
        self.results["king_lookup"] = stats

        print(f"King position lookup (n={iterations}):")
        print(f"  Mean: {stats['mean']:.8f}s")
        print(f"  Median: {stats['median']:.8f}s")
        print(f"  Operations per second: {1 / stats['mean']:.0f}")

    def test_piece_access(self, iterations: int = 10000):
        """Test piece collection access performance."""
        print("\nTesting piece collection access...")

        game = DabloGame(self.config)

        def access_pieces():
            return len(game.p1_pieces), len(game.p2_pieces)

        stats = self.run_multiple_times(access_pieces, iterations)
        self.results["piece_access"] = stats

        print(f"Piece collection access (n={iterations}):")
        print(f"  Mean: {stats['mean']:.8f}s")
        print(f"  Median: {stats['median']:.8f}s")
        print(f"  Operations per second: {1 / stats['mean']:.0f}")

    def test_move_generation(self, iterations: int = 1000):
        """Test move generation performance."""
        print("\nTesting move generation...")

        game = DabloGame(self.config)

        def generate_moves():
            return game.get_all_valid_moves()

        stats = self.run_multiple_times(generate_moves, iterations)
        self.results["move_generation"] = stats

        print(f"Move generation (n={iterations}):")
        print(f"  Mean: {stats['mean']:.6f}s")
        print(f"  Median: {stats['median']:.6f}s")
        print(f"  Operations per second: {1 / stats['mean']:.0f}")

    def test_piece_operations(self, iterations: int = 1000):
        """Test add/remove piece operations."""
        print("\nTesting piece add/remove operations...")

        def piece_operations():
            game = DabloGame(self.config, initial_state="empty")

            # Add pieces

            test_positions = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]

            for pos in test_positions:
                game.add_piece(pos, PieceType.P1_WARRIOR)

            # Check piece counts
            p1_count = len(game.p1_pieces)

            # Remove pieces
            for pos in test_positions:
                game.remove_piece(pos)

            return p1_count

        stats = self.run_multiple_times(piece_operations, iterations)
        self.results["piece_operations"] = stats

        print(f"Piece add/remove operations (n={iterations}):")
        print(f"  Mean: {stats['mean']:.6f}s")
        print(f"  Median: {stats['median']:.6f}s")
        print(f"  Operations per second: {1 / stats['mean']:.0f}")

    def test_game_simulation(self, iterations: int = 50):
        """Test full game simulation performance."""
        print("\nTesting full game simulation...")

        def simulate_game():
            game = DabloGame(self.config)
            moves_made = 0

            while (
                not game.game_over and moves_made < 100
            ):  # Limit to prevent infinite games
                valid_moves = game.get_all_valid_moves()
                if not valid_moves:
                    break

                # Make a random move
                success, _ = game.make_random_move()
                if success:
                    moves_made += 1
                else:
                    break

            return moves_made

        stats = self.run_multiple_times(simulate_game, iterations)
        self.results["game_simulation"] = stats

        print(f"Game simulation (n={iterations}):")
        print(f"  Mean: {stats['mean']:.4f}s")
        print(f"  Median: {stats['median']:.4f}s")

    def run_all_tests(self):
        """Run all performance tests."""
        print("=" * 60)
        print("DABLO GAME PERFORMANCE TESTS")
        print("=" * 60)

        iterations = 50_000
        self.test_game_initialization(iterations)
        self.test_king_position_lookup(iterations)
        self.test_piece_access(iterations)
        self.test_move_generation(iterations)
        self.test_piece_operations(iterations)
        self.test_game_simulation(2000)

        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)

        for test_name, stats in self.results.items():
            print(f"{test_name.replace('_', ' ').title()}:")
            print(f"  Average: {stats['mean']:.6f}s")
            if stats["mean"] > 0:
                print(f"  Ops/sec: {1 / stats['mean']:.0f}")

        # Calculate overall performance metrics
        print(
            f"\nTotal tests run: {sum(stats['iterations'] for stats in self.results.values())}"
        )
        print("Tests completed successfully!")


if __name__ == "__main__":
    tester = PerformanceTester()
    tester.run_all_tests()
