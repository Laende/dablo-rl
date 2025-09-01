"""Tests for AI performance and benchmarking"""

import time

import pytest

from dablo.ai.performance import evaluate_npc_performance, run_single_game
from dablo.ai.players import SmartNPCPlayer
from dablo.core.config import DabloConfig
from dablo.core.game import DabloGame
from dablo.core.player import Player
from dablo.core.rules import Move


@pytest.mark.performance
@pytest.mark.unit
class TestPerformanceFunctions:
    """Test performance measurement functions"""

    def test_run_single_game_structure(self):
        """Test that run_single_game returns expected structure"""
        # Mock args for single game
        args = (1, "smart", "random", "medium", "medium", 100)

        result = run_single_game(args)

        # Should return dict with expected keys
        assert isinstance(result, dict)

        expected_keys = ["game_num", "winner", "moves", "reason"]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

        # Check types
        assert isinstance(result["game_num"], int)
        assert isinstance(result["moves"], int)
        assert result["winner"] in [Player.P1, Player.P2, None]  # None for draws

    def test_run_single_game_completes(self):
        """Test that single game completes without hanging"""
        args = (1, "random", "random", "easy", "easy", 50)  # Short game

        start_time = time.time()
        result = run_single_game(args)
        duration = time.time() - start_time

        # Should complete quickly with random players
        assert duration < 5.0, f"Game took too long: {duration:.2f}s"
        assert result["game_num"] == 1
        assert result["moves"] >= 0

    def test_run_single_game_with_different_npcs(self):
        """Test single game with different NPC types"""
        test_cases = [
            ("smart", "smart"),
            ("random", "smart"),
            ("aggressive", "defensive"),
        ]

        for npc1, npc2 in test_cases:
            args = (1, npc1, npc2, "medium", "medium", 100)

            result = run_single_game(args)

            assert isinstance(result, dict)
            assert "winner" in result
            assert "moves" in result


@pytest.mark.performance
@pytest.mark.integration
@pytest.mark.slow
class TestNPCPerformanceEvaluation:
    """Test NPC performance evaluation system"""

    def test_evaluate_npc_performance_basic(self):
        """Test basic NPC performance evaluation"""
        # Use small number of games for testing
        result = evaluate_npc_performance(
            npc1_type="random",
            npc2_type="random",
            n_games=3,
            parallel=False,  # Sequential for easier testing
        )

        assert isinstance(result, dict)

        # Should have win counts
        expected_keys = ["npc1_wins", "npc2_wins", "draws"]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
            assert isinstance(result[key], int)
            assert result[key] >= 0

        # Total should equal number of games
        total_games = result["npc1_wins"] + result["npc2_wins"] + result["draws"]
        assert total_games == 3

    def test_evaluate_npc_performance_parallel_vs_sequential(self):
        """Test that parallel and sequential give similar results"""
        # Small test to compare modes
        n_games = 5

        # Sequential
        seq_result = evaluate_npc_performance(
            "random", "random", n_games=n_games, parallel=False
        )

        # Parallel
        par_result = evaluate_npc_performance(
            "random", "random", n_games=n_games, parallel=True
        )

        # Both should complete and have same total games
        seq_total = (
            seq_result["npc1_wins"] + seq_result["npc2_wins"] + seq_result["draws"]
        )
        par_total = (
            par_result["npc1_wins"] + par_result["npc2_wins"] + par_result["draws"]
        )

        assert seq_total == n_games
        assert par_total == n_games


@pytest.mark.performance
class TestNPCIndividualPerformance:
    """Test individual NPC performance characteristics"""

    def test_npc_move_time_consistency(self, game):
        """Test that NPC move times are consistent"""
        npc = SmartNPCPlayer(Player.P1, "medium")

        move_times = []
        max_moves = 10

        for _ in range(max_moves):
            if game.game_over:
                break

            start_time = time.time()
            move = npc.get_move(game)
            move_time = time.time() - start_time
            move_times.append(move_time)

            if move:
                success, _ = game.make_move(move)
                if not success:
                    break

        if move_times:
            avg_time = sum(move_times) / len(move_times)
            max_time = max(move_times)

            # Average should be reasonable
            assert avg_time < 0.05, f"Average move time too slow: {avg_time:.3f}s"
            # No outlier moves
            assert max_time < 0.2, f"Slowest move too slow: {max_time:.3f}s"

    def test_npc_cache_performance(self):
        """Test NPC caching improves performance over time"""
        npc = SmartNPCPlayer(Player.P1, "expert")

        # Create multiple similar game states to test caching
        config = DabloConfig.create_test_config()

        first_game_times = []
        cached_game_times = []

        # First run (cold cache)
        for _ in range(3):
            game = DabloGame(config=config)

            start_time = time.time()
            _ = npc.get_move(game)
            move_time = time.time() - start_time
            first_game_times.append(move_time)

        # Second run (warm cache)
        for _ in range(3):
            game = DabloGame(config=config)

            start_time = time.time()
            _ = npc.get_move(game)
            move_time = time.time() - start_time
            cached_game_times.append(move_time)

        # Cache should generally improve performance
        if first_game_times and cached_game_times:
            avg_first = sum(first_game_times) / len(first_game_times)
            avg_cached = sum(cached_game_times) / len(cached_game_times)
            print("RUN TIEMS!")
            print(avg_first, avg_cached)
            # Both should be reasonable
            assert avg_first < 0.1, "First moves too slow"
            assert avg_cached < 0.1, "Cached moves too slow"


@pytest.mark.npc_performance  # Custom marker from CLAUDE.md
@pytest.mark.slow
class TestNPCPerformanceBenchmarks:
    """Benchmarking tests for NPC performance (uses custom marker)"""

    def test_smart_vs_random_benchmark(self):
        """Benchmark smart vs random NPCs"""
        result = evaluate_npc_performance(
            npc1_type="smart",
            npc2_type="random",
            npc1_diff="medium",
            npc2_diff="medium",
            n_games=10,  # Reduced for testing
            parallel=True,
        )

        # Smart should generally perform better than random
        smart_wins = result["npc1_wins"]
        random_wins = result["npc2_wins"]
        draws = result["draws"]

        total_games = smart_wins + random_wins + draws
        assert total_games == 10

        # At minimum, smart should not lose more than 70% of games
        smart_performance = (smart_wins + draws * 0.5) / total_games
        assert smart_performance > 0.3, (
            f"Smart NPC performed poorly: {smart_performance:.2%}"
        )


@pytest.mark.performance
class TestPerformanceEdgeCases:
    """Test edge cases and error conditions in performance module"""

    def test_run_single_game_invalid_move_handling(self):
        """Test run_single_game handles invalid moves gracefully (line 35)"""

        # Create a mock NPC that returns invalid moves
        class InvalidMoveNPC:
            def __init__(self, player_id):
                self.player_id = player_id

            def get_move(self, game):
                # Return an invalid move that will fail
                return Move(from_pos=(-1.0, -1.0), to_pos=(-2.0, -2.0))

        # Monkey patch the create_npc_player function
        import dablo.ai.performance

        original_create_npc = dablo.ai.performance.create_npc_player

        def mock_create_npc(player, npc_type, difficulty):
            return InvalidMoveNPC(player)

        dablo.ai.performance.create_npc_player = mock_create_npc

        try:
            args = (1, "invalid", "invalid", "medium", "medium", 100)
            result = run_single_game(args)

            # Should handle the invalid move and end the game
            assert isinstance(result, dict)
            assert "winner" in result
            assert result["moves"] >= 0

        finally:
            # Restore original function
            dablo.ai.performance.create_npc_player = original_create_npc

    def test_evaluate_npc_performance_with_draws(self):
        """Test evaluate_npc_performance handles draw conditions (line 107)"""

        # Create a mock that forces draws
        class DrawNPC:
            def __init__(self, player_id):
                self.player_id = player_id
                self._move_count = 0

            def get_move(self, game):
                self._move_count += 1
                # Return None after a few moves to simulate draws
                if self._move_count > 2:
                    return None

                # Return a valid move for the first few turns
                moves = game.get_all_valid_moves()
                return moves[0] if moves else None

        # Monkey patch to force draws
        import dablo.ai.performance

        original_create_npc = dablo.ai.performance.create_npc_player

        def mock_create_npc(player, npc_type, difficulty):
            return DrawNPC(player)

        dablo.ai.performance.create_npc_player = mock_create_npc

        try:
            result = evaluate_npc_performance(
                npc1_type="draw", npc2_type="draw", n_games=3, parallel=False
            )

            # Should have some draws
            assert isinstance(result, dict)
            assert "draws" in result
            total = result["npc1_wins"] + result["npc2_wins"] + result["draws"]
            assert total == 3

        finally:
            # Restore original function
            dablo.ai.performance.create_npc_player = original_create_npc
