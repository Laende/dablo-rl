from multiprocessing import Pool, cpu_count
from typing import Any
import warnings

from ..core.config import DabloConfig
from ..core.game import DabloGame
from ..core.player import Player
from .config import npc_config
from .players import create_npc_player


# Suppress annoying multiprocessing warnings
warnings.filterwarnings(
    "ignore", message=".*found in sys.modules.*", category=RuntimeWarning
)


def run_single_game(args):
    """Run a single game between two NPCs - designed for multiprocessing"""
    game_num, npc1_type, npc2_type, npc1_diff, npc2_diff, move_limit = args

    game = DabloGame(config=DabloConfig(move_limit=move_limit))
    npc1 = create_npc_player(player=Player.P1, npc_type=npc1_type, difficulty=npc1_diff)
    npc2 = create_npc_player(player=Player.P2, npc_type=npc2_type, difficulty=npc2_diff)
    npcs = {Player.P1: npc1, Player.P2: npc2}

    # Game loop
    while not game.game_over:
        current_npc = npcs[game.current_player]
        move = current_npc.get_move(game)

        if move:
            success, _ = game.make_move(move)
            if not success:
                game.end_game(winner=-game.current_player, reason="invalid_move_bug")
        else:
            break

    # Collect cache stats if available
    threat_hits = threat_misses = 0
    for npc in npcs.values():
        if hasattr(npc, "_threat_cache_hits"):
            threat_hits += npc._threat_cache_hits
            threat_misses += npc._threat_cache_misses

    return {
        "game_num": game_num,
        "winner": game.winner,
        "moves": game.move_count,
        "reason": game.win_reason,
        "threat_hits": threat_hits,
        "threat_misses": threat_misses,
    }


def evaluate_npc_performance(
    npc1_type: str,
    npc2_type: str,
    npc1_diff: str = "medium",
    npc2_diff: str = "medium",
    n_games: int = 50,
    parallel: bool = True,
) -> dict[str, Any]:
    """Evaluates NPC performance by simulating games between two AI opponents."""

    results = {"npc1_wins": 0, "npc2_wins": 0, "draws": 0}
    move_limit = npc_config.performance_test.default_move_limit

    if parallel and n_games > 1:
        # Parallel execution
        game_args = [
            (game_num, npc1_type, npc2_type, npc1_diff, npc2_diff, move_limit)
            for game_num in range(1, n_games + 1)
        ]

        # Use all available CPU cores, but cap at reasonable number
        num_processes = min(cpu_count(), n_games, 8)
        print(f"    Running {n_games} games in parallel on {num_processes} cores...")

        with Pool(processes=num_processes) as pool:
            game_details = pool.map(run_single_game, game_args)
    else:
        # Sequential execution (for debugging or small runs)
        game_details = []
        for game_num in range(1, n_games + 1):
            game_args = (
                game_num,
                npc1_type,
                npc2_type,
                npc1_diff,
                npc2_diff,
                move_limit,
            )
            game_details.append(run_single_game(game_args))

    # Process results from parallel execution
    total_threat_hits = sum(game.get("threat_hits", 0) for game in game_details)
    total_threat_misses = sum(game.get("threat_misses", 0) for game in game_details)

    # Count wins and draws
    for game in game_details:
        if game["winner"] == Player.P1:
            results["npc1_wins"] += 1
        elif game["winner"] == Player.P2:
            results["npc2_wins"] += 1
        else:
            results["draws"] += 1

    # Calculate and return final statistics
    total = len(game_details)
    if total > 0:
        moves_list = [g["moves"] for g in game_details]
        npc1_wins_moves = [g["moves"] for g in game_details if g["winner"] == Player.P1]
        npc2_wins_moves = [g["moves"] for g in game_details if g["winner"] == Player.P2]

        # Win reason analysis
        win_reasons = {}
        for game in game_details:
            reason = game.get("reason", "unknown")
            win_reasons[reason] = win_reasons.get(reason, 0) + 1

        results.update(
            {
                "npc1_win_rate": results["npc1_wins"] / total,
                "npc2_win_rate": results["npc2_wins"] / total,
                "draw_rate": results["draws"] / total,
                "avg_moves": sum(moves_list) / total,
                "min_moves": min(moves_list),
                "max_moves": max(moves_list),
                "moves_std": (
                    sum(
                        (x - sum(moves_list) / len(moves_list)) ** 2 for x in moves_list
                    )
                    / len(moves_list)
                )
                ** 0.5,
                "npc1_avg_win_moves": sum(npc1_wins_moves) / len(npc1_wins_moves)
                if npc1_wins_moves
                else 0,
                "npc2_avg_win_moves": sum(npc2_wins_moves) / len(npc2_wins_moves)
                if npc2_wins_moves
                else 0,
                "win_reasons": win_reasons,
                "games_over_100_moves": sum(1 for m in moves_list if m > 100),
                "games_under_50_moves": sum(1 for m in moves_list if m < 50),
                # Cache statistics
                "cache_threat_hits": total_threat_hits,
                "cache_threat_misses": total_threat_misses,
            }
        )
    results["games"] = game_details
    return results


if __name__ == "__main__":
    """Main entry point for running NPC performance tests."""
    print("🤖 Testing NPC AI Performance")
    print("=" * 40)

    n_games = 100  # Reduced for testing lookahead performance
    for matchup in npc_config.performance_test.test_matchups:
        npc1, npc2, diff1, diff2 = matchup
        print(
            f"\n{npc1.title()} ({diff1}) vs {npc2.title()} ({diff2}) ({n_games} games)"
        )

        results = evaluate_npc_performance(
            npc1, npc2, diff1, diff2, n_games=n_games, parallel=True
        )

        print(
            f"  - {npc1.title()} Wins: {results.get('npc1_win_rate', 0):.1%} (avg {results.get('npc1_avg_win_moves', 0):.1f} moves)"
        )
        print(
            f"  - {npc2.title()} Wins: {results.get('npc2_win_rate', 0):.1%} (avg {results.get('npc2_avg_win_moves', 0):.1f} moves)"
        )
        print(f"  - Draws: {results.get('draw_rate', 0):.1%}")
        print(
            f"  - Move Stats: {results.get('avg_moves', 0):.1f}±{results.get('moves_std', 0):.1f} (range: {results.get('min_moves', 0)}-{results.get('max_moves', 0)})"
        )
        print(
            f"  - Game Length: {results.get('games_under_50_moves', 0)} short (<50), {results.get('games_over_100_moves', 0)} long (>100)"
        )
        if results.get("win_reasons"):
            reasons_str = ", ".join(
                [
                    f"{reason}: {count}"
                    for reason, count in results.get("win_reasons", {}).items()
                ]
            )
            print(f"  - Win Reasons: {reasons_str}")

        # Cache statistics
        threat_total = results.get("cache_threat_hits", 0) + results.get(
            "cache_threat_misses", 0
        )
        if threat_total > 0:
            threat_rate = results.get("cache_threat_hits", 0) / threat_total * 100
            print(
                f"  - Cache Hit Rate: {threat_rate:.1f}% ({results.get('cache_threat_hits', 0)}/{threat_total})"
            )
