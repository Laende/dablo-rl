#!/usr/bin/env python3
"""
Model evaluation and comparison script for Dablo RL.

Provides comprehensive evaluation of trained models including head-to-head
comparison, performance analysis, and game outcome statistics.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dablo.core.config import DabloConfig
from dablo.rl.training import DabloSelfPlayManager


class DabloModelEvaluator:
    """Comprehensive model evaluation for Dablo RL agents."""

    def __init__(self, env_config: dict = None, reward_config: dict = None):
        """Initialize evaluator with environment configuration."""
        self.env_config = env_config or {}
        self.reward_config = reward_config or {}

        self.manager = DabloSelfPlayManager(
            env_config=self.env_config, reward_config=self.reward_config
        )

    def evaluate_single_model(
        self,
        model_path: str,
        n_episodes: int = 100,
        deterministic: bool = True,
        render: bool = False,
        save_games: bool = False,
    ) -> dict[str, Any]:
        """
        Evaluate a single model's performance.

        Args:
            model_path: Path to the model file
            n_episodes: Number of episodes to evaluate
            deterministic: Use deterministic policy
            render: Render games during evaluation
            save_games: Save detailed game information

        Returns:
            Dictionary containing evaluation results
        """
        print(f"Evaluating model: {model_path}")
        print(f"Episodes: {n_episodes}, Deterministic: {deterministic}")

        # Load model
        model = self.manager.load_model(model_path)

        # Create evaluation environment
        env = self.manager.create_environment(render_mode="human" if render else None)

        # Track detailed statistics
        episode_rewards = []
        episode_lengths = []
        game_outcomes = []
        win_reasons = []
        material_differences = []

        start_time = time.time()

        for episode in range(n_episodes):
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                # Get action mask for MaskablePPO
                action_masks = env.action_masks()

                # Predict action
                action, _ = model.predict(
                    obs, action_masks=action_masks, deterministic=deterministic
                )

                # Take step
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                episode_reward += reward
                episode_length += 1

            # Record episode statistics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # Record game-specific information
            if "winner" in info:
                game_outcomes.append(info["winner"])
            if "win_reason" in info:
                win_reasons.append(info["win_reason"])
            if "material_p1" in info and "material_p2" in info:
                material_differences.append(info["material_p1"] - info["material_p2"])

            # Progress update
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{n_episodes} completed")

        evaluation_time = time.time() - start_time
        env.close()

        # Calculate statistics
        results = {
            "model_path": model_path,
            "n_episodes": n_episodes,
            "evaluation_time": evaluation_time,
            "deterministic": deterministic,
            # Reward statistics
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards),
            # Episode length statistics
            "mean_length": np.mean(episode_lengths),
            "std_length": np.std(episode_lengths),
            "min_length": np.min(episode_lengths),
            "max_length": np.max(episode_lengths),
            # Game outcome statistics
            "total_games": len(game_outcomes),
            "p1_wins": sum(
                1 for outcome in game_outcomes if outcome and "P1" in outcome
            ),
            "p2_wins": sum(
                1 for outcome in game_outcomes if outcome and "P2" in outcome
            ),
            "draws": sum(1 for outcome in game_outcomes if outcome is None),
            # Win reason distribution
            "win_reasons": {
                reason: win_reasons.count(reason)
                for reason in set(win_reasons)
                if reason
            },
            # Material analysis
            "mean_material_diff": np.mean(material_differences)
            if material_differences
            else 0,
            "std_material_diff": np.std(material_differences)
            if material_differences
            else 0,
        }

        # Calculate win rates
        if results["total_games"] > 0:
            results["p1_win_rate"] = results["p1_wins"] / results["total_games"]
            results["p2_win_rate"] = results["p2_wins"] / results["total_games"]
            results["draw_rate"] = results["draws"] / results["total_games"]

        # Save detailed data if requested
        if save_games:
            results["episode_rewards"] = episode_rewards
            results["episode_lengths"] = episode_lengths
            results["game_outcomes"] = game_outcomes
            results["win_reasons"] = win_reasons
            results["material_differences"] = material_differences

        return results

    def compare_models(
        self,
        model_paths: list[str],
        n_episodes_per_model: int = 50,
        save_results: str = None,
    ) -> dict[str, Any]:
        """
        Compare multiple models head-to-head.

        Args:
            model_paths: List of paths to model files
            n_episodes_per_model: Episodes to evaluate per model
            save_results: Path to save comparison results

        Returns:
            Dictionary containing comparison results
        """
        print(f"Comparing {len(model_paths)} models")
        print(f"Episodes per model: {n_episodes_per_model}")

        comparison_results = {
            "models": {},
            "comparison_summary": {},
            "evaluation_params": {
                "n_episodes_per_model": n_episodes_per_model,
                "models_evaluated": len(model_paths),
            },
        }

        # Evaluate each model
        for i, model_path in enumerate(model_paths):
            print(f"\nEvaluating model {i + 1}/{len(model_paths)}: {model_path}")

            try:
                results = self.evaluate_single_model(
                    model_path=model_path,
                    n_episodes=n_episodes_per_model,
                    deterministic=True,
                    render=False,
                )

                model_name = Path(model_path).stem
                comparison_results["models"][model_name] = results

            except Exception as e:
                print(f"Error evaluating {model_path}: {e}")
                continue

        # Generate comparison summary
        if comparison_results["models"]:
            self._generate_comparison_summary(comparison_results)

        # Save results if requested
        if save_results:
            with open(save_results, "w") as f:
                json.dump(comparison_results, f, indent=2, default=str)
            print(f"Results saved to: {save_results}")

        return comparison_results

    def _generate_comparison_summary(self, comparison_results: dict):
        """Generate summary statistics for model comparison."""
        models_data = comparison_results["models"]

        if not models_data:
            return

        # Extract metrics for comparison
        metrics = ["mean_reward", "p1_win_rate", "mean_length"]
        summary = {}

        for metric in metrics:
            values = [data[metric] for data in models_data.values() if metric in data]
            if values:
                summary[f"best_{metric}"] = max(values)
                summary[f"worst_{metric}"] = min(values)
                summary[f"mean_{metric}"] = np.mean(values)
                summary[f"std_{metric}"] = np.std(values)

        # Find best performing model for each metric
        for metric in metrics:
            best_model = max(
                models_data.items(),
                key=lambda x: x[1].get(metric, -float("inf")),
                default=(None, {}),
            )
            if best_model[0]:
                summary[f"best_model_{metric}"] = best_model[0]
                summary[f"best_value_{metric}"] = best_model[1].get(metric, 0)

        comparison_results["comparison_summary"] = summary

    def print_evaluation_report(self, results: dict[str, Any]):
        """Print formatted evaluation report."""
        print("\n" + "=" * 60)
        print("DABLO RL MODEL EVALUATION REPORT")
        print("=" * 60)

        if "models" in results:
            # Multi-model comparison report
            self._print_comparison_report(results)
        else:
            # Single model report
            self._print_single_model_report(results)

    def _print_single_model_report(self, results: dict[str, Any]):
        """Print report for single model evaluation."""
        print(f"Model: {Path(results['model_path']).name}")
        print(f"Episodes: {results['n_episodes']}")
        print(f"Evaluation Time: {results['evaluation_time']:.2f}s")
        print()

        print("PERFORMANCE METRICS:")
        print(
            f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}"
        )
        print(
            f"  Reward Range: [{results['min_reward']:.2f}, {results['max_reward']:.2f}]"
        )
        print()

        print("GAME STATISTICS:")
        print(
            f"  Mean Game Length: {results['mean_length']:.1f} ± {results['std_length']:.1f}"
        )
        print(f"  P1 Win Rate: {results['p1_win_rate']:.1%}")
        print(f"  P2 Win Rate: {results['p2_win_rate']:.1%}")
        print(f"  Draw Rate: {results['draw_rate']:.1%}")
        print()

        if results["win_reasons"]:
            print("WIN REASONS:")
            for reason, count in results["win_reasons"].items():
                percentage = count / results["total_games"] * 100
                print(f"  {reason}: {count} ({percentage:.1f}%)")

        print()
        print(
            f"Material Advantage: {results['mean_material_diff']:.2f} ± {results['std_material_diff']:.2f}"
        )

    def _print_comparison_report(self, results: dict[str, Any]):
        """Print report for multi-model comparison."""
        models = results["models"]
        summary = results.get("comparison_summary", {})

        print(f"Models Compared: {len(models)}")
        print(
            f"Episodes per Model: {results['evaluation_params']['n_episodes_per_model']}"
        )
        print()

        print("MODEL PERFORMANCE COMPARISON:")
        print("-" * 80)
        print(
            f"{'Model Name':<20} {'Mean Reward':<12} {'P1 Win Rate':<12} {'Game Length':<12}"
        )
        print("-" * 80)

        for model_name, data in models.items():
            print(
                f"{model_name:<20} {data['mean_reward']:<12.2f} "
                f"{data['p1_win_rate']:<12.1%} {data['mean_length']:<12.1f}"
            )

        print()

        if summary:
            print("SUMMARY:")
            for metric in ["mean_reward", "p1_win_rate", "mean_length"]:
                best_model = summary.get(f"best_model_{metric}")
                best_value = summary.get(f"best_value_{metric}")
                if best_model and best_value is not None:
                    print(f"  Best {metric}: {best_model} ({best_value:.3f})")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate Dablo RL models")

    # Model selection
    parser.add_argument(
        "models", nargs="+", help="Path(s) to model files for evaluation"
    )

    # Evaluation parameters
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to evaluate per model",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy (default: deterministic)",
    )
    parser.add_argument(
        "--render", action="store_true", help="Render games during evaluation"
    )

    # Game configuration
    parser.add_argument(
        "--quick-game", action="store_true", help="Use quick game configuration"
    )
    parser.add_argument(
        "--test-config", action="store_true", help="Use test configuration"
    )

    # Output options
    parser.add_argument(
        "--save-results", type=str, help="Path to save detailed results (JSON format)"
    )
    parser.add_argument(
        "--save-games", action="store_true", help="Save detailed game information"
    )

    args = parser.parse_args()

    # Prepare configuration
    env_config = {}
    if args.quick_game:
        env_config["config"] = DabloConfig.create_quick_game()
    elif args.test_config:
        env_config["config"] = DabloConfig.create_test_config()

    try:
        # Initialize evaluator
        evaluator = DabloModelEvaluator(env_config=env_config)

        if len(args.models) == 1:
            # Single model evaluation
            results = evaluator.evaluate_single_model(
                model_path=args.models[0],
                n_episodes=args.episodes,
                deterministic=not args.stochastic,
                render=args.render,
                save_games=args.save_games,
            )

            # Save results
            if args.save_results:
                with open(args.save_results, "w") as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"Results saved to: {args.save_results}")

        else:
            # Multi-model comparison
            results = evaluator.compare_models(
                model_paths=args.models,
                n_episodes_per_model=args.episodes,
                save_results=args.save_results,
            )

        # Print report
        evaluator.print_evaluation_report(results)

        return 0

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
