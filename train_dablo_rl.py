"""
Dablo RL training script using configuration files.

This script uses a clean configuration-based approach instead of many command-line arguments.
"""

import argparse
from pathlib import Path
import sys


# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dablo.rl.config import DabloRLConfig
from dablo.rl.training import DabloSelfPlayManager, DabloTrainingMonitor


def train_model(config: DabloRLConfig):
    """Train a new Dablo RL model using configuration."""
    print("=== Dablo RL Training ===")
    print(f"Training for {config.training_config.total_timesteps} timesteps")
    print(f"Model will be saved to: {config.checkpoint_config.model_path}")

    # Initialize training manager with self-play configuration
    manager = DabloSelfPlayManager(
        env_config=config.to_env_dict(),
        reward_config=config.to_reward_dict(),
        model_save_path=config.checkpoint_config.model_path,
        opponent_model_path=config.selfplay_config.opponent_model_path if config.selfplay_config.enabled and not config.selfplay_config.adaptive else None,
        adaptive_selfplay=config.selfplay_config.enabled and config.selfplay_config.adaptive,
        selfplay_strategy=config.selfplay_config.selection_strategy,
    )

    # Create training environment
    print("Creating training environment...")
    env = manager.create_vec_environment(n_envs=config.training_config.n_envs)

    # Create model
    print("Initializing MaskablePPO model...")
    model = manager.create_model(
        env=env, verbose=config.verbose, **config.to_training_dict()
    )

    # Create training monitor
    monitor = DabloTrainingMonitor(verbose=config.verbose)

    # Train model
    print("Starting training...")
    trained_model = manager.train_model(
        model=model,
        total_timesteps=config.training_config.total_timesteps,
        eval_freq=config.evaluation_config.eval_freq,
        n_eval_episodes=config.evaluation_config.n_eval_episodes,
        save_freq=config.checkpoint_config.save_freq,
        callback=monitor,
    )

    # Final evaluation
    if config.evaluation_config.final_eval:
        print("Running final evaluation...")
        results = manager.evaluate_model(
            trained_model,
            n_eval_episodes=config.evaluation_config.final_eval_episodes,
            deterministic=config.evaluation_config.deterministic,
            render=config.render_eval,
        )
        print(f"Final evaluation results: {results}")

    print("Training completed!")
    return trained_model


def evaluate_model(config: DabloRLConfig, model_path: str):
    """Evaluate an existing trained model."""
    print("=== Dablo RL Model Evaluation ===")
    print(f"Loading model from: {model_path}")

    # Initialize manager
    manager = DabloSelfPlayManager(
        env_config=config.to_env_dict(), reward_config=config.to_reward_dict()
    )

    # Load and evaluate model
    model = manager.load_model(model_path)
    results = manager.evaluate_model(
        model,
        n_eval_episodes=config.evaluation_config.final_eval_episodes,
        deterministic=config.evaluation_config.deterministic,
        render=config.render_eval,
    )

    print("Evaluation completed!")
    return results


def create_sample_configs():
    """Create sample configuration files."""
    # Quick training config
    quick_config = DabloRLConfig.create_quick_training()
    quick_config.save_to_file("configs/quick_training.json")

    # Production training config
    production_config = DabloRLConfig.create_production_training()
    production_config.save_to_file("configs/production_training.json")

    # Default config
    default_config = DabloRLConfig()
    default_config.save_to_file("configs/default.json")

    print("Sample configuration files created in 'configs/' directory:")
    print("  - quick_training.json: Fast training for testing")
    print("  - production_training.json: Full-scale training")
    print("  - default.json: Default parameters")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train Dablo RL agents using configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create sample config files
  python train_dablo_rl_v2.py --create-configs
  
  # Quick training with test config
  python train_dablo_rl_v2.py --config configs/quick_training.json
  
  # Production training
  python train_dablo_rl_v2.py --config configs/production_training.json
  
  # Evaluate a trained model
  python train_dablo_rl_v2.py --config configs/default.json --evaluate models/final_model.zip
        """,
    )

    # Main arguments
    parser.add_argument(
        "--config", type=str, help="Path to configuration file (JSON or YAML)"
    )
    parser.add_argument(
        "--create-configs",
        action="store_true",
        help="Create sample configuration files and exit",
    )
    parser.add_argument(
        "--evaluate",
        type=str,
        metavar="MODEL_PATH",
        help="Evaluate existing model instead of training",
    )

    # Quick options (alternative to config file)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use quick training configuration (overrides --config)",
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="Use production training configuration (overrides --config)",
    )

    # Override options
    parser.add_argument(
        "--timesteps", type=int, help="Override total training timesteps"
    )
    parser.add_argument(
        "--verbose", type=int, choices=[0, 1, 2], help="Override verbosity level"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render games during evaluation"
    )

    args = parser.parse_args()

    # Handle config creation
    if args.create_configs:
        create_sample_configs()
        return 0

    # Load or create configuration
    try:
        if args.quick:
            config = DabloRLConfig.create_quick_training()
            print("Using quick training configuration")
        elif args.production:
            config = DabloRLConfig.create_production_training()
            print("Using production training configuration")
        elif args.config:
            config = DabloRLConfig.load_from_file(args.config)
            print(f"Loaded configuration from: {args.config}")
        else:
            config = DabloRLConfig()
            print("Using default configuration")

        # Apply command-line overrides
        if args.timesteps:
            config.training_config.total_timesteps = args.timesteps
            print(f"Override: total_timesteps = {args.timesteps}")
        if args.verbose is not None:
            config.verbose = args.verbose
            print(f"Override: verbose = {args.verbose}")
        if args.render:
            config.render_eval = True
            print("Override: render_eval = True")

        # Execute training or evaluation
        if args.evaluate:
            results = evaluate_model(config, args.evaluate)
        else:
            model = train_model(config)

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose and args.verbose >= 2:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
