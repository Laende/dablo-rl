# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python implementation of the traditional Sami board game **Dablo** with reinforcement learning capabilities. The project has two main components:
1. **Core Game Engine** (`dablo/core/`) - Complete implementation of Dablo game rules and mechanics
2. **RL Training System** (`dablo/rl/`) - Gymnasium environment wrapper for training agents with Stable Baselines 3

## Common Development Commands

### Environment Setup
```bash
# Install dependencies
uv sync

# Install with dev dependencies
uv sync --group dev
```

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=dablo

# Run specific test file
pytest tests/test_core/test_game.py

# Run tests matching pattern
pytest -k "test_capture"
```

### Code Quality
```bash
# Format and lint code
ruff format
ruff check

# Fix auto-fixable issues
ruff check --fix
```

### Interactive Play
```bash
# Play Dablo interactively
python play_dablo.py

# Run performance tests between AI players
python performance_test.py
```

### RL Training
```bash
# Train RL model with default config
python train_dablo_rl.py --config configs/default.json

# Train with self-play
python train_dablo_rl.py --config configs/selfplay_training.json

# Evaluate trained model
python evaluate_model.py --model_path models/dablo_model.zip
```

## Architecture Overview

### Core Game Architecture (`dablo/core/`)
- **`DabloGame`** - Main game state manager with cached board graphs and optimized piece tracking
- **`GameRules`** - Handles move validation, capture sequences, and win conditions
- **`PieceTracker`** - Event-driven piece position tracking for performance optimization
- **`DabloConfig`** - Pydantic-based configuration with JSON support for different board setups

### RL Architecture (`dablo/rl/`)
- **`DabloRLEnvironment`** - Gymnasium wrapper implementing action masking for MaskablePPO
- **`DabloActionSpace`** - Handles complex move encoding (source→destination with capture positions)
- **`DabloObservationSpace`** - Multi-dict observation space (board state, piece counts, game flags)
- **`DabloSelfPlayManager`** - Manages opponent pools and adaptive self-play training

### Key Design Patterns
- **Caching Strategy**: Board graphs and piece setups are class-level cached for performance
- **Event-Driven Updates**: `PieceTracker` uses event callbacks to maintain piece positions efficiently
- **Configuration-Based Training**: JSON configs in `configs/` directory define all training parameters
- **Action Masking**: Invalid moves are masked during RL training for efficiency

### Game Mechanics
- **Graph-based Board**: Uses coordinate system with primary (integer) and secondary (half-integer) nodes
- **Chain Captures**: Mandatory capture sequences like checkers - pieces must continue capturing if possible
- **Piece Hierarchy**: Warriors < Princes < Kings (capture rules based on hierarchy)
- **Win Conditions**: Capture opponent's king, reduce them to king only, or cause stalemate

## Testing Strategy

The test suite uses comprehensive fixtures in `tests/conftest.py`:
- **Game State Fixtures**: `simple_capture_game`, `endgame_scenario`, `optional_chain_capture_scenario`
- **Factory Fixtures**: `game_factory` for parameterized testing
- **Assertion Helpers**: `assert_valid_move`, `assert_game_state_consistent`
- **Performance Monitoring**: Auto-detects slow tests (>1s)

## Configuration System

All training configurations are JSON-based in the `configs/` directory:
- `default.json` - Basic RL training setup
- `selfplay_training.json` - Self-play with opponent pools
- `production_training.json` - High-performance training settings
- `quick_training.json` - Fast training for development

Configs use nested structure: `game_config`, `reward_config`, `training_config`, `evaluation_config`, etc.

## Performance Considerations

- Board graphs are cached at class level to avoid recreation
- Piece tracking uses event callbacks rather than full board scans
- Action masking prevents invalid move exploration during training
- Vectorized environments (`n_envs`) supported for parallel training