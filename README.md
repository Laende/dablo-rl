# Dablo

A Python implementation of the traditional Sami board game **Dablo** with interactive gameplay and smart NPC opponents.

*This project will eventually include reinforcement learning capabilities to train AI agents that can play Dablo at various skill levels.*

## 🎮 What is Dablo?

Dablo is a family of traditional Sami board games that have been played throughout Sápmi. This implementation follows **sørsamisk dablo** (Southern Sami dablo), a "war dablo" variant documented by Anders and Hanna Nilsson from Frostviken in the early 1900s.

### Piece Types
- **Warriors** (dåarohke): Move forward only, capture other warriors
- **Princes** (gånkan elkie): Move forward only, capture warriors and princes  
- **Kings** (gånka): Move forward only, capture any piece

### Victory Conditions
You win by:
- Capturing the opponent's king
- Reducing the opponent to only their king
- Blocking all opponent moves (stalemate)

**Draw conditions:**
- Both players have only their kings remaining
- Move limit reached

### Game Features
- **Chain Captures**: Like checkers, capturing pieces can lead to mandatory chain captures
- **Strategic Depth**: Different piece types create complex tactical situations
- **Graph-based Board**: Unique board layout with primary and secondary nodes

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/Laende/dablo-rl.git
cd dablo-rl

# Install dependencies with uv
uv sync
```

### Play Interactively
```bash
python play_dablo.py
```

Choose from:
- Play vs NPC (Easy/Medium/Hard)
- Play vs Human (2 players)

## 📁 Current Project Structure

```
dablo/
├── __init__.py           # Package initialization
├── core/                 # Game engine
│   ├── __init__.py
│   ├── game.py           # Main game logic
│   ├── pieces.py         # Piece types and symbols
│   ├── player.py         # Player enum
│   ├── rules.py          # Game rules and validation
│   ├── config.py         # Game configuration
│   └── utils.py          # Utility functions
├── ai/                   # NPC opponents
│   ├── __init__.py
│   ├── players.py        # NPC implementations & performance testing
│   ├── config.py         # NPC configuration settings
│   └── base.py           # Base NPC classes
└── ui/                   # Visualization
    ├── __init__.py
    ├── visualizer.py     # Static matplotlib visualization
    └── interactive.py    # Interactive playable interface
```

## 🎯 Game Controls (Interactive Mode)

- **Click** on your pieces to select them
- **Click** on valid destination to move
- **R**: Reset game
- **Q**: Quit
- **Arrow Keys**: Navigate during chain captures

The interface highlights:
- **Green circles**: Valid moves
- **Red circles**: Capture moves
- **Lime outline**: Pieces that can move

## 🤖 NPC Opponents

Multiple NPC types with different play styles:

### NPC Types
- **Smart**: Balanced strategic play with tactical evaluation
- **Aggressive**: Prioritizes captures and forward movement
- **Defensive**: Focuses on king safety and piece protection
- **Random**: Unpredictable moves for casual play

### Difficulty Levels
- **Easy**: Basic strategy with more randomness
- **Medium**: Balanced strategic play
- **Hard**: Advanced strategy with minimal randomness

All strategic NPCs use intelligent evaluation rather than random moves, making them challenging opponents for learning the game.

### Performance Testing
Test NPC strategies against each other:
```bash
python -m dablo.ai.performance
```
Runs comprehensive matchups between different NPC types and difficulties, showing win rates and detailed statistics.

## 🔮 Future Plans

This project is designed to eventually support:
- **Reinforcement Learning**: Train AI agents using various RL algorithms
- **Multiple Environments**: Parallel training environments
- **Training Monitoring**: Tensorboard integration for training progress

The current game engine provides a solid foundation for these future RL capabilities.

## 🎲 Try It Out!

Start playing immediately:
```bash
python play_dablo.py
```

The game includes helpful visual feedback and supports both human vs human and human vs NPC gameplay!