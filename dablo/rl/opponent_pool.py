"""
Adaptive opponent selection system for self-play training.

Manages a pool of opponent models and selects them based on different strategies
like playing against the best model, random selection, or curriculum learning.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import glob

from sb3_contrib import MaskablePPO


class OpponentPool:
    """Manages a pool of opponent models for self-play training."""
    
    def __init__(
        self,
        model_directory: str = "./models",
        selection_strategy: str = "best",
        max_pool_size: int = 10,
        include_current_best: bool = True
    ):
        """
        Initialize opponent pool.
        
        Args:
            model_directory: Directory containing model files
            selection_strategy: Strategy for selecting opponents
                - "best": Always use the best performing model
                - "random": Random selection from pool
                - "curriculum": Start with weaker, progress to stronger
                - "mixed": Combination of strategies
            max_pool_size: Maximum number of models to keep in pool
            include_current_best: Whether to always include the current best model
        """
        self.model_directory = Path(model_directory)
        self.selection_strategy = selection_strategy
        self.max_pool_size = max_pool_size
        self.include_current_best = include_current_best
        
        # Pool state
        self.opponent_models: Dict[str, Dict] = {}  # model_path -> metadata
        self.loaded_models: Dict[str, MaskablePPO] = {}  # Cache loaded models
        self.selection_weights: Dict[str, float] = {}  # For weighted selection
        
        # Performance tracking
        self.model_performance: Dict[str, float] = {}  # model_path -> win_rate vs current
        self.games_played_against: Dict[str, int] = {}  # Track games per opponent
        
        # Initialize pool
        self.refresh_pool()
    
    def refresh_pool(self, verbose: bool = False) -> None:
        """Scan model directory and update opponent pool."""
        if verbose:
            print(f"Refreshing opponent pool from {self.model_directory}")
        
        # Find all model files
        model_patterns = ["*.zip", "best_model.zip", "checkpoint_*.zip", "final_model.zip"]
        model_files = []
        
        for pattern in model_patterns:
            model_files.extend(glob.glob(str(self.model_directory / pattern)))
        
        # Remove duplicates and sort by modification time
        model_files = list(set(model_files))
        model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        if verbose:
            print(f"Found {len(model_files)} potential opponent models")
        
        # Update pool with model metadata
        new_opponents = {}
        for model_path in model_files[:self.max_pool_size]:
            model_name = Path(model_path).stem
            model_stats = os.stat(model_path)
            
            new_opponents[model_path] = {
                'name': model_name,
                'path': model_path,
                'size': model_stats.st_size,
                'modified': model_stats.st_mtime,
                'win_rate': self.model_performance.get(model_path, 0.5),  # Default 50%
                'games_played': self.games_played_against.get(model_path, 0)
            }
        
        # Update pool
        self.opponent_models = new_opponents
        
        # Remove models no longer in pool from cache
        for model_path in list(self.loaded_models.keys()):
            if model_path not in self.opponent_models:
                del self.loaded_models[model_path]
        
        self._update_selection_weights()
        
        if verbose:
            print(f"Opponent pool updated: {len(self.opponent_models)} models available")
            for model_path, info in list(self.opponent_models.items())[:5]:  # Show top 5
                print(f"  - {info['name']}: {info['win_rate']:.1%} win rate, {info['games_played']} games")
    
    def select_opponent(self) -> Optional[str]:
        """Select opponent model path based on strategy."""
        if not self.opponent_models:
            print("Warning: No opponent models available")
            return None
        
        if self.selection_strategy == "best":
            return self._select_best()
        elif self.selection_strategy == "random":
            return self._select_random()
        elif self.selection_strategy == "curriculum":
            return self._select_curriculum()
        elif self.selection_strategy == "mixed":
            return self._select_mixed()
        else:
            print(f"Unknown selection strategy: {self.selection_strategy}, using random")
            return self._select_random()
    
    def _select_best(self) -> str:
        """Select the highest performing opponent."""
        return max(self.opponent_models.keys(), 
                  key=lambda x: self.opponent_models[x]['win_rate'])
    
    def _select_random(self) -> str:
        """Randomly select opponent."""
        return random.choice(list(self.opponent_models.keys()))
    
    def _select_curriculum(self) -> str:
        """Select opponent based on curriculum learning (weaker to stronger)."""
        # Sort by win rate (ascending for curriculum learning)
        sorted_opponents = sorted(
            self.opponent_models.keys(),
            key=lambda x: self.opponent_models[x]['win_rate']
        )
        
        # Simple curriculum: bias toward easier opponents early in training
        # This could be made more sophisticated based on training progress
        n_opponents = len(sorted_opponents)
        
        # Bias toward easier opponents (lower indices)
        weights = [n_opponents - i for i in range(n_opponents)]
        return random.choices(sorted_opponents, weights=weights)[0]
    
    def _select_mixed(self) -> str:
        """Mixed strategy: mostly best, some random, some curriculum."""
        strategy_choice = random.choices(
            ['best', 'random', 'curriculum'],
            weights=[0.5, 0.3, 0.2]  # 50% best, 30% random, 20% curriculum
        )[0]
        
        if strategy_choice == 'best':
            return self._select_best()
        elif strategy_choice == 'random':
            return self._select_random()
        else:
            return self._select_curriculum()
    
    def _update_selection_weights(self) -> None:
        """Update selection weights based on model performance and diversity."""
        total_games = sum(info['games_played'] for info in self.opponent_models.values())
        
        for model_path, info in self.opponent_models.items():
            # Base weight on performance (higher = more challenging)
            performance_weight = info['win_rate']
            
            # Reduce weight for overplayed opponents (encourage diversity)
            if total_games > 0:
                play_frequency = info['games_played'] / total_games
                diversity_penalty = max(0.1, 1.0 - play_frequency * 3)  # Reduce if >33% of games
            else:
                diversity_penalty = 1.0
            
            self.selection_weights[model_path] = performance_weight * diversity_penalty
    
    def load_opponent_model(self, model_path: str) -> Optional[MaskablePPO]:
        """Load opponent model, using cache if available."""
        if model_path in self.loaded_models:
            return self.loaded_models[model_path]
        
        try:
            print(f"Loading opponent model: {Path(model_path).name}")
            model = MaskablePPO.load(model_path)
            self.loaded_models[model_path] = model
            return model
        except Exception as e:
            print(f"Failed to load opponent model {model_path}: {e}")
            return None
    
    def update_opponent_performance(
        self, 
        model_path: str, 
        agent_won: bool, 
        agent_was_p1: bool
    ) -> None:
        """Update opponent performance statistics."""
        if model_path not in self.opponent_models:
            return
        
        # Update games played
        self.games_played_against[model_path] = self.games_played_against.get(model_path, 0) + 1
        
        # Update win rate (from opponent's perspective)
        opponent_won = not agent_won
        current_win_rate = self.model_performance.get(model_path, 0.5)
        games_played = self.games_played_against[model_path]
        
        # Running average of win rate
        new_win_rate = (current_win_rate * (games_played - 1) + (1.0 if opponent_won else 0.0)) / games_played
        self.model_performance[model_path] = new_win_rate
        
        # Update model metadata
        self.opponent_models[model_path]['win_rate'] = new_win_rate
        self.opponent_models[model_path]['games_played'] = games_played
    
    def get_pool_stats(self) -> Dict:
        """Get statistics about the current opponent pool."""
        if not self.opponent_models:
            return {"pool_size": 0}
        
        win_rates = [info['win_rate'] for info in self.opponent_models.values()]
        games_played = [info['games_played'] for info in self.opponent_models.values()]
        
        return {
            "pool_size": len(self.opponent_models),
            "avg_opponent_win_rate": sum(win_rates) / len(win_rates),
            "min_opponent_win_rate": min(win_rates),
            "max_opponent_win_rate": max(win_rates),
            "total_games_vs_opponents": sum(games_played),
            "strategy": self.selection_strategy
        }