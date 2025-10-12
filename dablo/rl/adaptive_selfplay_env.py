"""
Adaptive self-play environment with dynamic opponent selection.

Uses OpponentPool to select different opponent models dynamically,
ensuring the learning agent faces appropriate challenges.
"""

from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np

from .opponent_pool import OpponentPool
from .selfplay_env import DabloSelfPlayEnvironment


class DabloAdaptiveSelfPlayEnvironment(gym.Env):
    """
    Self-play environment with adaptive opponent selection.
    
    Dynamically selects opponent models from a pool based on performance
    and training objectives.
    """
    
    metadata = {"render_modes": ["human", "text"], "render_fps": 1}
    
    def __init__(
        self,
        model_directory: str = "./models",
        selection_strategy: str = "mixed",
        refresh_pool_frequency: int = 100,  # Episodes between pool refreshes
        **kwargs
    ):
        """
        Initialize adaptive self-play environment.
        
        Args:
            model_directory: Directory containing opponent models
            selection_strategy: Strategy for opponent selection
            refresh_pool_frequency: How often to refresh the opponent pool
            **kwargs: Arguments passed to base self-play environment
        """
        super().__init__()
        
        # Initialize opponent pool
        self.opponent_pool = OpponentPool(
            model_directory=model_directory,
            selection_strategy=selection_strategy
        )
        
        self.refresh_pool_frequency = refresh_pool_frequency
        self.episodes_since_refresh = 0
        
        # Current episode state
        self.current_opponent_path = None
        self.current_selfplay_env = None
        
        # Environment kwargs for creating self-play environments
        self.env_kwargs = kwargs
        
        # Get action/observation spaces from a temporary environment
        temp_opponent = self.opponent_pool.select_opponent()
        if temp_opponent:
            temp_env = DabloSelfPlayEnvironment(temp_opponent, **kwargs)
            self.action_space = temp_env.action_space
            self.observation_space = temp_env.observation_space
            temp_env.close()
        else:
            raise RuntimeError("No opponent models found for adaptive self-play")
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset with new opponent selection."""
        # Simple approach: only refresh occasionally and quietly
        self.episodes_since_refresh += 1
        
        # Refresh much less frequently (every 1000 episodes across all envs)
        if self.episodes_since_refresh % 1000 == 0:
            self.opponent_pool.refresh_pool(verbose=False)
            print(f"🔄 Opponent pool refreshed after {self.episodes_since_refresh} episodes")
        
        # Select new opponent for this episode
        new_opponent_path = self.opponent_pool.select_opponent()
        
        if not new_opponent_path:
            raise RuntimeError("No opponent models available")
        
        # Create new self-play environment if opponent changed
        if new_opponent_path != self.current_opponent_path:
            if self.current_selfplay_env:
                self.current_selfplay_env.close()
            
            self.current_opponent_path = new_opponent_path
            self.current_selfplay_env = DabloSelfPlayEnvironment(
                opponent_model_path=new_opponent_path,
                **self.env_kwargs
            )
            
            opponent_name = self.opponent_pool.opponent_models[new_opponent_path]['name']
            win_rate = self.opponent_pool.opponent_models[new_opponent_path]['win_rate']
            print(f"🎯 Selected opponent: {opponent_name} (win rate: {win_rate:.1%})")
        
        return self.current_selfplay_env.reset(seed=seed, options=options)
    
    def step(self, action):
        """Execute step and track opponent performance."""
        if not self.current_selfplay_env:
            raise RuntimeError("Environment not reset")
        
        obs, reward, terminated, truncated, info = self.current_selfplay_env.step(action)
        
        # Update opponent performance if episode ended
        if terminated or truncated:
            agent_won = reward > 0  # Positive reward means agent won
            agent_was_p1 = info.get('agent_player') == 'P1 (Aalmogh)'
            
            self.opponent_pool.update_opponent_performance(
                self.current_opponent_path,
                agent_won=agent_won,
                agent_was_p1=agent_was_p1
            )
            
            # Add pool stats to info
            pool_stats = self.opponent_pool.get_pool_stats()
            info.update({f'pool_{k}': v for k, v in pool_stats.items()})
        
        return obs, reward, terminated, truncated, info
    
    def action_masks(self) -> np.ndarray:
        """Get action masks from current environment."""
        if not self.current_selfplay_env:
            raise RuntimeError("Environment not reset")
        return self.current_selfplay_env.action_masks()
    
    def render(self):
        """Render with opponent pool information."""
        if not self.current_selfplay_env:
            return None
            
        base_render = self.current_selfplay_env.render()
        
        if base_render and self.current_selfplay_env.base_env.render_mode == "text":
            opponent_info = self.opponent_pool.opponent_models[self.current_opponent_path]
            pool_stats = self.opponent_pool.get_pool_stats()
            
            additional_info = (
                f"\n--- Adaptive Self-Play ---"
                f"\nOpponent: {opponent_info['name']} ({opponent_info['win_rate']:.1%} win rate)"
                f"\nPool: {pool_stats['pool_size']} models, strategy: {pool_stats['strategy']}"
                f"\nGames vs current opponent: {opponent_info['games_played']}"
            )
            base_render += additional_info
        
        return base_render
    

    def close(self):
        """Close environment."""
        if self.current_selfplay_env:
            self.current_selfplay_env.close()
    
    def get_wrapper_attr(self, name: str) -> Any:
        """Get attribute from current self-play environment."""
        if not self.current_selfplay_env:
            raise RuntimeError("Environment not reset")
        return self.current_selfplay_env.get_wrapper_attr(name)
    
    def get_opponent_pool_stats(self) -> Dict:
        """Get detailed opponent pool statistics."""
        return self.opponent_pool.get_pool_stats()