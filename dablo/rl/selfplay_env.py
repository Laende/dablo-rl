"""
Self-play environment for training against previous model versions.

This environment allows the learning agent to play against a fixed opponent model,
with alternating roles to ensure balanced training.
"""

import random
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO

from ..core.config import DabloConfig
from ..core.game import DabloGame
from ..core.player import Player
from ..core.rules import Move
from .environment import DabloRLEnvironment


class DabloSelfPlayEnvironment(gym.Env):
    """
    Self-play environment where agent trains against a fixed opponent model.
    
    The agent alternates between playing as P1 and P2, while the opponent
    model plays the other side. This ensures balanced training data.
    """
    
    metadata = {"render_modes": ["human", "text"], "render_fps": 1}
    
    def __init__(
        self,
        opponent_model_path: str,
        config: Optional[DabloConfig] = None,
        render_mode: Optional[str] = None,
        reward_config: Optional[Dict[str, float]] = None,
        opponent_randomness: float = 0.0,
    ):
        """
        Initialize self-play environment.
        
        Args:
            opponent_model_path: Path to opponent model file
            config: Game configuration
            render_mode: Rendering mode
            reward_config: Reward calculation parameters
            opponent_randomness: Probability of opponent making random moves (0.0 = always use model)
        """
        super().__init__()
        
        # Create base environment for spaces and game mechanics
        self.base_env = DabloRLEnvironment(
            config=config,
            render_mode=render_mode,
            reward_config=reward_config
        )
        
        # Copy spaces from base environment
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space
        
        # Load opponent model
        try:
            self.opponent_model = MaskablePPO.load(opponent_model_path)
            print(f"✓ Loaded opponent model from: {opponent_model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load opponent model from {opponent_model_path}: {e}")
        
        # Self-play configuration
        self.opponent_randomness = opponent_randomness
        self.episode_count = 0
        
        # Game state
        self.game = None
        self.agent_player = Player.P1  # Which player the learning agent controls
        self.opponent_player = Player.P2  # Which player the opponent model controls
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment and determine player roles for this episode."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Alternate who the agent plays as each episode
        self.episode_count += 1
        if self.episode_count % 2 == 1:
            self.agent_player = Player.P1
            self.opponent_player = Player.P2
        else:
            self.agent_player = Player.P2  
            self.opponent_player = Player.P1
        
        # Reset the base environment
        obs, info = self.base_env.reset(seed=seed, options=options)
        self.game = self.base_env.game
        
        # If opponent goes first, make opponent move(s) until it's agent's turn
        while self.game.current_player == self.opponent_player and not self.game.game_over:
            opponent_move = self._get_opponent_move()
            if opponent_move is None:
                break
                
            # Apply opponent move to base environment
            opponent_action = self._move_to_action(opponent_move)
            obs, reward, terminated, truncated, info = self.base_env.step(opponent_action)
            
            if terminated or truncated:
                # Game ended on opponent move, return terminal state
                # Flip reward since it was calculated for opponent
                return obs, -reward, terminated, truncated, info
        
        # Add self-play info
        info['agent_player'] = self.agent_player.name
        info['opponent_player'] = self.opponent_player.name
        
        return obs, info
    
    def step(self, action):
        """Execute agent action, then opponent response if needed."""
        # Agent makes move
        obs, reward, terminated, truncated, info = self.base_env.step(action)
        
        # If game ended, return results
        if terminated or truncated:
            # Adjust reward based on agent's player role
            if self.agent_player == Player.P2:
                reward = -reward  # Flip reward for P2
            
            info['agent_player'] = self.agent_player.name
            info['opponent_player'] = self.opponent_player.name
            return obs, reward, terminated, truncated, info
        
        # Opponent's turn - make opponent move(s)
        opponent_moves_made = 0
        max_opponent_moves = 10  # Prevent infinite loops in capture sequences
        
        while (self.game.current_player == self.opponent_player and 
               not self.game.game_over and 
               opponent_moves_made < max_opponent_moves):
            
            opponent_move = self._get_opponent_move()
            if opponent_move is None:
                break
            
            # Apply opponent move
            opponent_action = self._move_to_action(opponent_move)
            obs, opp_reward, terminated, truncated, info = self.base_env.step(opponent_action)
            opponent_moves_made += 1
            
            if terminated or truncated:
                # Game ended on opponent move
                # Adjust reward for agent's perspective
                if self.agent_player == Player.P2:
                    reward = -opp_reward
                else:
                    reward = opp_reward
                
                info['agent_player'] = self.agent_player.name
                info['opponent_player'] = self.opponent_player.name
                return obs, reward, terminated, truncated, info
        
        # Game continues, it's agent's turn again
        info['agent_player'] = self.agent_player.name  
        info['opponent_player'] = self.opponent_player.name
        return obs, 0.0, False, False, info  # No reward for intermediate states
    
    def _get_opponent_move(self) -> Optional[Move]:
        """Get move from opponent model."""
        if random.random() < self.opponent_randomness:
            # Make random move
            valid_moves = self.game.get_all_valid_moves()
            return random.choice(valid_moves) if valid_moves else None
        
        # Get move from opponent model
        try:
            obs = self.base_env._get_observation()
            action_masks = self.base_env.action_masks()
            
            action, _ = self.opponent_model.predict(
                obs, 
                action_masks=action_masks,
                deterministic=True  # Opponent plays deterministically
            )
            
            return self._action_to_move(action)
            
        except Exception as e:
            print(f"Warning: Opponent model prediction failed: {e}")
            # Fallback to random move
            valid_moves = self.game.get_all_valid_moves()
            return random.choice(valid_moves) if valid_moves else None
    
    def _move_to_action(self, move: Move) -> np.ndarray:
        """Convert Move object to action array."""
        from_idx = self.base_env.action_space_manager.pos_to_idx[move.from_pos]
        to_idx = self.base_env.action_space_manager.pos_to_idx[move.to_pos]
        action_idx = from_idx * 50 + to_idx
        return np.array([action_idx])
    
    def _action_to_move(self, action) -> Move:
        """Convert action array to Move object."""
        action_idx = int(action)
        from_idx = action_idx // 50
        to_idx = action_idx % 50
        
        from_pos = self.base_env.action_space_manager.idx_to_pos[from_idx]
        to_pos = self.base_env.action_space_manager.idx_to_pos[to_idx]
        
        return Move(from_pos, to_pos)
    
    def action_masks(self) -> np.ndarray:
        """Get action masks for current state."""
        return self.base_env.action_masks()
    
    def render(self):
        """Render current state with self-play info."""
        base_render = self.base_env.render()
        if base_render and self.base_env.render_mode == "text":
            base_render += f"\nAgent: {self.agent_player.name} | Opponent: {self.opponent_player.name}"
        return base_render
    
    def close(self):
        """Close environment."""
        self.base_env.close()
    
    def get_wrapper_attr(self, name: str) -> Any:
        """Get attribute from base environment."""
        return self.base_env.get_wrapper_attr(name)