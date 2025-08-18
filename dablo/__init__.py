"""
Dablo - Traditional Sami Board Game with RL Agent

A Python implementation of the traditional Sami board game Dablo,
complete with reinforcement learning training capabilities.
"""

__version__ = "0.1.0"
__author__ = "Gieril Ánde E. Lindi"

from .core.game import DabloGame
from .env.environment import DabloEnv


__all__ = ["DabloGame", "DabloEnv"]
