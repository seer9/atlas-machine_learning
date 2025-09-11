#!/usr/bin/env python3
"""using the frostlake environment in gym"""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """loads the frostlake environment from gym

    Args:
        desc: A list. A custom description of the map to load.
            Defaults to None.
        map_name: A str. The name of the map to load.
            Defaults to None.
        is_slippery: Whether to make the surface slippery.
            Defaults to False.

    Returns:
        gym.Env: The loaded frostlake environment.
    """
    return gym.make('FrozenLake-v1', desc=desc, render_mode="ansi")
                    