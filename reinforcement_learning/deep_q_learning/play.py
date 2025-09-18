#!/usr/bin/env python3
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
from tensorflow.keras.models import load_model
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory
from tensorflow.keras.optimizers import Adam

# Wrapper for compatibility with keras-rl
class RLWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

# Function to create the Atari environment
def make_atari_env(env_name):
    env = gym.make(env_name, render_mode="human")
    env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True)
    env = FrameStack(env, num_stack=4)
    env = RLWrapper(env)
    return env

# Function to configure the DQN agent for testing
def configure_agent(env, model):
    nb_actions = env.action_space.n
    memory = SequentialMemory(limit=200000, window_length=1)
    policy = GreedyQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, policy=policy,
                   nb_steps_warmup=0, target_model_update=1e-2, gamma=0.99)
    dqn.compile(Adam(learning_rate=5e-4), metrics=["mae"])
    return dqn

# Main function to play the game
def play_dqn():
    env = make_atari_env("ALE/Breakout-v5")
    model = load_model("policy.h5")
    dqn = configure_agent(env, model)

    dqn.test(env, nb_episodes=5, visualize=True)
    env.close()

if __name__ == "__main__":
    play_dqn()

