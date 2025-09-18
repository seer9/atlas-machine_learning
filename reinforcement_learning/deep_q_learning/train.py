#!/usr/bin/env python3
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
import numpy as np


# Wrapper for compatibility with keras-rl
class RLWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = np.clip(reward, -1, 1)
        done = terminated or truncated
        return obs, reward, done, info

# Function to create the Atari environment
def make_atari_env(env_name):
    env = gym.make(env_name, render_mode="rgb_array")
    env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True)
    env = FrameStack(env, num_stack=4)
    env = RLWrapper(env)
    return env

# Function to build the DQN model
def build_model(input_shape, actions):
    model = Sequential()
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation="relu", input_shape=input_shape, data_format="channels_first"))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation="relu"))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(actions, activation="linear"))
    return model

# Function to configure the DQN agent
def configure_agent(env, model):
    nb_actions = env.action_space.n
    memory = SequentialMemory(limit=1000000, window_length=1)
    policy = EpsGreedyQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=50000,
                   target_model_update=10000, policy=policy, gamma=0.99, train_interval=4, delta_clip=1.0)
    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])
    return dqn

# Main training function
def train_dqn():
    env = make_atari_env("ALE/Breakout-v5")
    input_shape = (4, 84, 84)
    nb_actions = env.action_space.n

    model = build_model(input_shape, nb_actions)
    dqn = configure_agent(env, model)

    dqn.fit(env, nb_steps=1750000, log_interval=10000)
    dqn.save_weights("dqn_breakout_weights.h5f", overwrite=True)
    dqn.model.save("policy.h5")
    env.close()

if __name__ == "__main__":
    train_dqn()
