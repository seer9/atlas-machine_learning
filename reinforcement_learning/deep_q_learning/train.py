#!/usr/bin/env python3
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from keras_rl2.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy

# Wrapper for compatibility with keras-rl
class RLWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, np.clip(reward, -1, 1), terminated or truncated, info

# Create Atari environment
def make_env(env_name):
    env = gym.make(env_name, render_mode="rgb_array")
    return RLWrapper(AtariPreprocessing(env, grayscale_obs=True, scale_obs=True))

# Build DQN model
def build_model(input_shape, actions):
    model = Sequential([
        Conv2D(32, (8, 8), strides=(4, 4), activation="relu", input_shape=input_shape, data_format="channels_first"),
        Conv2D(64, (4, 4), strides=(2, 2), activation="relu"),
        Conv2D(64, (3, 3), strides=(1, 1), activation="relu"),
        Flatten(),
        Dense(512, activation="relu"),
        Dense(actions, activation="linear")
    ])
    return model

# Configure DQN agent
def configure_agent(env, model):
    memory = SequentialMemory(limit=1000000, window_length=1)
    policy = EpsGreedyQPolicy()
    dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=50000,
                   target_model_update=10000, policy=policy, gamma=0.99, train_interval=4, delta_clip=1.0)
    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])
    return dqn

# Train the agent
def train():
    env = make_env("ALE/Breakout-v5")
    model = build_model((4, 84, 84), env.action_space.n)
    agent = configure_agent(env, model)
    agent.fit(env, nb_steps=1750000, log_interval=10000)
    agent.model.save("policy.h5")
    env.close()

if __name__ == "__main__":
    train()
