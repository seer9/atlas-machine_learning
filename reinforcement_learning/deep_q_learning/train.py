#!/usr/bin/env python3
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import Adam
from keras_rl2.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy


# wrapper for keras-rl
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

def make_env(env_name):
    env = gym.make(env_name, render_mode="rgb_array")
    env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True, frame_skip=4)
    env = RLWrapper(env)
    return env

# DQN model architecture
def build_model(input_shape, actions):
    model = Sequential([
        Conv2D(32, (8, 8), strides=(4, 4), activation="relu",
               input_shape=input_shape),
        Conv2D(64, (4, 4), strides=(2, 2), activation="relu"),
        Conv2D(64, (3, 3), strides=(1, 1), activation="relu"),
        Flatten(),
        Dense(512, activation="relu"),
        Dense(actions, activation="linear")
    ])
    return model

# setup DQN agent
def configure_agent(env, model):
    memory = SequentialMemory(limit=1000000, window_length=4)
    policy = EpsGreedyQPolicy()
    dqn = DQNAgent(model=model,
                   nb_actions=env.action_space.n,
                   memory=memory,
                   nb_steps_warmup=50000,
                   target_model_update=10000,
                   policy=policy,
                   gamma=0.99,
                   train_interval=4,
                   delta_clip=1.0)

    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])
    return dqn

# train the agent
def train():
    env = make_env("ALE/Breakout-v5")
    model = build_model((4, 84, 84), env.action_space.n)
    agent = configure_agent(env, model)
    agent.fit(env, nb_steps=1750000, log_interval=10000)
    agent.model.save("policy.h5")
    env.close()

if __name__ == "__main__":
    train()
