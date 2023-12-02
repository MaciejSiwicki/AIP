from custom_env import CustomEnv
import gym
import numpy as np


if __name__ == "__main__":
    env = CustomEnv()
    observation = env.reset()
    done = False

    while not done:
        action = None
        observation, reward, done, info = env.step(action)
        env.render(mode="random_smart")
