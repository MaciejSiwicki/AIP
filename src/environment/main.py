from custom_env import CustomEnv
import gym
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

if __name__ == "__main__":
    env = CustomEnv()
    observation = env.reset()
    model = A2C("MlpPolicy", env, device="cpu")
    model.learn(total_timesteps=10000000)

    model.save("pacman-a2c")
    del model
    model = A2C.load("pacman-a2c")
    done = False
    while not done:
        action, _states = model.predict(observation)
        observation, reward, done, info = env.step(int(action))
    #     print(observation, reward, done, info)
    #     env.render("human")
