from custom_env import CustomEnv
import gym
import numpy as np
import time
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

if __name__ == "__main__":
    env = CustomEnv(display=True)
    observation = env.reset()
    model = A2C("MlpPolicy", env, device="cpu")
    policy = model.policy
    start_time = time.time()
    model.learn(total_timesteps=10)
    # print(model.get_parameters())
    elapsed_time = time.time() - start_time
    print(f"Time taken for model.learn: {elapsed_time:.2f} seconds")
    model.save("pacman-a2c")
    del model
    model = A2C.load("pacman-a2c")
    env = CustomEnv(display=True)
    done = False
    while not done:
        action, _states = model.predict(observation)
        observation, reward, done, info = env.step(int(action))
        print(observation, reward, done, info)
