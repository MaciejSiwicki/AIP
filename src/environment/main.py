from custom_env import CustomEnv
from run import GameController
from constants import *
import gym
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from tensorflow.keras.models import load_model
from selfDQN import DQN
import random
import os

if __name__ == "__main__":
    env = CustomEnv()
    observation = env.reset()
    state_size = len(env.observation_space)
    action_size = env.action_space.n
    dqn_agent = DQN(state_size, action_size)

    method = A2C  # MODIFY THIS TO CHANGE THE METHOD
    testing = True  # MODIFY THIS TO TRUE TO TEST

    learning = not testing
    if testing:
        if method == A2C:
            model = A2C.load("pacman-a2c-2mln")
            done = False
            while not done:
                action, _states = model.predict(observation)
                observation, reward, done, info = env.step(int(action))
        elif method == DQN:
            dqn_agent.load_model("DQN-8900.h5")
            state = env.reset()
            done = False
            while not done:
                state = np.reshape(observation, [1, state_size])
                action = dqn_agent.act(state)
                observation, reward, done, info = env.step(int(action))

    if learning == True:
        if method == DQN:
            num_episodes = 2000
            num_timesteps = 10000
            total_timesteps = 0  # do not change
            if dqn_agent.epsilon > dqn_agent.epsilon_min:
                for episode in range(num_episodes):
                    observation = env.reset()
                    for t in range(num_timesteps):
                        env.render()
                        t += 1
                        total_timesteps += 1
                        state = np.reshape(observation, [1, state_size])
                        action = dqn_agent.act(state)
                        next_observation, reward, done, info = env.step(action)
                        dqn_agent.train(
                            np.reshape(observation, [1, state_size]),
                            action,
                            reward,
                            np.reshape(next_observation, [1, state_size]),
                            done,
                        )
                        observation = next_observation
                        if done:
                            print(
                                f"Episode {episode} total_timestep {total_timesteps}, Epsilon {dqn_agent.epsilon}"
                            )
                            break
            filename = f"{method.__name__}-{total_timesteps}.h5"
            dqn_agent.save_model(filename)

        elif method == A2C:
            model = A2C("MlpPolicy", env, device="cpu")
            model.learn(total_timesteps=1000000)
            model.save("pacman-a2c-1mln")
            del model
