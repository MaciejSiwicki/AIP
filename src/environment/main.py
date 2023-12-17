from custom_env import CustomEnv
from constants import *
import gym
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from selfDQN import DQN

if __name__ == "__main__":
    env = CustomEnv()
    observation = env.reset()

    ################### DQN ###################
    # num_episodes = 30
    # num_timesteps = 40
    # num_ghosts = 4
    # state_size = 15  # Długość wektora obserwacji
    # action_size = 5
    # batch_size = 32
    # dqn = DQN(state_size, action_size)
    # time_step = 0

    # for episode in range(num_episodes):
    #     state = env.reset()  # Resetujemy stan środowiska na początku każdego epizodu
    #     for t in range(num_timesteps):
    #         env.render()
    #         time_step += 1
    #         if time_step % dqn.update_rate == 0:
    #             dqn.update_target_network()
    #         action = dqn.epsilon_greedy(state)
    #         next_state, reward, done, _ = env.step(action)
    #         next_state = np.array(next_state).flatten()  # Spłaszczamy wektor stanu
    #         dqn.store_transition(state, action, reward, next_state, done)
    #         state = next_state
    #         if time_step % dqn.update_rate == 0:
    #             dqn.update_target_network()
    #         if done:
    #             break
    # learned_policy = dqn.main_network
    # learned_policy.save("pacman-dqn")
    # print("-----")
    # state = env.reset()
    # while True:
    #     env.render()
    #     action = np.argmax(learned_policy.predict(np.array([state]), verbose=0)[0])
    #     next_state, reward, done, _ = env.step(action)
    #     state = next_state
    #     if done:
    #         break
    ################### A2C ###################

    model = A2C("MlpPolicy", env, device="cpu")
    model.learn(total_timesteps=10000)
    model.save("pacman-a2c")
    del model
    model = A2C.load("pacman-a2c")
    done = False
    while not done:
        action, _states = model.predict(observation)
        observation, reward, done, info = env.step(int(action))
        print(observation, reward, done, info)
