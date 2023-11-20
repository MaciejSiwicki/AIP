from custom_env import CustomEnv
import gym
import numpy as np


if __name__ == "__main__":
    env = CustomEnv()
    observation = env.reset()
    done = False

    while not done:
        action = None
        if env.pygame.pacman.overshotTarget():
            action = np.random.randint(-2, 3)
        observation, reward, done, info = env.step(action)
        env.render()
