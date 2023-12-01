import gym
from gym import spaces
import numpy as np
from run import GameController
from constants import *


class CustomEnv(gym.Env):
    # metadata = {'render.modes' : ['human']}
    def __init__(self):
        gamecontroller = GameController()
        max_x = SCREENWIDTH
        max_y = SCREENHEIGHT
        num_ghosts = 4
        eatenPellets = 244
        self.pygame = gamecontroller.startGame()

        # 4 possible movements - -2 right, 2 left, 1 up, -1 down 0 stop
        self.action_space = spaces.Discrete(5, start=-2)
        print(self.action_space)

        # observation space has to be modified!
        self.observation_space = spaces.Tuple(
            tuple(
                spaces.Tuple(
                    [spaces.Discrete(SCREENWIDTH), spaces.Discrete(SCREENHEIGHT)]
                )
                for _ in range(num_ghosts + 1)
            )
            + (spaces.Discrete(max(eatenPellets, 1)),)
        )
        print(self.observation_space)

    def reset(self):
        del self.pygame
        gamecontroller = GameController()
        self.pygame = gamecontroller.startGame()
        obs = self.pygame.observe()
        return obs

    def step(self, action):
        self.pygame.action(action)
        obs = self.pygame.observe()
        reward = self.pygame.evaluate()
        done = self.pygame.is_done()
        return obs, reward, done, {}

    def render(self, mode):
        self.pygame.update(mode)

    def close(self):
        pass
