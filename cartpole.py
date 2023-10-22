import numpy as np
import gym
import random


def main():
    env = gym.make("Taxi-v3")

    # initialize q-table
    state_size = env.observation_space.n  # type: ignore
    action_size = env.action_space.n  # type: ignore
    qtable = np.zeros((state_size, action_size))

    learning_rate = 0.9
    discount_rate = 0.8
    epsilon = 1.0
    decay_rate = 0.005

    # training variables
    num_episodes = 1000
    max_steps = 99

    # training
    for episode in range(num_episodes):
        # reset the environment
        state = env.reset()[0]
        # print(state)
        done = False

        for s in range(max_steps):
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(qtable[state, :])

            new_state, reward, done, info, _ = env.step(action)

            # Q-learning algorithm
            qtable[state, action] = qtable[state, action] + learning_rate * (
                reward
                + discount_rate * np.max(qtable[new_state, :])
                - qtable[state, action]
            )

            state = new_state

            if done == True:
                break

        # Decrease epsilon
        epsilon = np.exp(-decay_rate * episode)

    input("Press Enter to watch trained agent...")

    env.close()
    env = gym.make("Taxi-v3", render_mode="human")
    state = env.reset()[0]
    done = False
    rewards = 0

    for s in range(max_steps):
        print(f"TRAINED AGENT")
        print("Step {}".format(s + 1))

        action = np.argmax(qtable[state, :])
        new_state, reward, done, info, _ = env.step(action)
        rewards += reward
        env.render()
        print(f"score: {rewards}")
        state = new_state

        if done == True:
            break

    env.close()


if __name__ == "__main__":
    main()
