import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def run(is_training=True, render=False):
    env = gym.make("CartPole-v1", render_mode="human" if render else None)

    ############# DATA ################
    learning_rate = 0.1  # alpha
    discount_factor = 0.99  # gamma
    epsilon = 1
    epsilon_decay_rate = 0.00001
    rng = np.random.default_rng()

    pos_bin = np.linspace(-2.4, 2.4, 10)
    vel_bin = np.linspace(-4, 4, 10)
    ang_bin = np.linspace(-0.2095, 0.2095, 10)
    ang_vel_bin = np.linspace(-4, 4, 10)
    ###################################

    if is_training:
        q = np.zeros(
            (
                len(pos_bin) + 1,
                len(vel_bin) + 1,
                len(ang_bin) + 1,
                len(ang_vel_bin) + 1,
                env.action_space.n,
            )
        )

    rewards_per_episode = []
    i = 0

    while True:
        state = env.reset()[0]
        state_p = np.digitize(state[0], pos_bin)
        state_v = np.digitize(state[1], vel_bin)
        state_a = np.digitize(state[2], ang_bin)
        state_av = np.digitize(state[3], ang_vel_bin)

        terminated = False
        rewards = 0

        while not terminated and rewards < 10000:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_p, state_v, state_a, state_av, :])

            new_state, reward, terminated, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_bin)
            new_state_v = np.digitize(new_state[1], vel_bin)
            new_state_a = np.digitize(new_state[2], ang_bin)
            new_state_av = np.digitize(new_state[3], ang_vel_bin)

            if is_training:
                q[state_p, state_v, state_a, state_av, action] = q[
                    state_p, state_v, state_a, state_av, action
                ]
                +learning_rate * (
                    reward
                    + discount_factor
                    * np.max(q[new_state_p, new_state_v, new_state_a, new_state_av, :])
                    - q[state_p, state_v, state_a, state_av, action]
                )

            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            state_a = new_state_a
            state_av = new_state_av

            rewards += reward

            if not is_training:
                print(f"Episode: {i}  Rewards: {rewards}")

        rewards_per_episode.append(rewards)
        mean_rewards = np.mean(rewards_per_episode[len(rewards_per_episode) - 100 :])

        if is_training and i % 500 == 0:
            print(
                f"Episode: {i} {rewards}  | Epsilon: {epsilon:0.2f} | Mean Rewards {mean_rewards:0.1f}"
            )

        if mean_rewards > 1000:
            break

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        i += 1

    env.close()

    mean_rewards = []
    for t in range(i):
        mean_rewards.append(np.mean(rewards_per_episode[max(0, t - 100) : (t + 1)]))
    plt.plot(mean_rewards)
    plt.savefig(f"cartpole.png")


if __name__ == "__main__":
    run(is_training=True, render=False)
    # run(is_training=False, render=True)
