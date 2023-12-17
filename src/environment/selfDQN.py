import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_buffer = deque(maxlen=5000)
        self.gamma = 0.9
        self.epsilon = 0.8
        self.update_rate = 1000
        self.main_network = self.build_model()
        self.target_network = self.build_model()
        self.target_network.set_weights(self.main_network.get_weights())

    def build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=np.prod(self.state_size), activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam())
        return model

    def store_transition(self, state, action, reward, next_state, done):
        # print(state, action, reward, next_state, done)
        self.replay_buffer.append((state, action, reward, next_state, done))

    def epsilon_greedy(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        Q_values = self.main_network.predict(np.array([state]), verbose=0)
        return np.argmax(Q_values[0])

    def train(self, batch_size):
        minibatch = np.array(random.sample(self.replay_buffer, batch_size))
        states = np.vstack(minibatch[:, 0])
        actions = minibatch[:, 1]
        rewards = minibatch[:, 2]
        next_states = np.vstack(minibatch[:, 3])
        dones = minibatch[:, 4]

        targets = self.main_network.predict(states)
        targets[range(batch_size), actions] = rewards + (
            1 - dones
        ) * self.gamma * np.amax(self.target_network.predict(next_states), axis=1)

        self.main_network.fit(states, targets, epochs=1)

    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())
