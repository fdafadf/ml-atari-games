import numpy as np


class QAgentMemory:
    def __init__(self, size, state_space_shape, action_space_size):
        self.size = size
        self.pointer = 0
        self.states = np.zeros((size, *state_space_shape), dtype=np.float32)
        self.actions = np.zeros(size, dtype=np.int64)
        self.next_states = np.zeros((size, *state_space_shape), dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.bool)
        self.batch_size = 32

    def add(self, state, action, next_state, reward, is_done):
        index = self.pointer % self.size
        self.states[index] = state
        self.actions[index] = action
        self.next_states[index] = next_state
        self.rewards[index] = reward
        self.dones[index] = is_done
        self.pointer += 1

    @property
    def is_filled(self):
        return self.pointer >= self.size

    def batch(self):
        size = min(self.pointer, self.size)
        batch = np.random.choice(size, self.batch_size, replace=False)
        states = self.states[batch]
        actions = self.actions[batch]
        next_states = self.next_states[batch]
        rewards = self.rewards[batch]
        dones = self.dones[batch]
        return states, actions, next_states, rewards, dones
