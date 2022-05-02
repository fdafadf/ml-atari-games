import numpy as np
import torch as torch

from QAgentMemory import QAgentMemory
from QNetwork import QNetwork


class QAgent:
    def __init__(self, environment, gamma, learning_rate, memory_size=50000):
        self.environment = environment
        self.gamma = gamma
        self.epsilon = 1
        self.epsilon_min = 0.1
        self.epsilon_decrement = 0.00001
        self.train_iteration_count = 0
        self.replace_target_network_interval = 1000
        self.action_space = [i for i in range(environment.action_space.n)]
        self.network_eval = QNetwork(learning_rate, environment.observation_space.shape, environment.action_space.n)
        self.network_target = QNetwork(learning_rate, environment.observation_space.shape, environment.action_space.n)
        self.memory = QAgentMemory(memory_size, environment.observation_space.shape, environment.action_space.n)

    def step(self, state):
        action = self.action(state)
        next_state, reward, is_done, info = self.environment.step(action)
        self.memory.add(state, action, next_state, reward, is_done)
        return next_state, reward, is_done, info

    def action(self, state):
        if np.random.random() > self.epsilon:
            network_input = torch.tensor([state], dtype=torch.float).to(self.network_eval.device)
            network_output = self.network_eval.forward(network_input)
            return torch.argmax(network_output).item()
        else:
            return np.random.choice(self.action_space)

    def train_batch(self):
        states, actions, next_states, rewards, dones = self.memory.batch()
        t_states = torch.tensor(states).to(self.network_eval.device)
        t_actions = torch.tensor(actions).to(self.network_eval.device)
        t_next_states = torch.tensor(next_states).to(self.network_eval.device)
        t_rewards = torch.tensor(rewards).to(self.network_eval.device)
        t_dones = torch.tensor(dones).to(self.network_eval.device)
        return t_states, t_actions, t_next_states, t_rewards, t_dones

    def train(self):
        if not self.memory.is_filled:
            return
        self.train_iteration_count += 1
        self.network_eval.optimizer.zero_grad()
        if self.train_iteration_count % self.replace_target_network_interval == 0:
            self.network_target.load_state_dict(self.network_eval.state_dict())
        states, actions, next_states, rewards, dones = self.train_batch()
        indices = np.arange(self.memory.batch_size)
        q_eval = self.network_eval.forward(states)[indices, actions]
        q_target = self.network_target.forward(next_states).max(dim=1)[0]
        q_target[dones] = 0.0
        q_target = rewards + self.gamma * q_target
        loss = self.network_eval.loss_function(q_target, q_eval).to(self.network_eval.device)
        loss.backward()
        self.network_eval.optimizer.step()
        self.epsilon = self.epsilon - self.epsilon_decrement if self.epsilon > self.epsilon_min else self.epsilon_min

    def play(self):
        is_game_done = False
        state = self.environment.reset()
        while not is_game_done:
            action = self.action(state)
            state, _, is_game_done, _ = self.environment.step(action)
