import gym
import numpy as np


class ActionTransform(gym.Wrapper):
    def __init__(self, env=None, repeat=4):
        super(ActionTransform, self).__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        self.observation_buffer = np.zeros_like((2, self.shape))

    def step(self, action):
        done = False
        total_reward = 0.0
        for i in range(self.repeat):
            observation, reward, done, info = self.env.step(action)
            total_reward += reward
            self.observation_buffer[i % 2] = observation
            if done:
                break
        observation = np.maximum(self.observation_buffer[0], self.observation_buffer[1])
        return observation, total_reward, done, info

    def reset(self):
        state = self.env.reset()
        self.observation_buffer = np.zeros_like((2, self.shape))
        self.observation_buffer[0] = state
        return state
