import gym
import numpy as np
import collections


class ObservationHistory(gym.ObservationWrapper):
    def __init__(self, env, repeat=4):
        super(ObservationHistory, self).__init__(env)
        self.observation_space = gym.spaces.Box(env.observation_space.low.repeat(repeat, axis=0), env.observation_space.high.repeat(repeat, axis=0), dtype=np.float32)
        self.items = collections.deque(maxlen=repeat)

    def reset(self):
        self.items.clear()
        observation = self.env.reset()
        for _ in range(self.items.maxlen):
            self.items.append(observation)
        return np.array(self.items).reshape(self.observation_space.low.shape)

    def observation(self, observation):
        self.items.append(observation)
        return np.array(self.items).reshape(self.observation_space.low.shape)
