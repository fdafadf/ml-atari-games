import cv2
import gym
import numpy as np


class ObservationTransform(gym.ObservationWrapper):
    def __init__(self, shape, env=None):
        super(ObservationTransform, self).__init__(env)
        # self.shape = (shape[2], shape[0], shape[1])
        self.shape = shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32)

    def observation(self, observation):
        grayscale = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        grayscale_resized = cv2.resize(grayscale, self.shape[1:], interpolation=cv2.INTER_AREA)
        grayscale_resized_reshaped = np.array(grayscale_resized, dtype=np.uint8).reshape(self.shape)
        grayscale_resized_reshaped_rescaled = grayscale_resized_reshaped / 255.0
        return grayscale_resized_reshaped_rescaled
