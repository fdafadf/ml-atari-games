import datetime
import os

import gym

from ActionTransform import ActionTransform
from ObservationHistory import ObservationHistory
from ObservationTransform import ObservationTransform
from QAgent import QAgent
from QLearning import QLearning


class QFactory:
    @staticmethod
    def create_pong_agent(render_mode='rgb_array', model_path=None, learning_rate=0.001):
        environment = QFactory.create_pong_environment(render_mode=render_mode)
        agent = QAgent(environment, gamma=0.99, learning_rate=learning_rate)
        if model_path:
            agent.network_eval.load(model_path)
            agent.network_eval.eval()
        return agent

    @staticmethod
    def create_pong_environment(render_mode='rgb_array'):
        environment = gym.make('PongNoFrameskip-v4', render_mode=render_mode)
        environment = ActionTransform(environment)
        environment = ObservationTransform((1, 84, 84), environment)
        environment = ObservationHistory(environment)
        return environment
