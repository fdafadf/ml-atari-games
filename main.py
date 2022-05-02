import os
import datetime
from QFactory import QFactory
from QLearning import QLearning

working_directory_path = 'C:\\Users\\pawel\\PycharmProjects\\atari_models'


def train():
    session_directory_name = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    session_directory_path = os.path.join(working_directory_path, session_directory_name)
    agent = QFactory.create_pong_agent(render_mode='rgb_array', learning_rate=0.0001)
    os.mkdir(session_directory_path)
    q_learning = QLearning(agent)
    q_learning.train(session_directory_path)


def play():
    model_path = os.path.join(working_directory_path, '20220415_2046\\best_eval')
    agent = QFactory.create_pong_agent(render_mode='human', model_path=model_path)
    agent.epsilon = 0
    agent.play()


play()
# train()
