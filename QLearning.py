import os
import numpy as np
import datetime

from QAgent import QAgent


class QLearning:
    def __init__(self, agent: QAgent):
        self.agent = agent
        self.numer_of_episodes = 3000

    def train(self, models_directory_path):
        history = []
        best_score = -np.inf
        for i in range(self.numer_of_episodes):
            episode_score = 0
            is_game_done = False
            state = self.agent.environment.reset()
            while not is_game_done:
                state, reward, is_game_done, info = self.agent.step(state)
                self.agent.train()
                episode_score += reward
            history.append(episode_score)
            avg_score = np.mean(history[-100:])
            if i % 300 == 0:
                self.agent.network_target.save(os.path.join(models_directory_path, f"episode_{i}_target"))
                self.agent.network_eval.save(os.path.join(models_directory_path, f"episode_{i}_eval"))
            if avg_score > best_score:
                self.agent.network_target.save(os.path.join(models_directory_path, f"best_target"))
                self.agent.network_eval.save(os.path.join(models_directory_path, f"best_eval"))
                best_score = avg_score
            print(f"{datetime.datetime.now().strftime('%H:%M:%S')} {i:4} {avg_score:.2f}")

