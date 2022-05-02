import numpy as np
import torch as torch
from torch import nn as torch_nn, optim as torch_optim
from torch.nn import functional as torch_fn


class QNetwork(torch_nn.Module):
    def __init__(self, learning_rate, observation_space_shape, action_space_size):
        super(QNetwork, self).__init__()
        self.layer_1 = torch_nn.Conv2d(observation_space_shape[0], 32, 8, stride=4)
        self.layer_2 = torch_nn.Conv2d(32, 64, 4, stride=2)
        self.layer_3 = torch_nn.Conv2d(64, 64, 3, stride=1)
        state = torch.zeros(1, *observation_space_shape)
        conv_output_shape = self.layer_1(state)
        conv_output_shape = self.layer_2(conv_output_shape)
        conv_output_shape = self.layer_3(conv_output_shape)
        linear_input_size = int(np.prod(conv_output_shape.size()))
        self.layer_4 = torch_nn.Linear(linear_input_size, 512)
        self.layer_5 = torch_nn.Linear(512, action_space_size)
        self.optimizer = torch_optim.RMSprop(self.parameters(), lr=learning_rate)
        self.loss_function = torch_nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        output_1 = torch_fn.relu(self.layer_1(state))
        output_2 = torch_fn.relu(self.layer_2(output_1))
        output_3 = torch_fn.relu(self.layer_3(output_2))
        output_3 = output_3.view(output_3.size()[0], -1)
        output_4 = torch_fn.relu(self.layer_4(output_3))
        output_5 = self.layer_5(output_4)
        return output_5

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
