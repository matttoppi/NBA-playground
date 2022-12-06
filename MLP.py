from evalMetric import *
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import numpy as np


class MLP(nn.Module):
    """
    Multi-layer perceptron model.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.loss_fn = nn.MSELoss()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        """

        :param input:
        :return:
        """
        hidden = self.input_layer(input)
        hidden = self.hidden_layer(hidden)
        output = self.output_layer(hidden)
        return output


def trainMLP(model):
    """
    :param model:
    :return:
    ""
    Training MLP model.

    :param model:
    :return:
    """
    m_l_p = MLP(input_size=10, hidden_size=32, output_size=1)

    # Define the loss function
    train_data = data_load()

    # Define the optimizer
    optimizer = torch.optim.Adam(m_l_p.parameters(), lr=0.01)





def data_load():
    # Read the CSV file into a NumPy array
    with open('preVectorDATA/ALL_GAME_DATA.csv', 'r') as f:
        reader = csv.reader(f)
        data = np.array([row for row in reader], dtype=np.float32)

    # Convert the NumPy array into a PyTorch tensor
    return torch.from_numpy(data)


# Create an MLP with 22 input nodes, 64 hidden nodes, and 1 output node
mlp = MLP(input_size=22, hidden_size=64, output_size=1)
