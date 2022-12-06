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
        #print("hidden1: ", hidden.size())
        hidden = self.hidden_layer(hidden)
        #print("hidden2: ", hidden.size())
        output = self.output_layer(hidden)
        #print("output: ", output.size())
        return output


def trainMLP(model):
    """
    Training MLP model.

    :param model: instance of MLP class
    :return:
    """


    # Define the loss function
    train_data = data_load()
    print(train_data[0].size())
    print(train_data[1].size())

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # Iterate over the training data in batches

    # input1 is the input data. each input1 is a tensor of size 57 representing a game
    # train_data[0] the entire data set including the target (21848x57)
    # train_data[1] the target data (21848x1)

    loss_fn = torch.nn.MSELoss()
    i = 0 #iteration counter
    for input1 in train_data[0]:
        i+=1
        # Make predictions using the MLP model
        output = model(train_data[0])
        # Compute the loss between the predictions and the target values

        loss = loss_fn(output, train_data[1][i])
        print(i, " loss: ", loss.item())
        # Zero the gradients
        optimizer.zero_grad()

        # Backpropagate the error
        loss.backward()

        # Update the model parameters
        optimizer.step()
        if(i>1000):
            break


def data_load():
    # Read the CSV file into a NumPy array
    with open('preVectorDATA/ALL_GAME_DATA.csv', 'r') as f:
        reader = csv.reader(f)
        data = np.array([[col for j, col in enumerate(row) if j > 0] for i, row in enumerate(reader) if i > 0],
                        dtype=np.float32)


    # Convert the NumPy array into PyTorch tensors
    return torch.from_numpy(data), torch.from_numpy(data[:, 23])
