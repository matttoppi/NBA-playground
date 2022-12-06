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



    train_data, train_targets, test_data, test_targets = data_load()
    print(train_data.size())
    print(test_data.size())
    print(train_targets.size())
    print(test_targets.size())

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.006)
    # Iterate over the training data in batches

    loss_fn = torch.nn.MSELoss()
    lowest_loss = 1000000
    lowest_test_loss = 1000000
    on_epoch = 0
    num_epochs = 1000
    batch_size = 100
    batch_count = 0
    for epoch in range(num_epochs):
        output = model(train_data)
        # Compute the loss between the predictions and the target values
        loss = loss_fn(output, train_targets)
        # Zero the gradients
        optimizer.zero_grad()
        # Backpropagate the error
        loss.backward()
        # Update the model parameters
        optimizer.step()
        if loss.item() < lowest_loss:
            lowest_loss = loss.item()
            on_epoch = epoch
            print("new low: ", lowest_loss, " on epoch ", epoch)
            torch.save(model.state_dict(), "models/MLP.pt")

        test_loss = loss_fn(model(train_data),train_targets)
        print("epoch: ", epoch, " test loss: ", test_loss.item(), "lowest test loss: ", lowest_test_loss)
        if test_loss.item() < lowest_test_loss:
            lowest_test_loss = test_loss.item()
            on_epoch = epoch
            torch.save(model.state_dict(), "models/MLP2.pt")
            print("epoch: ", epoch, " loss: ", loss.item(), "lowest loss: ", lowest_loss, " on epoch: ", on_epoch)


def data_load():
    # Read the CSV file into a NumPy array
    with open('preVectorDATA/ALL_GAME_DATA.csv', 'r') as f:
        reader = csv.reader(f)
        data = np.array([[col for j, col in enumerate(row) if j > 0] for i, row in enumerate(reader) if i > 0],
                        dtype=np.float32)

    # Split the data into training and testing sets
    train_data, train_targets = data[:16880, :], data[:16880, 23]
    test_data, test_targets = data[16878:21500, :], data[16878:21500, 23]
    train_data = np.delete(train_data,23,1)
    test_data = np.delete(test_data,23,1)
    # Convert the data into PyTorch tensors
    train_data, train_targets = torch.from_numpy(train_data), torch.from_numpy(train_targets)
    test_data, test_targets = torch.from_numpy(test_data), torch.from_numpy(test_targets)

    return train_data, train_targets, test_data, test_targets


