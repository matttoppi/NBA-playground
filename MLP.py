from evalMetric import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import csv
import numpy as np
import sklearn
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt


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
        2 hidden layers with ReLi activation functions
        :param input:
        :return:
        """
        hidden = self.input_layer(input)
        hidden = F.relu(hidden)
        hidden = self.hidden_layer(hidden)
        hidden = F.relu(hidden)
        hidden = self.hidden_layer(hidden)
        hidden = F.relu(hidden)
        output = self.output_layer(hidden)
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



    # Create a TensorDataset from the training data and targets
    train_dataset = TensorDataset(train_data, train_targets)

    # Define the batch size
    batch_size = 64

    # Create a DataLoader for the training dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)


    loss_fn = torch.nn.MSELoss()
    lowest_loss = 1000000
    lowest_test_loss = 1000000
    num_epochs = 800
    batch_list = []
    loss_list = []
    epoch_list = []
    batch_count = 0

    # Iterate over the training data in batches
    for epoch in range(num_epochs):
        for batch in train_loader:
            batch_count += 1
            # extract data and targets from the batch
            data, targets = batch
            #compute the predicted scores
            scores = model(data)
            #compute the loss
            loss = loss_fn(scores, targets)
            #zero the gradients
            optimizer.zero_grad()
            # backpropagate the error
            loss.backward()
            # update model parameters
            optimizer.step()

            if batch_count % 30 == 0:
                plt.show()
            if loss.item() < lowest_loss:
                lowest_loss = loss.item()
                batch_list.append(batch_count)
                loss_list.append(lowest_loss)
                epoch_list.append(epoch)



                plt.yscale("log")
                if lowest_loss < 10000: #avoid sharp drop in loss graph
                    plt.plot(batch_list, loss_list, 'r')
                plt.show()
                torch.save(model.state_dict(), "models/MLP1.pt")
                # Print the loss
                print(f'Epoch {epoch}: Loss = {loss.item()}')
            if lowest_loss<95: #hard stop to avoid overfitting
                break



# Define the model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(56, 64, 1)  # Modify this line to output 32 channels
        self.conv2 = nn.Conv1d(32, 64, 1)
        self.fc1 = nn.Linear(64 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        """
        CNN with max pooling and ReLu
        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 1, stride=2, dilation=1)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 1, stride=2, dilation=1)
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def trainCNN(model):
    train_data, train_targets, test_data, test_targets = data_load()
    train_data = torch.unsqueeze(train_data, dim=2)
    train_targets = torch.unsqueeze(train_targets, dim=1) #fix dimensionality
    train_dataset = TensorDataset(train_data, train_targets)


    # Define DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)


    # Define the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    loss_fn = nn.MSELoss()


    lowest_loss = 10000 #arbitrary lowest loss
    num_epochs = 500
    # Train the model for a number of epochs
    for epoch in range(num_epochs):
        # Iterate over the mini-batches of data
        count=0
        for x_batch, y_batch in train_loader:
            #print(x_batch.size())
            if x_batch.size()[0] != 32:
                #print("broken on: ", count)
                break
            #print(y_batch.size())
            # Forward pass: compute predicted y by passing x to the model
            count+=1
            y_pred = model(x_batch)


            # Compute and print loss
            loss = loss_fn(y_pred, y_batch)
            if loss.item() < lowest_loss:
                lowest_loss = loss.item()
                print(f"Epoch {epoch}: Loss = {loss.item()}")

            # Zero gradients, perform a backward pass, and update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


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
    # Convert data into PyTorch tensors
    train_data, train_targets = torch.from_numpy(train_data), torch.from_numpy(train_targets)
    test_data, test_targets = torch.from_numpy(test_data), torch.from_numpy(test_targets)

    return train_data, train_targets, test_data, test_targets


def test_model():
    model = MLP(input_size=56, hidden_size=128, output_size=1)
    model.load_state_dict(torch.load('models/MLP1.pt'))
    model.eval()
    train_data, train_targets, test_data, test_targets = data_load()

    with torch.no_grad():
        pred = model(test_data)
        for i in range(len(test_data)):
            print("pred: ", pred[i].item(), "act: ", test_targets[i].item())

        loss_fn = torch.nn.MSELoss()
        print("final: ", loss_fn(pred, test_targets))