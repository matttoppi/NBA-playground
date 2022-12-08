import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
#from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.autograd import Variable

input_file = "preVectorDATA/ALL_GAME_DATA.csv"


# feed forward neural network
def feedForward():
    print("Feed Forward Neural Network")
    train_input, train_target, test_input, test_target = data_load()

    # Define the model
    model = torch.nn.Sequential(
        torch.nn.Linear(56, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1),
    )

    # Define the loss function
    loss_fn = torch.nn.MSELoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000005)

    # Train the model
    loss = 0
    lowest_loss = 1000000000000
    loss_count = 0
    epochs = 100000
    batch_size = 2460
    # time each epoch
    import time

    plt.title('Loss vs Epoch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')

    epoch_list = []
    batch_list = []
    loss_list = []

    batch_data = []
    batch_target = []

    batch_count = 0
    averagetime = 0
    epoch_count = 0
    # Iterate over the training data in batches
    for epoch in range(epochs):
        epoch_count += 1
        i = 0
        s = 0

        start_time = time.time()
        for row in train_input:

            # Select the next batch of training data as they are read in
            # batch_data.append(row)
            # batch_target.append(train_target[i])
            #
            #
            #
            # # Forward pass: Compute predicted y by passing x to the model
            # # y_pred = model(train_input)
            # # y_pred = model(torch.tensor(batch_data))
            #
            # # Compute and print loss
            # # loss = loss_fn(y_pred, torch.tensor(batch_target))
            #
            # y_pred = model(row)
            #
            #
            #
            #
            # # Compute and print loss
            # loss = loss_fn(y_pred, train_target[i])
            #
            i += 1
            # # Zero gradients, perform a backward pass, and update the weights.
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            if i % batch_size == 0:
                batch_count += 1
                loss = loss_fn(model(train_input[i - batch_size:i]), train_target[i - batch_size:i])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if loss.item() < lowest_loss:
                    lowest_loss = loss.item()
                    loss_list.append(lowest_loss)
                    # put the epoch to the list
                    epoch_list.append(epoch_count)
                    batch_list.append(batch_count)
                    plt.plot(batch_list, loss_list, 'r')

                    torch.save(model.state_dict(), "models/feedForwardModel.pt")

                    print("new lowest loss at batch ", batch_count, " : |MSE: ", lowest_loss, "|root error: ",
                          math.sqrt(
                              lowest_loss))

                    # end of game in data loop

        y_pred = model(test_input)
        loss = loss_fn(y_pred, test_target)
        if loss.item() < lowest_loss:
            lowest_loss = loss.item()
            loss_list.append(lowest_loss)
            # put the epoch to the list
            epoch_list.append(epoch_count)
            batch_list.append(batch_count)
            plt.plot(batch_list, loss_list, 'r')

            torch.save(model.state_dict(), "models/feedForwardModel.pt")

            # plt.savefig('lossGraph.png')

        # end_time = time.time()
        # put the lowest loss to the list

        print(epoch_count, " : ", lowest_loss)

        # calculate loss
        loss = loss_fn(model(train_input), train_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # calculate loss against the test data
        y_pred = model(test_input)
        loss = loss_fn(y_pred, test_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # get the average time for each epoch
        # averagetime += (end_time - start_time)

        # print("\nThis epoch loss: ", loss.item(), "took: ", end_time - start_time, "seconds")
        # # estimate the time to complete the training using the average time per epoch
        # print("Estimated time to complete: ", (200 - epoch) * averagetime / 60, " minutes")

        # # print("Lowest loss: ", lowest_loss, "\n")
        # print("Lowest loss: ", lowest_loss, "\n")
        if epoch_count % 15 == 0:
            # make the y axis scale logarithmic
            plt.yscale('log')
            plt.show()

    plt.plot(epoch_list, loss_list)
    plt.savefig('lossGraph.png')
    plt.show()

    # # Save the model checkpoint
    # torch.save(model.state_dict(), 'model.ckpt') # this is saving the model to a file so it can be used later


def test_the_model():
    # Define the model
    model = torch.nn.Sequential(
        torch.nn.Linear(56, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1),
    )

    # Load the model checkpoint
    model.load_state_dict(torch.load('models/feedForwardModel.pt'))

    # Set the model to evaluation mode
    model.eval()

    # Read the CSV file into a NumPy array
    with open('preVectorDATA/ALL_GAME_DATA.csv', 'r') as f:
        reader = csv.reader(f)
        data = np.array([[col for j, col in enumerate(row) if j > 0] for i, row in enumerate(reader) if i > 0],
                        dtype=np.float32)

    # Split the data into training and test sets
    train_input = data[:18000, :]
    train_target = data[:, 23]
    test_input = data[18000:, :]
    test_target = data[:, 23]

    # Convert the NumPy arrays into PyTorch tensors
    train_input = torch.from_numpy(train_input)
    train_target = torch.from_numpy(train_target)

    test_input = torch.from_numpy(test_input)
    test_target = torch.from_numpy(test_target)

    # predict the target feature for the test data
    with torch.no_grad():
        test_pred = model(test_input)

        print(test_pred[0:10])
        print(test_target[0:10])

        # Compute the mean squared error of the predictions
        test_loss = torch.nn.functional.mse_loss(test_pred, test_target)
        print('Test loss:', test_loss.item())


def data_load():
    # Read the CSV file into a NumPy array
    with open('preVectorDATA/ALL_GAME_DATA.csv', 'r') as f:
        reader = csv.reader(f)
        data = np.array([[col for j, col in enumerate(row) if j > 0] for i, row in enumerate(reader) if i > 0],
                        dtype=np.float32)

    # use view to reshape the data
    # data = data.view(-1, 23)  # this is reshaping the data to be 1D

    # Split the data into training and test sets
    train_input = data[:17220, :]  # the first 17220 rows are the training data
    train_target = data[:17220, 23]  # the first 17220 rows are the training data
    test_input = data[17220:21500, :]  # the last 4280 rows are the test data
    test_target = data[17220:21500, 23]

    # remove the target feature from the training data
    train_input = np.delete(train_input, 23, 1)

    # remove the target feature from the test data
    test_input = np.delete(test_input, 23, 1)

    # print the shape of th e data
    print("train_input shape: ", train_input.shape)
    print("train_target shape: ", train_target.shape)
    print("test_input shape: ", test_input.shape)
    print("test_target shape: ", test_target.shape)

    # Convert the NumPy arrays into PyTorch tensors
    return torch.from_numpy(train_input), torch.from_numpy(train_target), torch.from_numpy(
        test_input), torch.from_numpy(test_target)


class NbaLstm(torch.nn.Module):
    """
    This class is the LSTM model for the NBA data
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(NbaLstm, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        This function is the forward pass of the LSTM model
        :param x:  the input data to the model in the form of a tensor
        :return:  the output of the model
        """
        # Set initial states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to("cpu")
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to("cpu")

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


def train_the_model_lstm():


    # create a new plot to show the loss over time
    plt.figure()
    plt.title("Loss over time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    epoch_list = []
    loss_list = []
    epoch_count = 0

    # Hyper-parameters
    input_size = 56
    hidden_size = 2
    num_layers = 1
    num_classes = 1
    num_epochs = 20_000
    batch_size = 512
    learning_rate = 0.001

    lowest_loss = 10000000

    # Load Data
    train_input, train_target, test_input, test_target = data_load()

    # Train the model
    model = NbaLstm(input_size, hidden_size, num_layers, num_classes)

    # Loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_input)
    for epoch in range(num_epochs):
        epoch_count += 1

        for i in range(0, total_step, batch_size):
            # Get the inputs; data is a list of [inputs, labels]
            inputs = train_input[i:i + batch_size]
            targets = train_target[i:i + batch_size]

            # get the needed shape for the LSTM
            inputs = inputs.view(-1, 1, 56)

            # ensure it is a tensor of type float
            inputs = inputs.type(torch.FloatTensor)
            targets = targets.type(torch.FloatTensor)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if the loss is lower than the lowest loss, save the model


        # print the loss every 100 epochs
        if (epoch + 1) % 2 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, loss.item()))

        if (epoch + 1) % 100 == 0:
            # plot the loss over time
            if loss < lowest_loss:
                lowest_loss = loss
                torch.save(model.state_dict(), 'models/lstm_model.pt')
                epoch_list.append(epoch_count)
                loss_list.append(loss.item())
                plt.plot(epoch_list, loss_list)
                plt.show()
