import torch as torch
import torch.nn as nn
import matplotlib as mpl
import matplotlib.pyplot as plt
import math as math


class TrainLoop:
    """
    This class is a general-purpose training loop that will work for either of my Attention-based models.
    """

    def __init__(self, MyModel, input_data, target_data, max_epochs, model_type="game", train_mode=1, optimizer="SGD"):
        # Initialize tracking variables
        self.epochs = 0
        self.max_epochs = max_epochs
        self.min_loss = 1000000
        self.prev_loss = 1000000
        self.mse_data = []  # List to hold MSE data after each epoch to plot
        self.min_loss_data = []  # List to hold min MSEs across training
        self.single_games_count = []  # List to hold # of games seen for single-game model
        # Set up model for training/testing
        self.model_type = model_type
        self.train_mode = train_mode
        self.model = MyModel
        # Loading pre-trained model to test on if not in train mode
        if model_type == "game" and self.train_mode == 0:
            self.model.load_state_dict(torch.load('models/AttentionPerGame.pt'))
            self.model.eval()
        elif model_type == "season" and self.train_mode == 0:
            self.model.load_state_dict(torch.load('models/AttentionPerSeason.pt'))
            self.model.eval()
        # Assign train and target
        self.input_data = input_data
        self.target_data = target_data
        # Set up loss function and optimizer
        self.loss_function = nn.MSELoss()
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001)

    def newData(self, newTrain, newTarget):
        self.input_data = newTrain
        self.target_data = newTarget

    def trainOnce(self):
        """
        This function performs one training step, IE feeds in one set of data (either a season matrix or individual
        game vectors), computes loss and gradient, and updates the given model's parameters as necessary.
        """

        self.optimizer.zero_grad()
        ouput = self.model.forward(self.input_data)

        loss = self.loss_function(ouput, self.target_data)

        loss.backward()

        self.optimizer.step()

        if abs(loss.item() - self.prev_loss) < 0.01:  # If current loss - prev_loss (change in loss) is less than 0.01
            return -1, self.min_loss

        if self.min_loss > loss.item():
            self.min_loss = loss.item()

        self.prev_loss = loss.item()
        return loss.item(), self.min_loss

    def train_one_epoch(self):
        if self.model_type == "game":  # Training for single-game model
            game_count = 0
            for x in range(len(self.input_data)):
                print("Input Data size: ", self.input_data.size())
                target_values = self.input_data[x][22]
                print("Target values size: ", target_values.size())
                newTrainData = torch.transpose(self.input_data[x], 0, 1)
                newTrainData = nn.functional.normalize(newTrainData, 2, dim=0)
                # print(f"Size of new data 2: {newTrainData.size()}")

                for i in range(len(newTrainData)):
                    game_count += 1
                    trainRow = newTrainData[i]
                    targetRow = target_values[i]
                    self.newData(trainRow, targetRow)
                    loss, min_loss = self.trainOnce()
                    self.mse_data.append(loss)
                    if game_count % 100 == 0:
                        print(f"MSE Loss for step {game_count}: {loss}")
                        print(f"Min Loss as of step {game_count}: {min_loss}")
                        self.min_loss_data.append(min_loss)
                        self.single_games_count.append(game_count)

                    if loss == -1:
                        print(f"Delta Loss is below threshold, stopping training...")
                        print(f"Min Loss as of step {game_count}: {min_loss}")
                        self.min_loss_data.append(min_loss)
                        self.single_games_count.append(game_count)
                        return
        else:  # Training for season model
            return

    def train_multi_epoch(self):
        for x in range(self.max_epochs):
            self.train_one_epoch()

    def testPrediction(self, fakeInput, actualValues):
        output = self.model.forward(fakeInput)

        print(f"Predicted: {output.item()}")
        print(f"Actual: {actualValues.item()}")
        print(f"Error: {(output - actualValues).item()}")

    def create_plots(self):
        # MSE plot
        fig, ax = plt.subplots()
        ax.plot(self.mse_data)
        plt.show()
        # Min MSE plot

        # Loss plot




