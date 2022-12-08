import torch as torch
import torch.nn as nn
import matplotlib as mpl
import matplotlib.pyplot as plt
import math as math


class TrainLoop:
    """
    This class is a general-purpose training loop that will work for either of my Attention-based models.
    """

    def __init__(self, MyModel, data_loader, max_epochs,
                 model_type="game", loss_threshold=50, train_mode=1, optimizer="SGD"):
        # Initialize tracking variables
        self.epochs = 0
        self.max_epochs = max_epochs
        self.min_loss = 1000000
        self.prev_loss = 1000000
        self.mse_data = []  # List to hold MSE data after each epoch to plot
        self.min_loss_data = []  # List to hold min MSEs across training
        self.predicted_scores = []  # List to hold predicted game scores over training
        self.actual_scores = []  # List to hold target game scores during training

        # Set up model for training/testing
        self.model_type = model_type
        self.train_mode = train_mode
        self.model = MyModel
        self.loss_threshold = loss_threshold
        self.hidden_predictions = []
        self.hidden_targets = []

        self.data_loader = data_loader  # New
        self.batch_count = 0

        # Loading pre-trained model to test on if not in train mode
        if model_type == "game" and self.train_mode == 0:
            self.model.load_state_dict(torch.load('models/AttentionPerGame.pt'))
            self.model.eval()
        elif model_type == "season" and self.train_mode == 0:
            self.model.load_state_dict(torch.load('models/AttentionPerSeason.pt'))
            self.model.eval()

        # Assign train and target
        #self.input_data = input_data
        #self.target_data = target_data
        self.current_input = None
        self.current_target = None

        # Set up loss function and optimizer
        self.loss_function = nn.MSELoss()
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.000001)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.000001)

    def newData(self, newInput, newTarget):
        self.current_input = newInput
        self.current_target = newTarget

    def feed_model(self):
        self.optimizer.zero_grad()
        self.hidden_targets.clear()
        self.hidden_predictions.clear()
        out = self.model(self.current_input)
        if self.model_type == "game":
            self.hidden_predictions.append(out.item())
            self.hidden_targets.append(self.current_target)
        else:
            for item in out.tolist():
                self.hidden_predictions.append(item)

            for item in self.current_target:
                self.hidden_targets.append(item)

        return out

    def update_model(self, limiter=-1):
        """
        This function performs one training step, IE feeds in one set of data (either a season matrix or individual
        game vectors), computes loss and gradient, and updates the given model's parameters as necessary.
        """

        output = self.feed_model()

        #print(output)

        loss = self.loss_function(output, self.current_target)

        loss.backward()

        self.optimizer.step()

        if self.min_loss > loss.item():
            self.min_loss = loss.item()

        self.prev_loss = loss.item()

        return loss.item()

        # self.optimizer.zero_grad()
        # output = self.model.forward(self.current_input)
        #
        # loss = self.loss_function(output, self.current_target)
        #
        # self.total_loss += loss.item()
        # self.total_loss_data.append(self.total_loss)
        #
        # # Tracking the predicted and actual scores throughout training
        # if limiter == -1:
        #     self.predicted_scores.append(output.item())
        #     self.actual_scores.append(self.current_target)
        # else:
        #     if limiter % 8 == 0:
        #         for item in output.tolist():
        #             self.predicted_scores.append(item)
        #
        #         for item in self.current_target:
        #             self.actual_scores.append(item)
        #
        # loss.backward()
        #
        # self.optimizer.step()
        #
        # if self.min_loss > loss.item():
        #     self.min_loss = loss.item()
        #
        # self.prev_loss = loss.item()
        # return loss.item()

    def train_one_epoch(self):
        if self.model_type == "game":  # Training for single-game model
            for x in range(len(self.input_data)):  # Loop through all seasons
                print("Now training on season ", x + 1, "...")
                target_values = self.input_data[x][22]  # Get target values column

                mask = torch.zeros(len(self.input_data[x][22]))
                score_cleared_input = torch.clone(self.input_data[x])
                score_cleared_input[22] = mask

                newTrainData = torch.transpose(score_cleared_input, 0, 1)  # Transpose the seasons to be games x features
                newTrainData = nn.functional.normalize(newTrainData, 2, dim=0)  # Normalize the data per row/game

                # Feed model in loop
                    # Take each output and corresponding prediction and add to a list
                # After loop, calculate loss and update gradient
                # Clear hidden lists

                for i in range(len(newTrainData)):  # Loop through individual seasons
                    self.newData(newTrainData[i], target_values[i])
                    self.feed_model()  # Provide the model new data

                loss = self.update_model()

                self.mse_data.append(loss)
                self.min_loss_data.append(self.min_loss)

                if self.min_loss < self.loss_threshold:
                    return


        else:  # Training for season model
            # for x in range(len(self.input_data)):  # Loop through all seasons
            #     print("Now training on season ", x + 1, "...")
            #     target_values = self.input_data[x][22]  # Get target values column MIGHT BE DIM 2, AND MIGHT NEED TRANS
            #     mask = torch.zeros(len(self.input_data[x][22]))
            #     score_cleared_input = torch.clone(self.input_data[x])
            #     score_cleared_input[22] = mask
            #
            #     # TODO: Put mask into games train and try that, print weights to see why all output is the same
            #
            #     #print("Score cleared: ", score_cleared_input)
            #     #print("Targets: ", self.input_data[x][22])
            #     trans_input = torch.transpose(score_cleared_input, 0, 1)  # Transpose the seasons to be games x features
            #     norm_input = nn.functional.normalize(trans_input, 2, dim=0)  # Normalize the data per row/game
            #     # torch.set_printoptions(profile="full")
            #     # print(norm_input)
            #     # torch.set_printoptions(profile="default")
            #     self.newData(norm_input, target_values)
            #
            #     self.feed_model()
            #
            #     loss = self.update_model(x)
            #
            #     self.mse_data.append(loss)
            #     self.min_loss_data.append(self.min_loss)
            #
            #     if self.min_loss < self.loss_threshold:
            #         return

            for batch in self.data_loader:
                self.batch_count += 1
                # Extract the data and targets from the batch
                data, targets = batch
                # Forward pass: compute the predicted scores
                scores = self.model(data)
                # Compute the loss between the predictions and the target scores
                loss = self.loss_function(scores, targets)
                # Zero the gradients
                self.optimizer.zero_grad()
                # Backpropagate the error
                loss.backward()
                # Update the model parameters
                self.optimizer.step()
                if loss.item() < self.min_loss:
                    self.min_loss = loss.item()
                    print("New Lowest Loss!")
                    print("Scores: ", scores)
                    print("Targets: ", targets)

                if self.min_loss < 95:
                    break

    def train_multi_epoch(self):
        for x in range(self.max_epochs):
            print("\nEPOCH ", x + 1, "======================================")
            self.train_one_epoch()
            self.epochs += 1
            if self.min_loss < self.loss_threshold:
                print("Loss has reached acceptable level, ending training...")
                return

    def testPrediction(self, fakeInput, actualValues):
        output = self.model.forward(fakeInput)

        print(f"Predicted: {output.item()}")
        print(f"Actual: {actualValues.item()}")
        print(f"Error: {(output - actualValues).item()}")

    def create_plots(self):
        # MSE plot
        fig, ax = plt.subplots()
        ax.plot(self.mse_data)
        ax.set_ylabel('MSE')
        if self.model_type == "game":
            ax.set_xlabel('Number of Games Trained')
        else:
            ax.set_xlabel('Number of Seasons Trained')
        ax.set_title('MSE Loss During Training')
        plt.show()

        # Min MSE plot
        fig, ax = plt.subplots()
        ax.plot(self.min_loss_data)
        ax.set_ylabel('Minimum MSE')
        if self.model_type == "game":
            ax.set_xlabel('Number of Games Trained')
        else:
            ax.set_xlabel('Number of Seasons Trained')
        ax.set_title('Minimum MSE Loss During Training')
        plt.show()

        # Predicted Vs. Target Scores
        fig, ax = plt.subplots()
        ax.scatter(range(len(self.hidden_predictions)), self.hidden_predictions, color='blue')
        ax.scatter(range(len(self.hidden_targets)), self.hidden_targets, color='green')
        ax.set_ylabel('Score')
        ax.set_xlabel('Number of Games Trained')
        ax.set_title('Predicted Versus Actual Scores During Training')
        plt.show()





