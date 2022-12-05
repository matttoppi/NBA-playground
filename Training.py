import random

from evalMetric import *
import numpy as np
import pandas as pd
#import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, MaxPooling1D, Flatten


from sklearn.preprocessing import MinMaxScaler




def evaluate(predictions, actual):
    """
    This method evaluates the model by calculating the average error
    :param predictions: The predictions made by the model
    :param actual: The actual scores
    :return:
    """
    # Calculate the average error
    return np.average(np.abs(predictions - actual))


def tobey_train(train, test):
    '''
    Trains the model on the training data and returns accuracy against test data
    :param train: training data
    :param test: testing data
    :return:
    '''
    print("-----------------------------\n"
          "Tobey's Model:\n")
    total_error = 0
    for season in test:
        for i in season[21]:
            # for each game in the test set
            # Predicts a random score for each team
            predicted = 100  # 100 for every score


            # turn the tensor into an int
            i = int(i)

            # print("actual: ", actual, "predicted: ", predicted, "\n")

            # Evaluates the prediction against the real score
            dif = scoreDiffSingleGame(predicted, i)  # get the differece between the predicted score and the actual
            # score
            total_error += dif  # add the difference to the total difference

    error = total_error / len(test[0][21])

    print("Tobey's Overall Error: ", total_error)
    print("Tobey's Avg Error:", error)
    print("-----------------------------\n")


def matt_train(train, test):
    """
    This function trains the model on the training data and returns the accuracy on the test data. Predicsts
    a random score for each team and evaluates the prediction against the real score.
    :param train: tensor of training data
    :param test: tensor of testing data
    :return:
    """

    print("-----------------------------\n"
          "Matt's Model:\n")

    total_diff = 0

    for season in test: # for each season in the test set
        for actual in season[21]: # for each game in the test set
            # for each game in the test set
            # Predicts a random score for each team
            predicted = random.randint(95, 110)  # random score between 95 and 110

            # get the score from that game
            # turn the tensor into an int
            actual = int(actual)

            # print("actual: ", actual, "predicted: ", predicted, "\n")

            # Evaluates the prediction against the real score
            dif = scoreDiffSingleGame(predicted, actual)  # get the differece between the predicted score and the actual
            # score
            total_diff += dif  # add the difference to the total difference

    error = total_diff / len(test[0][21])

    print("Matt's Overall Error: ", total_diff)
    print("Matt Avg Error:", error)
    print("-----------------------------\n")




def matts_LSTM_train(train, test):
    """
    LSTM model that is trained on the data
    author: Matt
    :return:
    """
    print("-----------------------------\n"
          "Matt's LSTM Model:\n")

    learning_rate = 0.01
    epochs = 100
    batch_size = 100
    num_units = 100

    X = torch.FloatTensor(train[0][0]) # input
    y = torch.FloatTensor(train[0][21]) # 21 is the index of the score column

    X = X.view(-1, 1, 1) # reshape the data to fit the LSTM model
    y = y.view(-1, 1, 1) # This is the main function that runs the program

    lstm_cell = nn.LSTMCell(1, num_units) # create the LSTM cell
    output, (hn, cn) = lstm_cell(X, (torch.zeros(batch_size, num_units), torch.zeros(batch_size, num_units))) # run the LSTM cell on the input data

    loss_fn = nn.CrossEntropyLoss() # create the loss function
    optimizer = optim.Adam(lstm_cell.parameters(), lr=learning_rate) # create the optimizer

    for epoch in range(epochs): # for each epoch
        optimizer.zero_grad()
        output, (hn, cn) = lstm_cell(X, (torch.zeros(batch_size, num_units), torch.zeros(batch_size, num_units))) # run the LSTM cell on the input data

        loss = loss_fn(output, y) # calculate the loss
        loss.backward() # backpropagate the loss
        optimizer.step() # update the weights
        print(f'Epoch: {epoch}, Loss: {loss.item()}')


    #evaluate the model
    with torch.no_grad(): # turn off gradient tracking
        logits = lstm_cell(X, (torch.zeros(batch_size, num_units), torch.zeros(batch_size, num_units))) # run the LSTM cell on the input data
        predictions = torch.argmax(logits, dim=1) # get the predictions
        accuracy = (predictions == y).float().mean() # calculate the accuracy
        print(f'Accuracy: {accuracy}') # print the accuracy


def matts_cnn_model(train_data, train_labels):
    """
    CNN model that is trained on the data
    Author: Matt
    :param train_data:
    :param train_labels:
    :return:
    """
    print("-----------------------------\n"
          "Matt's LSTM Model:\n")
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data = scaler.fit_transform(train_data)







def lucas_train(train, test):
    """
    This function "trains" my "model" using the train and test datasets passed in as parameters
    :param train: The game data that is being trained on
    :param test: The game data that we are predicting against
    :return:
    """
    testDiff = 0  # Total difference
    for season in test:  # Loops through each season, so could print out error for each season individually
        for gameScore in season[21]:
            prediction = lucas_model()

            actual = gameScore

            difference = scoreDiffSingleGame(prediction, actual)

            testDiff += difference
    avgError = testDiff / len(test[0][21])
    print("-----------------------------")
    print("Lucas's Model:\n")
    print(f'Overall Error: {testDiff}')
    print(f'Average Error: {avgError}')
    print("-----------------------------\n")


def lucas_model():
    """
    My model that simply returns a random number between 90 and 150 as a prediction
    :return: A random integer value between 90 and 150
    """
    return random.randint(90, 150)
