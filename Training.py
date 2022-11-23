import random

from evalMetric import *


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
            predicted = random.randint(95, 110)  # random score between 75 and 125

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


def lucas_train(train, test):
    """
    This function "trains" my "model" using the train and test datasets passed in as parameters
    :param train: The game data that is being trained on
    :param test: The game data that we are predicting against
    :return:
    """
    testDiff = 0  # Total difference 
    for season in test:
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
