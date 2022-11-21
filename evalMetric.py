"""File that contains the evaluation metrics for our model."""


def scoreDiffSingleGame(predicted, actual):
    """
    Simple evaluation metric that takes in the predicted and actual scores for a single game and returns the difference
    between the two. This value will always be a positive number
    :param predicted: The score for one team for one game that was predicted by our model
    :param actual: The actual score of the game for the team whose score we predicted
    """
    return abs(predicted - actual)


def scoreDiffMultiGame(scores):
    """
    This method creates a list of values containing the differences between predicted and actual scores for a given list
    of scores.
    :param scores: The list containing tuples of values representing (predictedScores, actualScores)
    """
    diffs = []
    for tup in scores:
        diffs.append(abs(tup[0] - tup[1]))

    # Sanity check to make sure the size of both lists are the same afterwards
    if len(diffs) != len(scores):
        print("ERROR: Length of predicted/actual scores list is different from output list.")
        print(f"Pred/Actual List length: {len(scores)}; Difference List length: {len(diffs)}")

    return diffs
