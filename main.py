# https://github.com/swar/nba_api
# https://github.com/swar/nba_api/tree/master/docs/examples

#from nba_api.stats.static import teams
#from nba_api.stats.endpoints import leaguegamefinder
#from nba_api.stats.static import teams

#from call_library import *
from FeatureVectorFactory import *
from IDManager import *
import pandas as pd
import numpy as np
import torch
from Training import *
from AttentionModel import *
from SingleGameAttention import *
from matts_training import *
from AttentionTrain import *


#def train_by_game(train, test):


    # Training loop, runs through the whole training dataset
    # count = 0
    # epochs = 5
    # for y in range(epochs):
    #
    #     for x in range(len(train)):
    #         target_values = train[x][22]
    #         newTrainData = torch.transpose(train[x], 0, 1)
    #         newTrainData = nn.functional.normalize(newTrainData, 2, dim=0)
    #         # print(f"Size of new data 2: {newTrainData.size()}")
    #
    #         for i in range(len(newTrainData)):
    #             count += 1
    #             trainRow = newTrainData[i]
    #             targetRow = target_values[i]
    #             newTrainLoop.newData(trainRow, targetRow)
    #             loss, min_loss = newTrainLoop.trainOnce()
    #             if count % 500 == 0:
    #                 print(f"MSE Loss for step {count}: {loss}")
    #                 print(f"Min Loss as of step {count}: {min_loss}")
    #
    #             if loss == -1:
    #                 print(f"Delta Loss is below threshold, stopping training...")
    #                 print(f"Min Loss as of step {count}: {min_loss}")
    #
    #                 testrows = torch.transpose(test[0], 0, 1)
    #                 testrows = nn.functional.normalize(testrows, 2, dim=0)
    #                 testtargets = test[0][22]
    #                 for p in range(10):
    #                     newTrainLoop.testPrediction(testrows[p], testtargets[p])
    #
    #                 return
    #
    # testrows = torch.transpose(test[0], 0, 1)
    # testrows = nn.functional.normalize(testrows, 2, dim=0)
    # testtargets = test[0][22]
    # for i in range(10):
    #     newTrainLoop.testPrediction(testrows[i], testtargets[i])



def lucas():
    gamefact = GameDataFeatureFactory()

    gamefact.split_by_season()

    list = gamefact.create_features()

    train, test = gamefact.split_train_test(list)

    newInFeats = 24
    newOutFeats = 128
    max_epochs = 10
    target_values = train[0][22]
    lucas_new_model = LucasNewModel(newInFeats, newOutFeats)

    newTrainLoop = TrainLoop(lucas_new_model, train[0], target_values, max_epochs)

    newTrainLoop.train_multi_epoch()
    newTrainLoop.create_plots()

    #for season in train:
        #print(len(season[0]))

    # Lucas Test Code ===================================================================

    #traintrans = torch.transpose(train[0], 0, 1)
    #traintrans = nn.functional.normalize(traintrans, 2, dim=0)
    #target_values = train[0][22]

    #inFeats = len(traintrans[0])
    #outFeats = 128

    # print(f'inFeats: {inFeats}')
    # print(f'traintrans: {traintrans}')
    # print(train[0][0])

    #lucas_model = LucasModel(inFeats, outFeats)

    # prediction = lucas_model.forward(traintrans)

    #trainLoop = TrainLoop(lucas_model, traintrans, target_values)

    #for i in range(1000):
    #    trainLoop.trainOnce()

    #trainLoop.testPrediction(traintrans, target_values)


    # End Lucas Test Code ===============================================================



def matt():
    # getDataMatt()
    load_features()
    print("Matt")



def tobey():
    print("Tobey")





def main():
    print("code started")
    #matt()
    lucas()
    # tobey()


# main()
if __name__ == '__main__':
    main()
