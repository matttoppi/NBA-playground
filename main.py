# https://github.com/swar/nba_api
# https://github.com/swar/nba_api/tree/master/docs/examples

from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams

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




def lucas():
    gamefact = GameDataFeatureFactory()

    gamefact.split_by_season()

    list = gamefact.create_features()

    train, test = gamefact.split_train_test(list)
    #for season in train:
        #print(len(season[0]))

    # Lucas Test Code ===================================================================
    traintrans = torch.transpose(train[0], 0, 1)
    traintrans = nn.functional.normalize(traintrans, 2, dim=0)
    target_values = train[0][22]

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
    
    # New Attention -------------------------------------------------
    newInFeats = 24
    newOutFeats = 128
    lucas_new_model = LucasNewModel(newInFeats, newOutFeats)

    newTrainData = train[0]
    print(f'Len of new Data {len(newTrainData)}')

    newTrainLoop = TrainLoopNew(lucas_new_model, newTrainData, target_values)

    for i in range(len(newTrainData)):
        trainRow = newTrainData[i]
        targetRow = target_values[i]
        newTrainLoop.newData(trainRow, targetRow)
        newTrainLoop.trainOnce()

    newTrainLoop.testPrediction(newTrainData[0], target_values[0])

    # End New Attention ---------------------------------------------

    # print(prediction)
    # print(prediction.size())

    # print(f'Output Vals: {outputVals}')
    # print(f'Output softmax values: {outputWeights}')

    # torch.set_printoptions(profile="full")
    # print(f'First output val')
    # torch.set_printoptions(profile="default")

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
