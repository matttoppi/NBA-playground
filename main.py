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
from AttentionTrain import *

from MLP import *




def lucas():
    gamefact = GameDataFeatureFactory()

    gamefact.split_by_season()

    list = gamefact.create_features()

    train, test = gamefact.split_train_test(list)

    newInFeats = 24
    newOutFeats = 128
    max_epochs_games = 10
    max_epochs_seasons = 50
    loss_threshold = 10
    target_values = train[0][22]
    lucas_new_model = LucasNewModel(newInFeats, newOutFeats)
    season_model = LucasModel(newInFeats, newOutFeats)

    newTrainLoop = TrainLoop(season_model, train, target_values, max_epochs_seasons,
                            model_type="season", loss_threshold=loss_threshold)
    #newTrainLoop = TrainLoop(lucas_new_model, train, target_values, max_epochs_games,
    #                         model_type="game", loss_threshold=loss_threshold)

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
    print("Matt")
    # getDataMatt()
    # feedForward()
    train_the_model_lstm()
    # test_the_model()




def tobey():
    mlp = MLP(input_size=57, hidden_size=32, output_size=1)
    trainMLP(mlp)
    print("Tobey - code started")






def main():
    #matt()
    lucas()
    # tobey()


# main()
if __name__ == '__main__':
    main()
