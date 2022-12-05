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


def main():
    # df = pd.read_csv('preVectorDATA/Player_Team_Lookups.csv') # player team lookups
    '''
    #copy = pd.DataFrame()
    df = df[['NAME', 'ABBREV', 'SEASON','PLAYER_ID']] #restricting to necessary columns
    copy = df # copy of df
    copy["NEW"] = df[['ABBREV','SEASON']].apply(lambda row: '-'.join(row.values.astype(str)), axis=1) # create a new column that combines the team abbreviation and season
    copy = copy[['NAME', 'NEW', 'PLAYER_ID']] #restrict columns to necessary
    copy = copy.pivot(columns='NEW', values='NAME')
    df1 = copy.apply(lambda x: pd.Series(x.dropna().to_numpy()))
    df1.to_csv('TEAM_ROSTERS.csv')
    '''

    # data = pd.read_csv('GameDATA.csv')
    # 
    # 
    # 
    # #pd.reset_option('all')
    # #print(df1.head(150))
    # 
    # 
    #raptorfactory = RaptorVectorFactory()
    #raptorfactory.team_vectors()

    gamefact = GameDataFeatureFactory()

    gamefact.split_by_season()

    list = gamefact.create_features()

    train, test = gamefact.split_train_test(list)

    # Lucas Test Code ===================================================================
    traintrans = torch.transpose(train[0], 0, 1)
    traintrans = nn.functional.normalize(traintrans, 2, dim=0)

    inFeats = len(traintrans[0])
    outFeats = 128

    print(f'inFeats: {inFeats}')
    print(f'traintrans: {traintrans}')
    #print(train[0][0])

    lucas_model = LucasModel(inFeats, outFeats)

    prediction = lucas_model.forward(traintrans)

    print(prediction)

    #print(f'Output Vals: {outputVals}')
    #print(f'Output softmax values: {outputWeights}')

    #torch.set_printoptions(profile="full")
    #print(f'First output val')
    #torch.set_printoptions(profile="default")

    # End Lucas Test Code ===============================================================


    # train = pd.read_csv('preVectorDATA/GameData.csv')
    # test = pd.read_csv('preVectorDATA/GameData.csv')

    # print(test)
    # print(len(test))

    '''
    tobey_train(train, test)
    matt_train(train, test)
    lucas_train(train, test)
    '''


# main()
if __name__ == '__main__':
    main()
