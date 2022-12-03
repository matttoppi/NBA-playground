# https://github.com/swar/nba_api
# https://github.com/swar/nba_api/tree/master/docs/examples

from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams

from call_library import *
from FeatureVectorFactory import *
from IDManager import *
import pandas as pd
import numpy as np
from Training import *


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
    raptorfactory = RaptorVectorFactory()
    raptorfactory.team_vectors()


    gamefact = GameDataFeatureFactory()

    gamefact.split_by_season()

    list = gamefact.create_features()

    train, test = gamefact.split_train_test(list)

    # train = pd.read_csv('preVectorDATA/GameData.csv')
    # test = pd.read_csv('preVectorDATA/GameData.csv')

    # print(test)
    # print(len(test))

    '''
    tobey_train(train, test)
    matt_train(train, test)
    lucas_train(train, test)
    '''




#main()
if __name__ == '__main__':
    main()
