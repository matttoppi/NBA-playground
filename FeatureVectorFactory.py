# this is a class that takes in data from the preVectorDATA folder and creates a feature vector data file to put into
# the featureVECTOR data folder

import pandas as pd
import numpy as np
import math
import csv
import os
import sys
import time
import datetime
import random
import torch
from sklearn.model_selection import train_test_split


# this is a class that takes in data from the preVectorDATA folder and creates a feature vector data file to put into
# the featureVECTOR data folder

class RaptorVectorFactory:
    """
    This method takes in a player name and returns the player's ID
    """
    file = ""
    raptor_id_file = ""

    def _init_(self):
        # load the raptor ID data file
        file = "preVectorDATA/RaptorData (2014-2023) - raptor_data14-23.csv"
        raptor_id_file = "preVectorDATA/raptor_player_id_dict.csv"
        # raptorIDs = self.player_id_dict()
        # name_to_id, id_to_name = self.player_id_dict()
        super_data = self.load_raptor_data()
        super_data.to_csv()
        self.team_vectors()

    def load_raptor_data(self):
        """

        :param file:
        :return:
        """
        return pd.read_csv("preVectorDATA/RaptorData.csv")

    def player_id_dict(self):
        """
        :param file: filename
        :returns dict, dict tuple of player_name->player_id and player_id->player_name respectively
        """
        df = pd.read_csv("preVectorDATA/raptor_player_id_dict.csv")
        df = df[['player_name', 'player_id']]
        x = dict(df.itertuples(False, None))
        y = {v: k for k, v in x.items()}
        return x, y

    def team_vectors(self):
        """
        This method is used to create the team vectors for the raptor data

        """
        super_data = pd.read_csv("preVectorDATA/RaptorData.csv")
        rosters = pd.read_csv('preVectorDATA/TEAM_ROSTERS.csv')
        feature_vectors = pd.DataFrame()
        name_to_id, id_to_name = self.player_id_dict()
        # super_data.to_csv("./super_data.csv")
        for (team_season, roster) in rosters.items():  # iterate through team-season columns
            season = team_season[-2:]
            feature_vectors[team_season] = pd.Series(np.zeros(shape=(13), dtype=float))  #blank col for each team-season
            print(team_season)
            count = 0

            for player in roster:  # iterate through rosters for each team-season column
                if str(player) != "nan" and not str(player).isdigit() and int(season) < 23:  #verify player name is valid
                    print(player)  # Individual player from the roster

                    try:
                        id = name_to_id[player]
                        lookup = str(id) + "-" + str(season)
                        p_d = super_data[lookup].to_numpy()
                        print("minutes:", p_d[1])
                        feature_vectors[team_season] = feature_vectors[team_season] + (p_d[1] * p_d)
                        count += 1
                    except:
                        feature_vectors[team_season] = feature_vectors[team_season] + pd.Series(np.zeros(shape=13,
                                                                                                         dtype=float))
                        continue

                    # add player's raptor data to temp DF

                    # print(temp)
                    # temp = pd.concat([temp, temp], axis=0)

                    if count > 0:
                        print("count: " + str(count))

            #print(temp[0])

            #feature_vectors[team_season].append(temp[0].transpose())
            feature_vectors[team_season] = feature_vectors[team_season] / count

        del feature_vectors["Unnamed: 0"]
        feature_vectors.to_csv("./features.csv")

        #         print("\n" + player)
        #         try:
        #             id = name_to_id[player]
        #             lookup = str(id) + "-" + str(season)
        #
        #             #add the player's data to the temp dataframe
        #             temp = temp.append(super_data.loc[super_data['player_id_season'] == lookup])
        #             count += 1
        #
        #             #super_data.iloc[super_data.index[super_data["player_id_season"] == lookup]].transpose()
        #         except:
        #             # print("player not found")
        #             continue
        # if count>0:
        #     print("count: ", count)
        #     # temp[team_season] = (temp[team_season] / count)
        # print(temp)
        # # feature_vectors[team_season] = temp.iloc[:, :].sum(axis=1)

        # math goes here to combine all players vectors. Scale the vectors based on the minutes player (MP) played
        # by each player

        # get the minutes played for each player

        # temp2 =
        # feature_vectors[team_season] = temp2 # here add the weighted team vector to output


class TeamDataFeatureFactory:
    """This class is meant to be used to load in data from our CSV files and create feature vectors for
    team stats out of them."""

    def __init__(self, filename="preVectorDATA/TeamData.csv", debug=True):
        """Initialize the factory by providing it with the file to read data from"""
        self.file = filename  # PATH to the CSV file containing team data
        self.dataframe = pd.read_csv(self.file)  # The overall data frame containing all teams data for all seasons
        self.seasons = []  # List of data frames split by season
        self.debug = debug
        # Can also put split_by_season in here since you have to do it anyway

    def split_by_season(self):
        """
        This method takes the data from the CSV file and splits it up into matrices that represent each season
        :returns No return, just appends the data to the self.seasons parameter
        """
        if len(self.seasons) != 0:
            print("Seasons list already has data in it! Team split_by_season() exiting...")
        else:
            for x in range(14, 23):
                mask = self.dataframe['SEASON'] == x  # Split the data frame to only have the current season rows
                self.seasons.append(self.dataframe[mask])  # Append the current season to the list of split seasons

    def create_single_season_features(self, season):
        """
        Creates a PyTorch feature matrix/tensor out of the season param
        :param season: A dataframe holding the data for all teams from an individual season.
        :return: A PyTorch tensor containing all the features representing a teams performance during a given season
        """
        if len(season) == 0:
            print("Empty season data frame provided to create_single_season_features! Exiting...")
            return None
        else:
            teamid = torch.tensor(season['TEAM_ID'].values)
            gp = torch.tensor(season['GP'].values)
            w = torch.tensor(season['W'].values)
            l = torch.tensor(season['L'].values)
            offrtg = torch.tensor(season['OFFRTG'].values)
            defrtg = torch.tensor(season['DEFRTG'].values)
            netrtg = torch.tensor(season['NETRTG'].values)
            astp = torch.tensor(season['AST%'].values)
            astto = torch.tensor(season['AST/TO'].values)
            astratio = torch.tensor(season['AST RATIO'].values)
            orebp = torch.tensor(season['OREB%'].values)
            drebp = torch.tensor(season['DREB%'].values)
            rebp = torch.tensor(season['REB%'].values)
            tovp = torch.tensor(season['TOV%'].values)
            efgp = torch.tensor(season['EFG%'].values)
            tsp = torch.tensor(season['TS%'].values)
            pace = torch.tensor(season['PACE'].values)
            pie = torch.tensor(season['PIE'].values)
            poss = torch.tensor(season['POSS'].values)

            stats = (teamid, gp, w, l, offrtg, defrtg, netrtg, astp, astto, astratio, orebp, drebp, rebp, tovp, efgp,
                     tsp, pace, pie, poss)

            teamtensor = torch.stack(stats)  # might have to be concat

            if self.debug:
                print(f"Function create_single_season_features() tensor size: {teamtensor.size()}")
                print(f"Expected size of returned tensor: (30, ~19)")

            return teamtensor

            # This code might work if we take out all the columns with string values
            # numpyframe = season.to_numpy()
            # tensor = torch.tensor(numpyframe)
            # return tensor

    def create_team_features(self):
        """
        Takes the self.seasons parameter and creates a tensor of individual tensors for each dataframe in the list
        :return: A list containing tensors of size [numTeams (30), numStats (19 with team id)], with one
        tensor for each season (10) we cover. Overall shape should be [10, 30, ~19]
        """
        bigtensor = torch.empty(1)  # Aggregate tensor that we will stack each new season onto
        for season in self.seasons:
            x = self.create_single_season_features(season)
            if x is None:
                print("Tensor returned from single season creator was None! Exiting create_team_features...")
                return None
            torch.stack((bigtensor, x))

        if self.debug:
            print(f"Function create_team_features() tensor size: {bigtensor.size()}")
            print(f"Expected size of returned tensor: (10, 30, ~19)")
        return bigtensor


class RPMFeatureFactory:
    """This class is meant to be used to load in data from our CSV files and create feature vectors for
    player RPM stats out of them."""

    def __init__(self, filename="preVectorDATA/RPM_Player_Season.csv", debug=True):
        """Initialize the factory by providing it with the file to read data from"""
        self.file = filename  # PATH to the CSV file containing team data
        self.dataframe = pd.read_csv(self.file)  # The overall data frame containing all teams data for all seasons
        self.seasons = []  # List of data frames split by season
        self.debug = debug
        # Can also put split_by_season in here since you have to do it anyway

    def split_by_season(self):
        """
        This† method takes the data from the CSV file and splits it up into matrices that represent each season
        :returns No return, just appends the data to the self.seasons parameter
        """
        if len(self.seasons) != 0:
            print("Seasons list already has data in it! RPM split_by_season() exiting...")
        else:
            for x in range(14, 23):
                mask = self.dataframe['Season'] == x  # Split the data frame to only have the current season rows
                self.seasons.append(self.dataframe[mask])  # Append the current season to the list of split seasons

    def create_single_season_features(self, season):
        """
        Create a feature vector representing every player's RPM stats for a given season
        :param season: The season dataframe being turned into a tensor
        :return: A PyTorch tensor of size (numPlayers (variable per season),
        numFeatures (~4 not including names/labels))
        """
        if len(season) == 0:
            print("Empty season data frame provided to create_single_season_features! Exiting...")
            return None
        else:
            orpm = torch.tensor(season['ORPM'].values)
            drpm = torch.tensor(season['DRPM'].values)
            rpm = torch.tensor(season['RPM'].values)
            wins = torch.tensor(season['WINS'].values)
            # teamnum = torch.tensor(season['TEAMID'].values)
            # playerName -> team (abbrev) -> Team id number

            stats = (orpm, drpm, rpm, wins)

            seasontensor = torch.stack(stats)

            if self.debug:
                print(f"Function create_single_season_features() tensor size: {seasontensor.size()}")
                print(f"Expected size of returned tensor: (#playersInSeason, ~4)")

            return seasontensor


class GameDataFeatureFactory:
    """
    This class is responsible for creating feature vectors for the individual game data. We will need to first pair
    up the game IDs


    """

    def __init__(self, filename="preVectorDATA/GameDATA.csv", debug=True):
        """Initialize the factory by providing it with the file to read data from"""
        self.file = filename  # PATH to the CSV file containing team data
        self.dataframe = pd.read_csv(self.file)  # The overall data frame containing all teams data for all seasons
        self.seasons = []  # List of data frames split by season
        self.seasonstuples = []  # List of data frames split by season, with tuples based on game ID
        self.debug = debug

    def split_by_season(self):
        """
        This method takes the data from the CSV file and splits it up into matrices that represent each season
        :returns No return, just appends the data to the self.seasons parameter
        """
        if len(self.seasons) != 0:
            print("Seasons list already has data in it! GameData split_by_season() exiting...")
        else:
            seasonids = (22014, 22015, 22016, 22017, 22018, 22019, 22020, 22021, 22022, 22023)
            for x in seasonids:
                mask = self.dataframe['SEASON_ID'] == x  # Split the data frame to only have the current season rows
                self.seasons.append(self.dataframe[mask])  # Append the current season to the list of split seasons

    def create_game_tuples(self):
        """
        This method takes the season-wise data gotten from split_by_season() and returns a similar object,
        except this one has all the rows of games paired up by game ID (since there are two rows for each game,
        one row for each team's game stats).
        :return: Appends the seasons to the self.seasonstuples parameter
        """
        if len(self.seasonstuples) != 0:
            print("Seasons Tuples already has data in it! GameData create_game_tuples() exiting...")
        elif len(self.seasons) == 0:
            print("Seasons list is empty! You either haven't called split_by_season() yet, or you have a problem...")
        else:
            for season in self.seasons:
                game_tuples = []
                for index, row in season.iterrows():
                    game_tuples.append((row['GAME_ID'], row['GAME_ID']))
                self.seasonstuples.append(game_tuples)

    def create_features(self):
        """
        This method takes each row of GameData and makes a feature vector out of it.
        :return: A feature matrix
        """
        seasonstensorlist = []
        for season in self.seasons:
            SEASID = torch.tensor(season['SEASON_ID'].values, dtype=torch.float32)  # index 0
            TEAMID = torch.tensor(season['TEAM_ID'].values, dtype=torch.float32)  # index 1
            GAMEID = torch.tensor(season['GAME_ID'].values, dtype=torch.float32)  # index 2
            WL = torch.tensor(season['WL'].values, dtype=torch.float32)  # index 3
            MIN = torch.tensor(season['MIN'].values, dtype=torch.float32)  # index 4
            FGM = torch.tensor(season['FGM'].values, dtype=torch.float32)  # index 5
            FGA = torch.tensor(season['FGA'].values, dtype=torch.float32)  # index 6
            FG_PCT = torch.tensor(season['FG_PCT'].values, dtype=torch.float32)  # index 7
            FG3M = torch.tensor(season['FG3M'].values, dtype=torch.float32)  # index 8
            FG3A = torch.tensor(season['FG3A'].values, dtype=torch.float32)  # index 9
            FG3_PCT = torch.tensor(season['FG3_PCT'].values, dtype=torch.float32)  # index 10
            FTM = torch.tensor(season['FTM'].values, dtype=torch.float32)  # index 11
            FTA = torch.tensor(season['FTA'].values, dtype=torch.float32)  # index 12
            FT_PCT = torch.tensor(season['FT_PCT'].values, dtype=torch.float32)  # index 13
            OREB = torch.tensor(season['OREB'].values, dtype=torch.float32)  # index 14
            DREB = torch.tensor(season['DREB'].values, dtype=torch.float32)  # index 15
            REB = torch.tensor(season['REB'].values, dtype=torch.float32)  # index 16
            AST = torch.tensor(season['AST'].values, dtype=torch.float32)  # index 17
            STL = torch.tensor(season['STL'].values, dtype=torch.float32)  # index 18
            BLK = torch.tensor(season['BLK'].values, dtype=torch.float32)  # index 19
            TOV = torch.tensor(season['TOV'].values, dtype=torch.float32)  # index 20
            PF = torch.tensor(season['PF'].values, dtype=torch.float32)  # index 21
            PTS = torch.tensor(season['PTS'].values, dtype=torch.float32)  # index 22
            PLUS_MINUS = torch.tensor(season['PLUS_MINUS'].values, dtype=torch.float32)  # index 23

            stats = (SEASID, TEAMID, GAMEID, WL, MIN, FGM, FGA, FG_PCT, FG3M, FG3A, FG3_PCT, FTM, FTA, FT_PCT, OREB,
                     DREB, REB, AST, STL, BLK, TOV, PF, PTS, PLUS_MINUS)

            gamestensor = torch.stack(stats)
            # print(f'Games Tensor: {gamestensor}')
            seasonstensorlist.append(gamestensor)

        return seasonstensorlist

    def split_train_test(self, seasontensorlist):
        train = seasontensorlist[:7]
        test = seasontensorlist[7:]
        # print(f'Train: {train}')
        # print(f'Test: {test}')

        return train, test


def get_final_tensors(game_tensors, weighted_tensors):
    """
    takes the data from the gametensors dataframe
    :param game_tensors:  torch.tensor
    :param wegihted_tesnors:  torch.tensor
    :return:
    """

    final_tensor_list = []

    for team in weighted_tensors:  # for each team in weighted tensors
        teamid = team[0]  # get the teamid from index 0
        for game in game_tensors:  # for each game in game tensors
            if teamid == game[1]:  # if the teamid matches the teamid in the game tensor
                final_tensor = torch.cat((team, game), 0)  # concatenate the two tensors
                final_tensor_list.append(final_tensor)  # append the final tensor to the final tensor list



    final_tensor_list = pd.DataFrame(final_tensor_list) # convert the final tensor list to a dataframe
    final_tensor_list.to_csv('./vectorDATA/final_tensor_list.csv') # save the final tensor list to a csv file
