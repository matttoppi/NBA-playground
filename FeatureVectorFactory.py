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
        #raptorIDs = self.player_id_dict()
        name_to_id, id_to_name = self.player_id_dict()
        super_data = self.load_raptor_data()

    def load_raptor_data(file):
        """

        :param file:
        :return:
        """
        return pd.read_csv(file)

    def player_id_dict(self):
        """
        :param file: filename
        :returns dict, dict tuple of player_name->player_id and player_id->player_name respectively
        """
        df = pd.read_csv(self.raptor_id_file)
        x = dict(df.itertuples(False, None))
        y = {v: k for k, v in x.items()}
        return x, y

    def team_vectors(self):
        """
        This method is used to create the team vectors for the raptor data

        """
        self.name_to_id("Stephen Curry")



class TeamDataFeatureFactory:
    """This class is meant to be used to load in data from our CSV files and create feature vectors for
    team stats out of them."""

    def __init__(self, filename, debug=True):
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

            stats = (gp, w, l, offrtg, defrtg, netrtg, astp, astto, astratio, orebp, drebp, rebp, tovp, efgp, tsp, pace,
                     pie, poss)

            teamtensor = torch.stack(stats)

            if self.debug:
                print(f"Function create_single_season_features() tensor size: {teamtensor.size()}")
                print(f"Expected size of returned tensor: (30, ~18)")

            return teamtensor

            # This code might work if we take out all the columns with string values
            # numpyframe = season.to_numpy()
            # tensor = torch.tensor(numpyframe)
            # return tensor

    def create_team_features(self):
        """
        Takes the self.seasons parameter and creates a tensor of individual tensors for each dataframe in the list
        :return: A list containing tensors of size [numTeams (30), numStats (18 without team names/labels)], with one
        tensor for each season (10) we cover. Overall shape should be [10, 30, ~18]
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
            print(f"Expected size of returned tensor: (10, 30, ~18)")
        return bigtensor


class RMPFeatureFactory:
    """This class is meant to be used to load in data from our CSV files and create feature vectors for
    player RPM stats out of them."""
    def __init__(self, filename, debug=True):
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
        self.debug = debug

    def split_by_season(self):
        """
        This† method takes the data from the CSV file and splits it up into matrices that represent each season
        :returns No return, just appends the data to the self.seasons parameter
        """
        if len(self.seasons) != 0:
            print("Seasons list already has data in it! GameData split_by_season() exiting...")
        else:
            seasonids = (22013, 22014, 22015, 22016, 22017, 22018, 22019, 22020, 22021, 22022)
            for x in seasonids:
                mask = self.dataframe['SEASON_ID'] == x  # Split the data frame to only have the current season rows
                self.seasons.append(self.dataframe[mask])  # Append the current season to the list of split seasons

    def create_game_tuples(self):
        """
        This method will create a list of tuples. The pair will be matching the game ids for the home and away teams
        :return: A list of tuples containing the game ids for the home and away teams
        """
        game_tuples = []
        for season in self.seasons:
            for index, row in season.iterrows():
                game_tuples.append((row['GAME_ID'], row['GAME_ID']))
        return game_tuples
