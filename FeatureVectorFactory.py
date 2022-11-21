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
        raptorIDs = self.player_id_dict()
        file = "preVectorDATA/RaptorData (2014-2023) - raptor_data14-23.csv"
        raptor_id_file = "preVectorDATA/raptor_player_id_dict - raptor_player_id_dict.csv"

    def load_raptor_data(file):
        """

        :param file:
        :return:
        """
        return pd.read_csv(file)

    name_to_id, id_to_name = player_id_dict()
    super_data = load_raptor_data("preVectorDATA/raptor_data.csv")

    def player_id_dict(self):
        """

        :param file: filename
        :returns dict, dict tuple of player_name->player_id and player_id->player_name respectively
        """
        df = pd.read_csv(raptor_id_file)
        x = dict(df.itertuples(False, None))
        y = {v: k for k, v in x.items()}
        return x, y


class TeamDataFeatureFactory:
    """This class is meant to be used to load in data from our CSV files and create feature vectors out of them."""

    def __init__(self, filename, debug=True):
        """Initialize the factory by providing it with the file to read data from"""
        self.file = filename  # PATH to the CSV file containing team data
        self.dataframe = pd.read_csv(self.file)  # The overall data frame containing all teams data for all seasons
        self.seasons = []  # List of data frames split by season
        self.debug = debug

    def split_by_season(self):
        """
        This method takes the data from the CSV file and splits it up into matrices that represent each season
        :returns No return, just appends the data to the self.seasons parameter
        """
        if len(self.seasons) != 0:
            print("Seasons list already has data in it! split_by_season() exiting...")
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

            # This code might work if we take out all of the columns with string values
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
            torch.stack((bigtensor, x))

        if self.debug:
            print(f"Function create_team_features() tensor size: {bigtensor.size()}")
            print(f"Expected size of returned tensor: (10, 30, ~18)")
        return bigtensor
