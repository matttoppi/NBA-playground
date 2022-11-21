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

    # Global variables for the class ====================================================
    file = None  # PATH to file containing the team history data
    dataframe = None  # Variable representing the processed/read-in CSV file

    def __init__(self, filename):
        """Initialize the factory by providing it with the file to read data from"""
        file = filename
        dataframe = pd.read_csv(file)

    def split_by_season(self):
        """
        This method takes the data from the CSV file and splits it up into matrices that represent each season
        :returns A list containing new matrices, with each representing all the team data for a given season/year
        """
        