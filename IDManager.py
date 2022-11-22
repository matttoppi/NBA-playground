from fuzzywuzzy import fuzz
import pandas as pd


class IDManager:
    """
        Raptor ID Dict (name) -> RPM (name to  team abbreviation/season) -> this file to get team ID

        Team ID format: 0-30 based on alphabetical order
        EX: ATL = 0, BOS = 1, WAS = 29

        Go from name in either raptor or RPM, get team abbreviation from that same file. Then from there, send team
        abbreviation in to this file, this file takes the abbreviation and matches it to a value from 0-29 based on
        teams alphabetical order. This number is already in team stats, so we just have to match the players to the
        team numbers.
        """

    # list of all nba team abbreviations
    team_abbrevs = ["ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", "HOU", "IND", "LAC", "LAL",
                    "MEM", "MIA", "MIL", "MIN", "NOP", "NYK", "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR",
                    "UTA", "WAS"]

    alternate_team_abbrevs = ["SA", "NO", "GS", "UTAH", "NY"]

    def getteam_enum(self, abbr):
        """
        :returns the index of the team abbreviation
        :return:
        """

        # if the abbreviation is in the list, return the index
        if abbr in self.team_abbrevs:
            return self.team_abbrevs.index(abbr)

        # match the abbr to the closest abbreviation from team_abbrevs using regex similarity
        else:
            best_match_index = 0
            best_match_score = 0
            for i in range(len(self.team_abbrevs)):
                testscore = fuzz.ratio(abbr, self.team_abbrevs[i])
                if fuzz.ratio(abbr, self.team_abbrevs[i]) > best_match_score:
                    best_match_index = i
                    best_match_score = testscore

            return best_match_index


def add_playerid_colum():
    """
    using pandas dataframes
    loop through the raptor player id dictionary file
    for each player, get their id
    
    loop through the player/team lookups file and create a new colum in that row that holds the player id
    :return: 
    """

    # read in the raptor player id dictionary file
    raptor_player_id_dict = pd.read_csv("preVectorDATA/raptor_player_id_dict.csv")

    # read in the player/team lookups file
    changeFile = pd.read_csv("preVectorDATA/RPM_Player_Season.csv")

    # for all the players in the raptor player id dictionary file
    for index, row in raptor_player_id_dict.iterrows():
        # get the player name
        player_name = row["player_name"]
        player_id = row["id_num"]
        print("\nLOOP:", player_name, player_id)

        # loop through the player/team lookups file
        for index2, row2 in changeFile.iterrows():

            # if the player name is in the player/team lookups file
            if player_name in row2["NAME"]:
                # add the player id to the player/team lookups file
                changeFile.at[index2, "PLAYER_ID"] = player_id

        print("\nID added\n\n")

    # update the csv file Player_Team Lookups.csv with the dataframe
    changeFile.to_csv("preVectorDATA/RPM_Player_Season.csv", index=False)


def delete_row_no_id():
    """
        # if the PLAYER_ID column is empty, delete that row
    """
    player_team_lookups = pd.read_csv("preVectorDATA/RPM_Player_Season.csv")
    player_team_lookups = player_team_lookups.dropna(subset=["PLAYER_ID"]) # drop rows with no player id
    player_team_lookups.to_csv("preVectorDATA/RPM_Player_Season.csv", index=False)
