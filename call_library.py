from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import players


def get_team_id(team_name):
    # takes team abbreviation or full name as a string
    # returns team ID as a string

    all_teams = teams.get_teams()
    team_name = [x for x in all_teams if x['full_name'] == team_name][0]
    team_id = team_name['id']
    return team_id


def get_team_name(team_id):
    """
    takes team ID as a string
    returns team name as a string
    :param team_id:
    :return: team name as a string
    """
    from nba_api.stats.static import teams

    all_teams = teams.get_teams()
    team_name = [x for x in all_teams if x['id'] == team_id][0]['full_name']
    return team_name


def id_to_team_obj(id):
    """
    takes an ID to a team and returns a team object as a list
    :param id:
    :return:
    """
    # takes an ID to a team
    # returns a team object
    from nba_api.stats.static import teams

    all_teams = teams.get_teams()
    team = [x for x in all_teams if x['id'] == id]
    return team


def get_player_id(player_name):
    # takes a player name as a string
    # returns player ID as a string
    from nba_api.stats.static import players

    all_players = players.get_players()
    player_name = [x for x in all_players if x['full_name'] == player_name][0]
    player_id = player_name['id']
    return player_id


def id_to_player_obj(id):
    # takes an ID to a player
    # returns a player object
    from nba_api.stats.static import players

    all_players = players.get_players()
    player = [x for x in all_players if x['id'] == id]
    return player


def get_team_games(team_id, season):
    # takes team ID as a string
    # takes season as a string
    # returns a list of game objects for all games that season

    from nba_api.stats.endpoints import leaguegamefinder

    gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id, season_nullable=season)
    games = gamefinder.get_data_frames()[0]
    return games

def get_player_games(player_id, season):
    """
    takes player ID as a string and returns a list of game objects
    :param player_id:
    :param season:
    :return:
    """
    # takes player ID as a string
    # takes season as a string
    # returns a list of game objects
    from nba_api.stats.endpoints import leaguegamefinder

    gamefinder = leaguegamefinder.LeagueGameFinder(player_id_nullable=player_id, season_nullable=season)
    games = gamefinder.get_data_frames()[0]
    return games


def get_player_stats(player_id, season):
    """
    takes player ID as a string and returns a list of game objects
    :param player_id:
    :param season:
    :return:
    """

    from nba_api.stats.endpoints import playergamelog

    playergamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
    games = playergamelog.get_data_frames()[0]
    return games


def get_team_stats(team_id, season):
    """
    takes team ID as a string and returns a list of game objects
    :param team_id:
    :param season:
    :return:
    """
    from nba_api.stats.endpoints import teamgamelog

    teamgamelog = teamgamelog.TeamGameLog(team_id=team_id, season=season)
    games = teamgamelog.get_data_frames()[0]
    return games


def get_player_info(player_id):
    """
    takes player ID as a string
    # returns a list of player objects
    :param player_id:
    :return:
    """
    #
    from nba_api.stats.endpoints import commonplayerinfo

    player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
    player_info = player_info.get_data_frames()[0]
    return player_info


def get_team_info(team_id):
    """
    takes team ID as a string
    returns a list of team objects
    :param team_id:
    :return:
    """
    from nba_api.stats.endpoints import commonteamroster

    team_info = commonteamroster.CommonTeamRoster(team_id=team_id)
    team_info = team_info.get_data_frames()[0]
    return team_info


def get_player_name(player_id):
    """
    takes player ID as a string
    returns player name as a string
    :param player_id:
    :return:
    """
    from nba_api.stats.static import players

    all_players = players.get_players()
    player_name = [x for x in all_players if x['id'] == player_id][0]['full_name']
    return player_name


def get_player_team(player_id):
    """
    takes player ID as a string
    returns team ID as a string
    :param player_id:
    :return:
    """
    from nba_api.stats.endpoints import commonteamroster

    player_info = commonteamroster.CommonTeamRoster(player_id=player_id)
    player_info = player_info.get_data_frames()[0]
    team_id = player_info['TEAM_ID'][0]
    return team_id

def get_player_game_logs(player_id, param):
    """
    takes player ID as a string
    returns a list of game objects
    :param player_id:
    :return:
    """
    from nba_api.stats.endpoints import playergamelog

    playergamelog = playergamelog.PlayerGameLog(player_id=player_id)
    games = playergamelog.get_data_frames()[0]
    return games

def player_obj_for_game_id(game_id):
    """
    takes a game ID as a string
    returns a list of player objects
    :param game_id:
    :return:
    """
    from nba_api.stats.endpoints import boxscoreplayertrackv2

    boxscoreplayertrackv2 = boxscoreplayertrackv2.BoxScorePlayerTrackV2(game_id=game_id)
    players = boxscoreplayertrackv2.get_data_frames()[0]
    return players




def average_stat_for_team(stat_list):
    """
    takes team ID as a string
    takes season as a string
    takes stat as a string
    returns average stat as a float
    :param team_id:
    :param season:
    :param stat: a list of floats or ints to be averaged together for a team
    :return:
    """

    total = 0
    for stat in stat_list:
        total += stat
    average = total / len(stat_list)
    return average



def get_season_id(season):
    """
    takes season as a string
    returns season ID as a string
    :param season: string
    :return:
    """
    season = season.split('-')[0]
    season = '2' + season
    return season


def get_game_id(game):
    """
    takes game as a string
    returns game ID as a string
    :param game: string
    :return:
    """
    game = game.split('/')[1]
    game = game.split('.')[0]
    return game






# THIS IS WHERE WE GET DATA FOR A SPECIFIC GAME

def get_single_game_data(game_obj, game_num):

    return game_obj.loc[game_num]





# THIS IS WHERE WE GET THE ADVANCED STATS

def calculate_PER(player_id, season):
    """
    takes player ID as a string
    returns PER as a float
    :param player_id:
    :param season:
    :return:
    """
    from nba_api.stats.endpoints import playergamelog

    playergamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
    games = playergamelog.get_data_frames()[0]
    PER = games['PLAYER_EFFICIENCY_RATING'].mean()
    return PER


def calculate_true_shooting_percentage(player_id, season):
    """
    takes player ID as a string
    calculate true shooting percentage
    :param player_id: player ID as a string
    :return:
    """
    from nba_api.stats.endpoints import playergamelog

    #  only get preVectorDATA from the regular season
    playergamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
    games = playergamelog.get_data_frames()[0]


    # calculate true shooting percentage
    true_shooting_percentage = (games['PTS'] / (2 * (games['FGA'] + 0.44 * games['FTA']))).mean()
    return true_shooting_percentage


def calculate_effective_field_goal_percentage(player_id, season):
    """
    takes player ID as a string
    calculate effective field goal percentage
    :param player_id: player ID as a string
    :return:
    """
    from nba_api.stats.endpoints import playergamelog

    #  only get preVectorDATA from the regular season
    playergamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
    games = playergamelog.get_data_frames()[0]

    # calculate effective field goal percentage
    effective_field_goal_percentage = ((games['FGM'] + 0.5 * games['FG3M']) / games['FGA']).mean()
    return effective_field_goal_percentage


# method that puts the scores of every game in a season into a CSV file
# def get_season_scores():
#     """
#     takes season as a string
#     returns a list of game objects
#     :param season:
#     :return:
#     """
#     from nba_api.stats.endpoints import leaguegamelog
#
#     season_id = get_season_id(2014)
#
#     leaguegamelog = leaguegamelog.LeagueGameLog(season=season_id)
#     games = leaguegamelog.get_data_frames()[0]
#     print(games)
#     return games



#  calculates box plus minus for a player using the formula from https://www.basketball-reference.com/about/bpm.html
def calculate_box_plus_minues():
    """
    takes player ID as a string
    returns BPM as a float
    :param player_id:
    :return:
    """
    from nba_api.stats.endpoints import playergamelog

    playergamelog = playergamelog.PlayerGameLog(player_id=player_id)
    games = playergamelog.get_data_frames()[0]

#     calculate BPM using regression formula
