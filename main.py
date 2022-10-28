# https://github.com/swar/nba_api
# https://github.com/swar/nba_api/tree/master/docs/examples

from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams

from call_library import *
import pandas as pd


def main():
    # doing some testing

    # get true shooting percentage for a player

    print("\n\n\n------\nGET PLAYER ID\n")
    player_name = 'Jayson Tatum'
    player_id = get_player_id(player_name)
    season = '2022-23'
    print("Player ID for " + player_name + " is " + player_id)

    print("\n\n\n------\nGET TRUE SHOOTING PERCENTAGE (using player_id from above\n")
    true_shooting_percentage = calculate_true_shooting_percentage(player_id, season)
    print(f'{player_name} has a true shooting percentage of {true_shooting_percentage} in the {season} season.')

    print("\n\n\n------\nGET PLAYER STATS\n")
    player_name = 'Wilt Chamberlain'
    player_id = get_player_id(player_name)
    season = '1962-63'

    player_stats = get_player_stats(player_id, season)
    print(player_stats)

    print("\n\n\n------\nGET SEASON ID FROM SEASON\n")
    season_id = get_season_id(season)
    print(f"The season ID for the 1962-63 season is {season_id}")

    #  get effective field goal percentage for a player
    print("\n\n\n------\nGET EFFECTIVE FIELD GOAL PERCENTAGE\n")
    player_name = 'Michael Jordan'
    player_id = get_player_id(player_name)
    season = '1996-97'

    effective_field_goal_percentage = calculate_effective_field_goal_percentage(player_id, season)
    print(
        f'{player_name} has an effective field goal percentage of {effective_field_goal_percentage} in the {season} season.')

    # this gets game data for the 40th game of the season
    print("\n\n\n----------\nGET GAME DATA FOR THE 40TH GAME OF THE CELTICS 2021-22 SEASON\n")
    # get the score for a specific game

    team_name = 'Boston Celtics'
    team_id = get_team_id(team_name)
    season = '2021-22'
    games = get_team_games(team_id, season)

    print(get_single_game_data(games, 40))

    # uncomment to get the data for all the games in for that season

    # x = 0
    #
    # while x < 82:
    #     print(get_single_game_data(games, x))
    #     x += 1


if __name__ == '__main__':
    main()
