import csv
import torch


input_file = "preVectorDATA/ALL_GAME_DATA.csv"
output_file = "features.csv"

def getDataMatt():
    # Read in the CSV file
    with open("preVectorDATA/ALL_GAME_DATA.csv", "r") as file:
        reader = csv.reader(file)

        # Get the index of the 'PTS' column
        headers = next(reader)
        pts_index = headers.index("PTS")

        # Create lists to store the data
        features = []
        targets = []

        # Loop through the rows in the CSV file
        for row in reader:
            # Convert the values in the row to floats
            values = [float(x) for x in row[1:]]

            # Add the values to the list of features
            features.append(values)

            # Add the target value (the points scored) to the list of targets
            targets.append(float(row[pts_index]))

    # Convert the lists of features and targets to PyTorch tensors
    features = torch.Tensor(features)
    targets = torch.Tensor(targets)

    train_size = int(0.8 * len(features))
    test_size = len(features) - train_size
    train_features, test_features = features[:train_size], features[train_size:]
    train_targets, test_targets = targets[:train_size], targets[train_size:]

    # Create a CSV writer object
    writer = csv.writer(open("features.csv", "w"))

    # Loop through the features tensor and write each row to the CSV file
    for row in features:
        writer.writerow(row)

    print("Done writing features.csv")

    # Do the same for the targets tensor
    writer = csv.writer(open("targets.csv", "w"))
    for row in targets:
        writer.writerow([row])

    print("Done writing targets.csv")



import csv
import torch

def load_features():
    # define the column names of the CSV file
    column_names = [
        "TEAM_SEASON",
        "TEAM_ID",
        "GAME_ID",
        "GAME_DATE",
        "HOME/AWAY",
        "WL",
        "MIN",
        "FGM",
        "FGA",
        "FG_PCT",
        "FG3M",
        "FG3A",
        "FG3_PCT",
        "FTM",
        "FTA",
        "FT_PCT",
        "OREB",
        "DREB",
        "REB",
        "AST",
        "STL",
        "BLK",
        "TOV",
        "PF",
        "PTS",
        "PLUS_MINUS",
        "raptor_box_offense",
        "raptor_box_defense",
        "raptor_onoff_offense",
        "raptor_onoff_defense",
        "raptor_offense",
        "raptor_defense",
        "raptor_total",
        "war_reg_season",
        "predator_offense",
        "predator_defense",
        "pace_impact",
        "ORPM",
        "DRPM",
        "RPM",
        "WINS",
        "W",
        "L",
        "OFFRTG",
        "DEFRTG",
        "NETRTG",
        "AST%",
        "AST/TO",
        "AST_RATIO",
        "OREB%",
        "DREB%",
        "REB%",
        "TOV%",
        "EFG%",
        "TS%",
        "PACE",
        "PIE",
        "POSS",
    ]

    # open the input and output files
    with open("preVectorDATA/ALL_GAME_DATA.csv", "r") as input_file, open("features.csv", "w") as output_file:
        # create CSV reader and writer objects
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)

        # specify the target column
        target_column = "PTS"

        # write the column names to the output file
        writer.writerow(column_names)

        # iterate over the rows of the input file
        for i, row in enumerate(reader):
            # store the value of the target column
            target_value = row[target_column]

            # convert the row data into PyTorch tensors
            input_tensor = torch.tensor([float(x) for x in row[1:-1]])
            target_tensor = torch.tensor([float(target_value)])

            # write the tensor data to the output file
            writer.writerow(input_tensor.tolist() + target_tensor.tolist())

            # print the progress
            if i % 1000 == 0:
                print(f"Processed {i} rows")

        print("Done writing features.csv")
