import csv
import torch
import numpy as np


input_file = "preVectorDATA/ALL_GAME_DATA.csv"


# feed forward neural network
def feedForward():
    print("Feed Forward Neural Network")
    train_input, train_target, test_input, test_target = data_load()

    # train_input = train_input.view(-1, 23)  # this is reshaping the data to be 1D
    # train_target = train_target.view(-1, 1) # this is reshaping the data to be 1D
    #
    # test_input = test_input.view(-1, 23)  # this is reshaping the data to be 1D
    # test_target = test_target.view(-1, 1) # this is reshaping the data to be 1D


    # Define the model
    model = torch.nn.Sequential(
        torch.nn.Linear(57, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1),
    )

    # Define the loss function
    loss_fn = torch.nn.MSELoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


    lowest_loss = 100000
    loss_count = 0

#time each epoch
    import time

    # Iterate over the training data in batches
    for epoch in range(1000):
        i=0
        start_time = time.time()
        for row in train_input:
            # Forward pass: Compute predicted y by passing x to the model
            #y_pred = model(train_input)
            y_pred = model(row)

            # Compute and print loss
            loss = loss_fn(y_pred, train_target[i])
            # print(epoch, loss.item())
            # print(i, " loss: ", loss.item())
            i+=1
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print the loss every 1000 iterations
            if i % 1000 == 0:
                print('Epoch: ', epoch, "Iter", i, ' loss: ', loss.item())
        end_time = time.time()
        print("\n\n\n\n\n\n-----------------------Epoch: ", epoch, " Loss: ", loss.item(), " Time: ", end_time -
              start_time)
        if loss < lowest_loss:
            lowest_loss = loss
            torch.save(model.state_dict(), "models/feedForwardModel.pt")



    # Test the model
    with torch.no_grad():
        y_pred = model(test_input)
        loss = loss_fn(y_pred, test_target)
        print('Test loss:', loss.item())

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt') # this is saving the model to a file so it can be used later






# def test_the_model():
#     # Define the model
#     model = torch.nn.Sequential(
#         torch.nn.Linear(57, 128),
#         torch.nn.ReLU(),
#         torch.nn.Linear(128, 128),
#         torch.nn.ReLU(),
#         torch.nn.Linear(128, 1),
#     )
#
#     # Load the model checkpoint
#     model.load_state_dict(torch.load('model.ckpt'))
#
#     # Test the model
#     with torch.no_grad():
#         y_pred = model(test_input)
#         loss = loss_fn(y_pred, test_target)
#         print('Test loss:', loss.item())






def data_load():
    # Read the CSV file into a NumPy array
    with open('preVectorDATA/ALL_GAME_DATA.csv', 'r') as f:
        reader = csv.reader(f)
        data = np.array([[col for j, col in enumerate(row) if j > 0] for i, row in enumerate(reader) if i > 0],
                        dtype=np.float32)

    # Split the data into training and test sets
    train_input = data[:18000, :]
    train_target = data[:, 23]
    test_input = data[18000:, :]
    test_target = data[:, 23]

    # Convert the NumPy arrays into PyTorch tensors
    return torch.from_numpy(train_input), torch.from_numpy(train_target), torch.from_numpy(test_input), torch.from_numpy(test_target)
