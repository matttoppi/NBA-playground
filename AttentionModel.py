"""
This file contains the code for Lucas's Neural Networks. These will be attention-based models, with one being a very
simple scaled-dot product attention, and the other being a custom attention implementation.
"""

from evalMetric import *
import torch as torch
import torch.nn as nn
import math as math


class BasicProjectionLayer(nn.Module):
    """
    This class is a simple linear layer that projects the data from GameData tensors into a latent space to work with.
    The parameters are in_feats and out_feats, with in_feats being the size of each input tensor (ie one row of game
    data), and out_feats being the desired length of the output latent space vector.
    Currently:
    in_feats = rows = ~2000
    out_feats = X
    """

    def __init__(self, in_feats_from_gamedata, out_feats_from_gamedata):
        super(BasicProjectionLayer, self).__init__()
        self.in_features_from_gamedata = in_feats_from_gamedata
        self.out_features_from_gamedata = out_feats_from_gamedata
        self.raw_gamedata_projection_layer = nn.Linear(self.in_features_from_gamedata, self.out_features_from_gamedata)

    def forward(self, in_feats):
        """
        This is the forward method for projection, which simply sends the input tensors through a linear layer
        :param in_feats: The feature vector tensors gotten from GameDataFeatureFactory. This (ideally) will be
        batched by season, so the in_feats input would only be data from one season, so shape (1, 30, 23) or (30, 23)
        :return: A latent-space vector matrix of the game data
        """
        #print(f'In: {self.in_features_from_gamedata}; Out: {self.out_features_from_gamedata}')
        return self.raw_gamedata_projection_layer(in_feats)


class BasicAttentionLayer(nn.Module):
    """
    This class holds the layer that will be performing the attention calculations on our latent-space game data vectors
    that we will get as output from out BasicProjectionLayer.
    """

    def __init__(self, layer_dimensions):
        """
        Initialize the attention layer.
        :param layer_dimensions: This is equal to the number of features outputted by the projection layer. This
        SHOULD stay the same throughout this layer. Currently, this value is: X
        """
        super(BasicAttentionLayer, self).__init__()
        self.layer_dims = layer_dimensions

        # Layer weights for Q, K, and V
        # self.WQ = nn.Linear(self.layer_dims, self.layer_dims)
        # self.WK = nn.Linear(self.layer_dims, self.layer_dims)
        # self.WV = nn.Linear(self.layer_dims, self.layer_dims)

        # Final softmax layer
        self.softmax = nn.Softmax()

    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        """
        This method performs the attention calculations with the given query, key, and values. This is an
        implementation of scaled dot-product attention.
        """
        # Calculate the value to scale query dot key_transpose by
        scale_value = math.sqrt(self.layer_dims)

        # Transpose key
        key_transpose = torch.transpose(key, -2, -1)
        #print(f'query size: {query.size()}')
        #torch.set_printoptions(profile="full")
        #print(query[0])
        #torch.set_printoptions(profile="default")
        # Calculate the query dot key score
        scores = torch.matmul(query, key_transpose)
        #print("Before softmax")
        #print(scores.size())
        #print(scores)

        scores = scores / scale_value
        #print("After scaling")
        #print(scores)

        # Send the score through a softmax
        softmaxed_score = self.softmax(scores)
        #print("After softmax")
        #torch.set_printoptions(profile="full")
        #print(softmaxed_score[0])
        #torch.set_printoptions(profile="default")

        # Return the attention matrix along with the softmax scores for the matrix
        return torch.matmul(softmaxed_score, value), softmaxed_score

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        """
        The forward function that passes the input query, key, and value through the attention function
        :return:
        """
        return self.attention(query, key, value)


class BasicPredictionLayer(nn.Module):
    """
    This method takes the output attention matrix from the attention layer and uses it to make a prediction for scores.
    """

    def __init__(self):
        super(BasicPredictionLayer, self).__init__()
        self.reverseProjection = nn.Linear(128, 24)
        self.activationLayer = nn.Softmax()
        self.predictRow = nn.Linear(24, 1)

    def forward(self, attention_data):
        revProj = self.reverseProjection(attention_data)
        denorm = nn.functional.normalize(revProj, 1/2, dim=0)
        return self.predictRow(self.activationLayer(denorm))


class LucasModel(nn.Module):
    """
    This class is what combines all the lower-level layers into one big model.
    """

    def __init__(self, in_feats_from_gamedata, out_feats_from_gamedata):
        super(LucasModel, self).__init__()
        # General Model Parameters Below --------------------------------------------

        # Linear Projection Layer
        self.in_feats_from_gamedata = in_feats_from_gamedata
        self.out_feats_from_gamedata = out_feats_from_gamedata
        self.projection = BasicProjectionLayer(self.in_feats_from_gamedata, self.out_feats_from_gamedata)

        # Attention Layer
        self.layer_dimensions = in_feats_from_gamedata
        self.attention = BasicAttentionLayer(self.layer_dimensions)

        # Decoder/Prediction Layer
        self.prediction = BasicPredictionLayer()

    def forward(self, input_data):
        projections = self.projection(input_data)  # Projecting the data into a latent space
        #print(f'Projections size: {projections.size()}')

        attentionScores, softmaxVals = self.attention(projections, projections, projections)

        predictions = self.prediction(attentionScores)

        return predictions


class TrainLoop:
    """

    """

    def __init__(self, MyModel, train_data, target_data):
        self.epochs = 0
        self.total_loss = 0
        self.model = MyModel
        self.train_data = train_data
        self.target_data = target_data
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)


    def trainOnce(self):
        """

        """

        self.optimizer.zero_grad()
        ouput = self.model.forward(self.train_data)

        loss = self.loss_function(ouput, self.target_data)

        print(loss.item())

        loss.backward()

        self.optimizer.step()

        self.total_loss += loss.item()

    def testPrediction(self, fakeInput, actualValues):
        output = self.model.forward(fakeInput)

        print(f"Predicted: {output}")
        print(f"Actual: {actualValues}")
        print(f"Error: {output - actualValues}")




"""
Questions for sepideh: 
- How are we meant to get a desired output from an attention model? Is it just a linear layer at the end that takes the
output of the attention layer and projects it? Or do I have to multiply the attention values by the original data, and 
then throw it into a linear layer?
- After my model is trained, how do I get it to make predictions when only given two teams as input?
- The output of attention is a list of probabilities, where what I want is a set prediction value. How would you 
recommend I translate this?
- The number of games played in a given season differ from season to season. How would you recommend I account for that
with my model, since I have to have set sizes of input and output for each layer?
- Is there any way I can project data into a latent space without changing the dimensions of it? Right now I am 
projecting downwards and losing some information, where ideally I would keep it the same size throughout until I need
to make a prediction.
- Some of my data is meant to simply represent something (ie a number for a team or game identifier) so that data can be
linked to a team or player, etc.  Ex: the game IDs are a very large number, so when sent into the attention/softmax it 
weights these heavier even though it really doesn't represent any useful data. Should these values be included in my 
attention, or will this throw things off?

- So currently, my model takes in tensors generated from our dataset, and each tensor represents an NBA season. This is
sent through a linear layer to project it into a latent space (which I was told by Dietz to do). Then, the latent space
matrix is sent into my "attention" layer which gives me a matrix of size #games x #games (2460 x 2460). This attention 
matrix will then be fed into a layer that takes the attention values and does SOMETHING to them to get a final output.
This final output is meant to be some sort of probability, which I will then use to predict the score of the matchup I 
want.

Group by teams
each training sample should represent the objective (name, features, result)

"""
