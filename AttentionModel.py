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
    in_feats = 23
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
        return self.raw_gamedata_projection_layer(in_feats)


class BasicAttentionLayer(nn.Module):
    """
    This class holds the layer that will be performing the attention calculations on our latent-space game data vectors
    that we will get as output from out BasicProjectionLayer.
    """

    def __init__(self, layer_dimensions, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        """
        Initialize the attention layer.
        :param layer_dimensions: This is equal to the number of features outputted by the projection layer. This
        SHOULD stay the same throughout this layer. Currently, this value is: X
        """
        super(BasicAttentionLayer, self).__init__()
        self.layer_dims = layer_dimensions
        self.query = query
        self.key = key
        self.value = value

        # Layer weights for Q, K, and V
        # self.WQ = nn.Linear(self.layer_dims, self.layer_dims)
        # self.WK = nn.Linear(self.layer_dims, self.layer_dims)
        # self.WV = nn.Linear(self.layer_dims, self.layer_dims)

        # Final softmax layer
        self.softmax = nn.Softmax()

    def attention(self):
        """
        This method performs the attention calculations with the given query, key, and values. This is an
        implementation of scaled dot-product attention.
        """
        # Calculate the value to scale query dot key_transpose by
        scale_value = math.sqrt(self.layer_dims)

        # Transpose key
        key_transpose = torch.transpose(self.key, -2, -1)

        # Calculate the query dot key score
        scores = torch.matmul(self.query, key_transpose) / scale_value

        # Send the score through a softmax
        softmaxed_score = self.softmax(scores)

        # Return the attention matrix along with the softmax scores for the matrix
        return torch.matmul(softmaxed_score, self.value), softmaxed_score

    def forward(self):
        """
        The forward function that passes the input query, key, and value through the attention function
        :return:
        """
        return self.attention()


class LucasModel(nn.Module):
    """
    This class is what combines all the lower-level layers into one big model.
    """

    def __init__(self, in_feats_from_gamedata, out_feats_from_gamedata, input_data):
        super(LucasModel, self).__init__()
        # General Model Parameters Below --------------------------------------------

        # Linear Projection Layer
        self.in_feats_from_gamedata = in_feats_from_gamedata
        self.out_feats_from_gamedata = out_feats_from_gamedata
        self.input_data = input_data
        self.projection = BasicProjectionLayer(self.in_feats_from_gamedata, self.out_feats_from_gamedata)

        # Attention Layer
        self.layer_dimensions = out_feats_from_gamedata
        self.attention = BasicAttentionLayer(self.layer_dimensions, self.query, self.key, self.value)

        # Decoder/Prediction Layer

    def forward(self):
        projections = self.projection(self.input_data)  # Projecting the data into a latent space

        attentionScores, softmaxVals = self.attention()
