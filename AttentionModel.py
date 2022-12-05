"""
This file contains the code for Lucas's Neural Networks. These will be attention-based models, with one being a very
simple scaled-dot product attention, and the other being a custom attention implementation.
"""

from evalMetric import *
import torch
import torch.nn as nn


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
        return self.raw_gamedata_projection_layer(in_feats)

class BasicAttentionLayer(nn.Module):
    """
    This class holds the layer that will be performing the attention calculations on our latent-space game data vectors
    that we will get as output from out BasicProjectionLayer.
    """

    def __init__(self, ):
