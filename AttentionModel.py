"""
This file contains the code for Lucas's Neural Networks. These will be attention-based models, with one being a very
simple scaled-dot product attention, and the other being a custom attention implementation.
"""

from evalMetric import *
import torch



def attentiontest(train, test):
