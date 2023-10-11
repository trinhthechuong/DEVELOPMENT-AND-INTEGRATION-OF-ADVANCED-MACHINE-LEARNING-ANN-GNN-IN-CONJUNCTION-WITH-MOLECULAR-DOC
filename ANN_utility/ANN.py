import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from torch.utils.data import TensorDataset
import pandas as pd
import numpy as np

class Net(torch.nn.Module):
    def __init__(self,params):
        super(Net,self).__init__()
        #n_layers
        self.n_layers_mlp = params["n_layers_mlp"]
        # dense_neurons
        self.dense_neurons = params["dense_neurons"]

        self.mlp_layers = torch.nn.ModuleList([])
        for i in range(self.n_layers_mlp):
            if i == 0:
                intput = params["in_features"]
                output = self.dense_neurons
            elif i != self.n_layers_mlp - 1:
                intput =  int(self.dense_neurons/(2**(i-1)))
                output = int(self.dense_neurons/(2**(i)))
            else:
                intput = int(self.dense_neurons/(2**(i-1)))
                output = 1
            self.mlp_layers.append(Linear(intput, output))

    def forward(self,x,params):
        rate = params["rate"]
        for i in range(self.n_layers_mlp):
            if i != self.n_layers_mlp-1:
                x = torch.relu(self.mlp_layers[i](x))
                x = F.dropout(x, p=rate, training=self.training)
            else:
                x =  torch.sigmoid(self.mlp_layers[i](x))
        
        return x   