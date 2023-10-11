import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
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

class Data_preprocess():
    """
    This class converts numerical activity values to binary values and splits the original dataset into training, validation, and test tensor d   datasets. Additionally, it creates a data loader for each of the split datasets.

    Input:
    ------
    file csv, activity threshold, activiy column's name, device (GPU or CPU), batch size

    Returns:
    --------
    Tensor splitted dataset, data loader
    """
    def __init__(self, path, y_name,threshold, col_drop, device, batch_size):
        self.path = path
        self.data = pd.read_csv(self.path)
        self.data = self.data.drop(col_drop,axis=1)
        self.Y_name = y_name
        self.device = device
        self.batch_size = batch_size
        self.threshold = threshold

    def target_bin(self,data):
        #Converting numerical values to binary values
        t1 = data[self.Y_name] < self.threshold 
        data.loc[t1, self.Y_name] = 0
        t2 = data[self.Y_name] >= self.threshold 
        data.loc[t2, self.Y_name] = 1
        data[self.Y_name] = data[self.Y_name].astype('int64')
        return data

    def create_dataloader(self,data):
        X = data.drop(data.columns[0],axis = 1)
        y = data[self.Y_name]


        #Split Data train, Data_test, Data_validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                                        random_state =42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1,
                                                                        random_state =42, stratify=y_train)


        # change numpy to tensor gpu
        self.X_train = torch.tensor(X_train.values , device= self.device).float()
        self.X_val = torch.tensor(X_val.values , device= self.device).float()
        self.X_test = torch.tensor(X_test.values, device= self.device).float()
        
        self.y_train = torch.tensor(y_train.values , device=self.device).float()
        self.y_val = torch.tensor(y_val.values, device=self.device).float()
        self.y_test = torch.tensor(y_test.values, device=self.device).float()
        
        # convert into dataloader
        self.train_dataset = TensorDataset(self.X_train, self.y_train)
        self.val_dataset = TensorDataset(self.X_val, self.y_val)
        self.test_dataset = TensorDataset(self.X_test, self.y_test)
        
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(dataset=self.val_dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=False)


    def fit(self):
        self.data_bin = self.target_bin(self.data)
        self.create_dataloader(self.data_bin)
    









