import torch
import torch.nn as nn
from sklearn.metrics import f1_score, average_precision_score, classification_report
import random
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

class external_validation():
    '''
    This class predict an unseen data loader using trained model.
    Input:
    ------
    Trained model, unseen data

    Returns:
    --------
    y_predict, probability
    '''
    def __init__(self,model, params, path, data_test,device):
        self.model = model
        self.save_dir = path
        self.data_test = data_test
        self.device = device
        self.params = params

    def predict(self):
        checkpoint = torch.load(self.save_dir)
        self.model = self.model.to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        #validation_loss = 0
        truelabels_val = []
        probas_val = []
        proba_flat_val = []
        pred_flat_val = []
        with torch.no_grad():
            for fps, labels in self.data_test:
                out_put = self.model(fps,self.params)
                #validation_loss += self.criterion(out_put[:,0], labels).cpu()
 
                a = list(np.asarray(labels.detach().cpu()))
                truelabels_val.append(a)

                probas_val.append(np.asarray(out_put.detach().cpu()))

        
        flatten_list = lambda truelabels_val:[element for item in truelabels_val for element in flatten_list(item)] if type(truelabels_val) is list else [truelabels_val]
        truelabels_val = flatten_list(truelabels_val)
        for i in probas_val:
            for j in i:
                proba_flat_val.append(j)
    
        pred_flat_val = proba_flat_val.copy()
        for key, value in enumerate(proba_flat_val):
            if value < 0.5:
                pred_flat_val[key] = 0
            else:
                pred_flat_val[key] = 1
        return pred_flat_val, truelabels_val
    
    def predict_proba(self):
        checkpoint = torch.load(self.save_dir)
        self.model = self.model.to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        #validation_loss = 0
        truelabels_val = []
        probas_val = []
        proba_flat_val = []
        with torch.no_grad():
            for fps, labels in self.data_test:
                out_put = self.model(fps, self.params)
                #validation_loss += self.criterion(out_put[:,0], labels).cpu()
 
                a = list(np.asarray(labels.detach().cpu()))
                truelabels_val.append(a)

                probas_val.append(np.asarray(out_put.detach().cpu()))

        flatten_list = lambda truelabels_val:[element for item in truelabels_val for element in flatten_list(item)] if type(truelabels_val) is list else [truelabels_val]
        truelabels_val = flatten_list(truelabels_val)
        for i in probas_val:
            for j in i:
                proba_flat_val.append(j)
    
        pred_flat_val = proba_flat_val.copy()
        for key, value in enumerate(proba_flat_val):
            if value < 0.5:
                pred_flat_val[key] = 0
            else:
                pred_flat_val[key] = 1
        return proba_flat_val,truelabels_val
