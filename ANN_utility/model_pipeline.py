from sklearn.metrics import f1_score, average_precision_score, classification_report
import random 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import RepeatedStratifiedKFold
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


class model_pipeline():
    '''
    This class contains a training pipeline including:training, evaluating and cross validation.
    Input:
    ------
    model architecture, data loader, optimizer, loss function, parameters for training model

    Returns:
    --------
    Trained model and training curve
    '''
    def __init__(self,model,params, criterion, optimizer,epochs,train_loader=None, valid_loader=None, seed=42, device=torch.device("cpu"),
                 save_dir=".", show_progress = True):
        self.device = device
        self.params = params
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.seed  = seed 
        self.save_dir = save_dir
        self.epochs = epochs
        self.show_progress = show_progress

    def seed_everything(self):
        #Ensuring reproducible of training
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def train(self):
        #Training
        self.model.train()
        trainning_loss = 0
        truelabels = []
        probas = []
        proba_flat = []
        pred_flat = []
        for fps, labels in self.train_loader:
            #Optimizer
            self.optimizer.zero_grad()
            output = self.model(fps, self.params)
            
            #Loss function
            loss = self.criterion(output[:,0], labels) 
            loss.backward()
            self.optimizer.step()
            trainning_loss += loss.item()

            probas.append(np.asarray(output.detach().cpu()))
            a = list(np.asarray(labels.detach().cpu()))
            truelabels.append(a)

        
        #Flatten prediction array
        for i in probas:
            for j in i:
                proba_flat.append(j)
    
        pred_flat = proba_flat.copy()
        for key, value in enumerate(proba_flat):
            if value < 0.5:
                pred_flat[key] = 0
            else:
                pred_flat[key] = 1
        flatten_list = lambda truelabels:[element for item in truelabels for element in flatten_list(item)] if type(truelabels) is list else [truelabels]
        truelabels = flatten_list(truelabels) 

        #Calculating loss, f1, average precision for training set
        loss = trainning_loss/len(self.train_loader)
        f1 = f1_score(truelabels,pred_flat)
        ap = average_precision_score(truelabels,proba_flat)
        return loss, f1, ap
    
    def evaluate(self):
        #Evaluate model
        self.model.eval()
        validation_loss = 0
        truelabels_val = []
        probas_val = []
        proba_flat_val = []
        pred_flat_val = []
        with torch.no_grad():
            for fps, labels in self.valid_loader:
                out_put = self.model(fps, self.params)
                validation_loss += self.criterion(out_put[:,0], labels).cpu()
 
                probas_val.append(np.asarray(out_put.detach().cpu()))
                a = list(np.asarray(labels.detach().cpu()))
                truelabels_val.append(a)

        #Flatten prediction
        for i in probas_val:
            for j in i:
                proba_flat_val.append(j)
    
        pred_flat_val = proba_flat_val.copy()
        for key, value in enumerate(proba_flat_val):
            if value < 0.5:
                pred_flat_val[key] = 0
            else:
                pred_flat_val[key] = 1


        flatten_list = lambda truelabels_val:[element for item in truelabels_val for element in flatten_list(item)] if type(truelabels_val) is list else [truelabels_val]
        truelabels_val = flatten_list(truelabels_val)

        #Calculation metrics: validation loss, f1 score and average precision
        loss_val = validation_loss/len(self.valid_loader)
        f1_val = f1_score(truelabels_val,pred_flat_val)
        ap_val = average_precision_score(truelabels_val,proba_flat_val)
        return loss_val, f1_val, ap_val
        
    
    def save_model(self):
        #Saving model into *pth file in provided saving directory
        print("Saving...")
        torch.save({
                'epoch': self.epochs,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.criterion,
                }, self.save_dir)
                 
    def fit(self):
        self.seed_everything()
        self.history = {"train_loss":[], "val_loss": [],
              "train_f1":[], "val_f1":[],
              "train_ap":[], "val_ap":[]}
        for epoch in range(self.epochs):
            train1_loss, train1_f1, train1_ap = self.train()
            
            val1_loss,val1_f1, val1_ap = self.evaluate()
                 
            #self.lr_scheduler.step(train1_f1)
            self.history["train_loss"].append(train1_loss)
            self.history["val_loss"].append(val1_loss.detach().numpy())
            self.history["train_f1"].append(train1_f1)
            self.history["val_f1"].append(val1_f1)
            self.history["train_ap"].append(train1_ap)
            self.history["val_ap"].append(val1_ap)
            if self.show_progress == True:
                if (epoch+1) % 5 == 0:
                    print("Epoch: {}/{}.. ".format(epoch+1, self.epochs),
              "Training Loss: {:.3f}.. ".format(train1_loss),
              "validation Loss: {:.3f}.. ".format(val1_loss),
            "validation f1_score: {:.3f}.. ".format(val1_f1),
                  "validation average precision: {:.3f}.. ".format(val1_ap),
             )
            else: 
                pass
        
        
        if self.show_progress == True:
            self.visualize()
        else:
            pass
        self.save_model()
        print("Complete the process...")
              
   
    def visualize(self):
        sns.set()

        # create a subplot with three axes
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))


        ax1.plot(self.history["train_loss"], label='Training Loss')
        ax1.plot(self.history["val_loss"], label='Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.set_title('Training and Validation Loss over Epochs')

        ax2.plot(self.history["train_f1"], label='Training F1 Score')
        ax2.plot(self.history["val_f1"],label="Validation F1 Score")
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('F1 score')
        ax2.legend()
        ax2.set_title('F1 Score over Epochs')

        ax3.plot(self.history["train_ap"], label='Training AP Score')
        ax3.plot(self.history["val_ap"],label="Validation AP Score")
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Average precision score')
        ax3.legend()
        ax3.set_title('Average precision Score over Epochs')

        fig.suptitle('Training Metrics over Epochs')


        plt.show()

    def reset_weights(self,model):

        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    
    def cross_val_score(self, X, y,model, cv,batch_size, device,verbose = True):
        #Splitting original dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state =42, stratify=y)
        
        self.History = {"F1_record":[],"AP_record":[]}
        
        for i, (train_index, test_index) in enumerate(cv.split(X_train, y_train)):
            Xtrain = torch.tensor(X_train.iloc[train_index,:].values , device=self.device).float()
            Xtest = torch.tensor(X_train.iloc[test_index,:].values, device=self.device).float()

            ytrain = torch.tensor(y_train.iloc[train_index].values , device=self.device).float()
            ytest = torch.tensor(y_train.iloc[test_index].values, device=self.device).float()
    
            train_dataset = TensorDataset(Xtrain, ytrain)
            test_dataset = TensorDataset(Xtest, ytest)
    
            self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)
            self.valid_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
            
            self.model = model.to(self.device)
            self.model.apply(lambda m: self.reset_weights(m))
            

            for epoch in range(self.epochs):
                train1_loss,train1_f1, train1_ap = self.train()
            val1_loss,val1_f1, val1_ap = self.evaluate()

            if verbose == True:
                print("Fold: {}.. ".format(i+1),
                "validation f1_score: {:.3f}.. ".format(val1_f1),
                      "validation average precision: {:.3f}.. ".format(val1_ap),
                 )
            else:
                pass
            self.History["F1_record"].append(val1_f1)
            self.History["AP_record"].append(val1_ap)
 
        self.mean_scores_ap = sum(self.History["AP_record"]) / len(self.History["AP_record"])
        print(f"Overall AP score = {self.mean_scores_ap :.4f}")
        self.mean_scores_f1 = sum(self.History["F1_record"]) / len(self.History["F1_record"])
        print(f"Overall F1 score = {self.mean_scores_f1:.4f}")

        