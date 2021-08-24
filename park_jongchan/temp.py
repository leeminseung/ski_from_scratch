import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

INPUT_SIZE = 7
OUTPUT_SIZE = 7
HIDDEN_SIZE = 64
EPOCHS = 300
ENSEMBLE_SIZE = 5
BATCH_SIZE = 32

path = "DataSet_ARO1_LHT_LC010.CSV"
csv = pd.read_csv(path, encoding = 'utf-8')

SEQUENCE_SIZE = 20


####################################################### load data
data = np.load('./result/forged_data.npy')
label = np.load('./result/forged_label.npy')
#######################################################
class TransitionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(INPUT_SIZE * SEQUENCE_SIZE, HIDDEN_SIZE)
        self.hidden2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.hidden3 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.hidden4 = nn.Linear(HIDDEN_SIZE , HIDDEN_SIZE)
        self.hidden5 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        self.output =  nn.Sigmoid()

        self.Normed = nn.LayerNorm(HIDDEN_SIZE)

        nn.init.kaiming_uniform_(self.hidden1.weight)
        nn.init.kaiming_uniform_(self.hidden2.weight)
        nn.init.kaiming_uniform_(self.hidden3.weight)
        nn.init.kaiming_uniform_(self.hidden4.weight)
        nn.init.kaiming_uniform_(self.hidden5.weight)

    def prediction(self, x):
        prediction = self.forward(x)
        return prediction

    def forward(self, x):
        layer1 = self.hidden1(x)
        layer2 = self.hidden2(layer1)
        layer3 = self.hidden3(layer2)
        layer4 = self.hidden4(layer3)
        layer5 = self.hidden5(layer4)
        return self.output(layer5)
#######################################################
class EnsembleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.models = [TransitionModel() for _ in range(ENSEMBLE_SIZE)]
    
    def prediction(self,  x): 
        predictions = list(model.forward(x) for model in (self.models))
        prediction = (predictions[random.randint(0, 4)])
        return prediction

    def parameters(self):
        list_of_params = [list(model.parameters()) for model in self.models]
        parameters = ([p for ps in list_of_params for p in ps])
        return parameters

    def to(self, device):
        for model in self.models:
            model.to(device)
        return self

#######################################################
data = torch.Tensor(data).to('cuda:0')
label = torch.Tensor(label).to('cuda:0')

######################################################## split
train_data, val_data, train_label, val_label = train_test_split(data,label, test_size = 0.1, shuffle = True)

train_dataset = TensorDataset(train_data, train_label)
val_dataset = TensorDataset(val_data, val_label)

train_data = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle= True)
val_data = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle= True)
########################################################
def train(model, optimizer, device):
    train_loss = 0
    train_len = 0
######################################################## train's main part
    for e in range(EPOCHS):
        for batch_idx, train_input in enumerate(train_data):
            data, label = train_input #load train data
            
            self_prediction = model.prediction(data)  # prediction
            loss = F.mse_loss(self_prediction, label).to(device) 
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
########################################################
            train_len += 1

            if(train_len % 1000 == 0):
                print("Epochs: ", e ," Steps: ", train_len)
                print("train_mse: ", train_loss/ train_len)
        print("train_mse for this epochs: ", train_loss)
    train_loss /= train_len
    return train_loss, model

def main():
    model = EnsembleModel().to('cuda:0')
    param = list(model.parameters())

    optimizer = optim.Adam(param, lr = 0.005, weight_decay= 0.0001)
    
    train_loss, model_ = train(model, optimizer, 'cuda:0')

main()
