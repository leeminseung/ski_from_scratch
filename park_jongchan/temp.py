import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
path = "DataSet_ARO1_LHT_LC010.CSV"
csv = pd.read_csv(path, encoding = 'utf-8')

SEQUENCE_SIZE = 20



df = pd.DataFrame(csv)
df = df[2:1048559]
df.columns = ['Time', 'ENV1', 'ENV2', 'ENV3', 'ENV4', 'Target', 'Current', 'AGENT', 'REWARD']
df = df.drop(['Time'], axis = 1)
df = df.astype(float)

df_for_forty =  pd.DataFrame((df.loc[df['Target'] == 40]))


df.drop(df.loc[df['Target'] == 40].index, inplace = True)


q1 = df['REWARD'].quantile(q = 0.001)
q3 = df['REWARD'].quantile(q = 0.99)
df.drop(df.loc[df['REWARD'] < q1].index, inplace = True)
df.drop(df.loc[df['REWARD'] > q3].index, inplace = True)

q1 = df['AGENT'].quantile(q = 0.001)
q3 = df['AGENT'].quantile(q = 0.99)
df.drop(df.loc[df['AGENT'] < q1].index, inplace = True)
df.drop(df.loc[df['AGENT'] > q3].index, inplace = True)

q1 = df['Current'].quantile(q = 0.001)
q3 = df['Current'].quantile(q = 0.99)
df.drop(df.loc[df['Current'] < q1].index, inplace = True)
df.drop(df.loc[df['Current'] > q3].index, inplace = True)
##############################
corr_mat = df.corr()
print(corr_mat['REWARD'].sort_values(ascending = False))
corr_mat_for_forty = df.corr()
print(corr_mat_for_forty['REWARD'].sort_values(ascending = False))
df['REWARD'].plot()
plt.show()
##############################
df = df.drop(['Target'], axis = 1)

key = np.arange(0, df['Current'].size)
df['key'] = key

df['idx'] = df.index
df = np.array(df).tolist()

data = []
buffer = []
recent_idx = 0
for idx in range(len(df)):
    if idx % 10000 == 0:
        print(idx, " is Done")
    if idx == 0:
        recent_idx = df[idx][-1]
        input = df[idx][:-2]
        buffer.append(input)
        continue


    if recent_idx + 1 != df[idx][-1]:
        buffer = []
    
    input = df[idx][:-2]
    buffer.append(input)

    recent_idx = df[idx][-1]
    if(len(buffer) == SEQUENCE_SIZE + 1):
        data.append(np.reshape(buffer,(-1,)).tolist())
        buffer = buffer[1:]
######################################################################

np.save('./result/forged_data', data)

data = np.load('./result/forged_data.npy')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from sklearn.model_selection import train_test_split

INPUT_SIZE = 7
OUTPUT_SIZE = 7
HIDDEN_SIZE = 64
EPOCHS = 300
ENSEMBLE_SIZE = 5

class TransitionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(INPUT_SIZE * SEQUENCE_SIZE, HIDDEN_SIZE)
        self.hidden2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE // 2)
        self.hidden3 = nn.Linear(HIDDEN_SIZE // 2, OUTPUT_SIZE)
        self.output =  nn.Sigmoid()

        self.Normed = nn.LayerNorm(HIDDEN_SIZE)

        nn.init.kaiming_uniform_(self.hidden1.weight)
        nn.init.kaiming_uniform_(self.hidden2.weight)
        nn.init.kaiming_uniform_(self.hidden3.weight)

    def prediction(self, x):
        prediction = self.forward(x)
        return prediction

    def forward(self, x):
        layer1 = self.hidden1(x)
        layer2 = self.hidden2(layer1)
        layer3 = self.hidden3(layer2)
        return self.output(layer3)


class EnsembleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.models = [TransitionModel() for _ in range(ENSEMBLE_SIZE)]
    
    def prediction(self,  x): 
        predictions = list(model.forward(x) for model in (self.models))
        sum_of_predict = np.zeros(len(predictions[0]))
        for i in range (len(predictions[0])):
            for e in range (ENSEMBLE_SIZE):
                sum_of_predict[i] += float(predictions[e][i])
        avg_prediction = (sum_of_predict / ENSEMBLE_SIZE)
        return avg_prediction

    def parameters(self):
        list_of_params = [list(model.parameters()) for model in self.models]
        parameters = ([p for ps in list_of_params for p in ps])
        return parameters

    def to(self, device):
        for model in self.models:
            model.to(device)
        return self

data = torch.Tensor(data).to('cuda:0')
train_data, val_data = train_test_split(data, test_size = 0.1, shuffle = True)

def train(model, optimizer, device):
    train_loss = 0
    train_len = 0
    for e in range(EPOCHS):
        for train_input in (train_data):
            if(train_len % 1000 == 0):
                print("Epochs: ", e, train_len)
            self_prediction = torch.Tensor(model.prediction(train_input[:-INPUT_SIZE])).to(device)
            loss = torch.mean(pow(self_prediction - (train_input[-INPUT_SIZE]), 2)).to(device)
            loss.requires_grad = True
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_len += 1
        train_loss /= train_len
        print("train_mse: ", train_loss)
    
    return train_loss, model

def main():
    model = EnsembleModel().to('cuda:0')
    param = list(model.parameters())

    optimizer = optim.Adam(param, weight_decay= 0.00001)
    
    train_loss, model_ = train(model, optimizer, 'cuda:0')

main()
