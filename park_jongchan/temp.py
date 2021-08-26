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
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SEQUENCE_SIZE = 32
INPUT_SIZE = 6
OUTPUT_SIZE = 1 
HIDDEN_SIZE = 256
EPOCHS = 100
ENSEMBLE_SIZE = 5
BATCH_SIZE = 10000
STOP_PATIENCE = 10
path = "DataSet_ARO1_LHT_LC010.CSV"
csv = pd.read_csv(path, encoding = 'utf-8')



df = pd.DataFrame(csv)
df = df[2:1048559]
df.columns = ['Time', 'ENV1', 'ENV2', 'ENV3', 'ENV4', 'Target', 'Current', 'AGENT', 'REWARD']
df = df.drop(['Time'], axis = 1)
df = df.astype(float)

df_for_forty =  pd.DataFrame((df.loc[df['Target'] == 40]))


df.drop(df.loc[df['Target'] == 40].index, inplace = True)


q3 = df['ENV1'].quantile(q = 0.98)
df.drop(df.loc[df['ENV1'] > q3].index, inplace = True)


q1 = df['REWARD'].quantile(q = 0.001)
q3 = df['REWARD'].quantile(q = 0.99)
df.drop(df.loc[df['REWARD'] < q1].index, inplace = True)
df.drop(df.loc[df['REWARD'] > q3].index, inplace = True)

q1 = df['AGENT'].quantile(q = 0.001)
q3 = df['AGENT'].quantile(q = 0.995)
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
df = df.drop(['REWARD'], axis = 1)

###############################
columns = df.columns    
for i in columns:
    i_mean= np.mean(df[i])
    i_std = np.std(df[i])
    df[i] -= i_mean
    df[i] /= i_std
################################

key = np.arange(0, df['Current'].size)
df['key'] = key
df['idx'] = df.index

df = np.array(df).tolist()


data = []
buffer = []
label = []
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
        data.append(np.reshape(buffer[: -1],(-1,)))
        label.append(buffer[-1][-4]) # only current
        buffer = buffer[1:]
######################################################################

np.save('./result/forged_data', data)
np.save('./result/forged_label', label)




####################################################### load data
data = np.load('./result/forged_data.npy')
label = np.load('./result/forged_label.npy')
#######################################################
class TransitionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(INPUT_SIZE * SEQUENCE_SIZE , HIDDEN_SIZE)
        self.hidden2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.hidden3 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.hidden4 = nn.Linear(HIDDEN_SIZE , HIDDEN_SIZE)
        self.hidden5 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

        self.Normed = nn.LayerNorm(OUTPUT_SIZE)

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
        return self.Normed(layer5)
#######################################################
class EnsembleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.models = [TransitionModel() for _ in range(ENSEMBLE_SIZE)]
    
    def prediction(self,  x): 
        predictions = list(model.forward(x) for model in (self.models))
        prediction = random.choice(predictions)
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
label = label.reshape(-1,1)

######################################################## split
train_data, val_data, train_label, val_label = train_test_split(data,label, test_size = 0.1, shuffle = True)

train_dataset = TensorDataset(train_data, train_label)
val_dataset = TensorDataset(val_data, val_label)

train_data = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle= True)
val_data = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle= True)
########################################################
def train(model, optimizer, device):
    train_loss = 0
    
    patience_stack = 0
    patience = 999999999
    ckpt_model = model

    train_loss_list = []
    val_loss_list = []
######################################################## train's main part
    for e in range(EPOCHS):
        epoch_loss = 0

        for batch_idx, train_input in enumerate(train_data):
            data, label = train_input #load train data
            
            self_prediction = model.prediction(data)  # prediction
            loss = F.mse_loss(self_prediction, label).to(device)
            train_loss += loss.item()
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        train_loss_epoch = epoch_loss / len(train_data)
        print("train_mse for epoch ", e , " : ", train_loss_epoch)
        train_loss_list.append(train_loss_epoch)


########################################################


        val_loss = 0

        for batch_idx, val_input in enumerate(val_data):
            v_data, v_label = val_input #load train data
            
            self_prediction = model.prediction(v_data)  # prediction
            loss = F.mse_loss(self_prediction, v_label).to(device) 
            val_loss += loss.item()
            
        val_loss_epoch = val_loss / len(val_data)
        print("val_mse for epoch ", e , " : ", val_loss_epoch)
        val_loss_list.append(val_loss_epoch)
        if val_loss_epoch >= patience:
            patience_stack += 1
            if patience_stack == STOP_PATIENCE:
                break
        else:
            patience_stack = 0
            patience = val_loss_epoch
            ckpt_model = model
    
    return train_loss_list, val_loss_list, ckpt_model

def main():
    model = EnsembleModel().to('cuda:0')
    param = list(model.parameters())

    optimizer = optim.Adam(param, lr = 0.0005, weight_decay= 0.0001)
    
    train_loss , val_loss,  model_ = train(model, optimizer, 'cuda:0')

    Time = time.strftime('%Y_%m_%d', time.localtime(time.time()))
    torch.save(model_, './result/' + Time + '_model.pt')

    
    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot()
    ax.plot( train_loss, 'r')
    ax.plot( val_loss, 'g')
    plt.xlabel("EPOCHS")
    plt.ylabel("LOSS")
    plt.savefig('./result/' + Time + '_graph.png')
    
    
main()
