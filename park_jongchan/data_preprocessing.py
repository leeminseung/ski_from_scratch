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
df = df.drop(['REWARD'], axis = 1)

columns = df.columns
for i in columns:
    df[i] -= np.mean(df[i])
    df[i] /= max(abs(df[i].min()), abs(df[i].max()))

df.hist(bins = 50, figsize=(20,15))
plt.show()





key = np.arange(0, df['Current'].size)
df['key'] = key

df['idx'] = df.index
df = np.array(df).tolist()

data = []
buffer = []
label = []
recent_idx = 0
for idx in range(len(df) - 1):
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
    if(len(buffer) == SEQUENCE_SIZE):
        data.append(buffer)
        label.append(df[idx + 1][:-2])
        buffer = buffer[1:]
######################################################################
print(np.shape(label), np.shape(data))

np.save('./result/forged_data', data)
np.save('./result/forged_label', label)

