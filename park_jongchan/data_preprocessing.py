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
a = (df.loc[df['key'] == 0])
a = a.drop(['key'], axis =1)



data = []
buffer = []
recent_idx = 0
for idx in range(len(df['Current'])):
    if idx % 10000 == 0:
        print(idx, " is Done")
    if idx == 0:
        recent_idx = df.loc[df['key'] == idx].index
        input = (df.loc[df['key'] == idx].drop(['key'], axis =1))
        buffer.append(input)
        continue


    if recent_idx + 1 != df.loc[df['key'] == idx].index:
        buffer = []
    
    input = (df.loc[df['key'] == idx].drop(['key'], axis =1))
    buffer.append(input)

    recent_idx = df.loc[df['key'] == idx].index
    if(len(buffer) == SEQUENCE_SIZE):
        data.append(np.reshape(buffer,(-1,)).tolist())
        buffer = buffer[1:]




