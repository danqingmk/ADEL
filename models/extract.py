import numpy as np
import pandas as pd
from rdkit import Chem

df = pd.read_csv('/home/weili/zyh/dataset/keti/mukuo/fangcha/screen_7/screen_result.csv')
# data = df.sort_values(by=['mean'], axis=0, ascending=[True])
data = df.sort_values(by=['std'], axis=0, ascending=[False])
data = data[data['gcnscore'] >= -15]
data = data[data['gatscore'] >= -15]

# total_mean = data['mean'][:100].mean(axis=0)
# print(total_mean)
# f = open('/home/weili/zyh/dataset/keti/score.txt','a')
# f.writelines(str(total_mean)+ '\n')
# f.close()

# dock = data.iloc[:2000]
# dock = dock[['Smiles']]
# print(dock[:10])
# print(len(dock))
# dock.to_csv('/home/weili/zyh/dock/mltest/5model20000/iter5.csv',index=False)

old = pd.read_csv('/home/weili/zyh/dataset/keti/mukuo/fangcha/train7.csv')
data = data[~data['Smiles'].isin(old['Smiles'].tolist())]
print(data[:10])
data = data.iloc[:2000]
# data = data.sort_values(by=['std'], axis=0, ascending=[False])
# data = data.iloc[:175]
data = data[['Smiles', 'dock_score']]
print(data[:10])
mer = old.append(data)
print(len(mer))
mer.to_csv('/home/weili/zyh/dataset/keti/mukuo/fangcha/train8.csv',index=False)
'''
set1 = data[data['mean'] <= -8]
set1_sample = set1.sample(n=round(2000*len(set1)/len(data)))
print(len(set1_sample))
set2 = data[data['mean'].map(lambda x:(x<=-7)&(x>=-8))]
set2_sample = set2.sample(n=round(2000*len(set2)/len(data)))
print(len(set2_sample))
set3 = data[data['mean'].map(lambda x:(x<=-6)&(x>=-7))]
set3_sample = set3.sample(n=round(2000*len(set3)/len(data)))
print(len(set3_sample))
set4 = data[data['mean'].map(lambda x:(x<=-5)&(x>=-6))]
set4_sample = set4.sample(n=round(2000*len(set4)/len(data)))
print(len(set4_sample))
set5 = data[data['mean'].map(lambda x:(x<=-4)&(x>=-5))]
set5_sample = set5.sample(n=round(2000*len(set5)/len(data)))
print(len(set5_sample))
set6 = data[data['mean'] >= -4]
set6_sample = set6.sample(n=round(2000*len(set6)/len(data)))
print(len(set6_sample))
ss = pd.concat([set1_sample,set2_sample,set3_sample,set4_sample,set5_sample,set6_sample])
print(len(ss))
ss = ss[['Smiles', 'dock_score']]
mer = old.append(ss)
print(len(mer))
mer.to_csv('/home/weili/zyh/dataset/keti/trainset/2000new/train8.csv',index=False)
'''