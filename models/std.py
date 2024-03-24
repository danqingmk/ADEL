import pandas as pd
import os
import numpy as np
import time

s = time.time()
path = "/home/weili/zyh/dataset/keti/mukuo/B2AR2/fangcha/screen_8"
filenames = os.listdir(path)
print(filenames)
screen_result = []
for file in filenames:
    data = pd.read_csv(path+'/'+file,index_col=False)
    data = data.sort_values(by=['Smiles'], axis=0, ascending=[True])
    data = data.reset_index(drop=True)
    # data = data.drop(['index'],axis=1)
    screen_result.append(data)
print('concat')
result = pd.concat(screen_result,axis=1)
print(result[:10])
result = result.T.drop_duplicates(keep='first').T
print(result.columns)

df = result[['Smiles','dock_score','label','gcnscore','SVMscore','XGBscore','gatscore','LGBMscore','RFscore','DNNscore','Ridgescore']]
print(df[:10])
df['mean'] = df[['gcnscore','SVMscore','XGBscore','gatscore','LGBMscore','RFscore','DNNscore','Ridgescore']].mean(axis=1)
df['std'] = df[['gcnscore','SVMscore','XGBscore','gatscore','LGBMscore','RFscore','DNNscore','Ridgescore']].std(axis=1)
df = df[['Smiles','dock_score','mean','label']]
df.to_csv('/home/weili/zyh/dataset/keti/mukuo/B2AR2/fangcha/screen_8/screen_result.csv',index=False)
'''
df = result[['Smiles','dock_score','gcnscore','SVMscore','XGBscore','gatscore','LGBMscore','RFscore','DNNscore','Ridgescore']]
print(df.columns)
df['mean'] = df[['gcnscore','SVMscore','XGBscore','gatscore','LGBMscore','RFscore','DNNscore','Ridgescore']].mean(axis=1)
df['std'] = df[['gcnscore','SVMscore','XGBscore','gatscore','LGBMscore','RFscore','DNNscore','Ridgescore']].std(axis=1)
df.to_csv('/home/weili/zyh/dataset/keti/mukuo/fangcha/screen_7/screen_result.csv',index=False)
'''
print(len(df), time.time() - s)

