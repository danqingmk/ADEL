import os
import shutil
import pandas as pd
import time
from functools import reduce

s = time.time()
scr_path = "/home/weili/zyh/library/24_211/result"
to_dir_path = "/home/weili/zyh/library/24_211/guodu"
key = '_{}w'
states = 306
for state in range(states):
    state = (state + 211) * 100
    key1 = key.format(state)
    for file in os.listdir(scr_path):
        if os.path.isfile(scr_path+'/'+file):
            if key1 in file:
                shutil.copy(scr_path+'/'+file,to_dir_path+'/'+file)

    filenames = os.listdir(to_dir_path)
    print(filenames)
    screen_result = []
    for file in filenames:
        data = pd.read_csv(to_dir_path+'/'+file,index_col=False)
        data = data.sort_values(by=['Smiles'], axis=0, ascending=[True])
        data = data.reset_index(drop=True)
        screen_result.append(data)
    result = reduce(lambda left,right:pd.merge(left,right,on=['Smiles']),screen_result)
    # result = pd.concat(screen_result,axis=1)
    # result = result.T.drop_duplicates(keep='first').T
    result = result.loc[:,~result.columns.duplicated()]
    df = result[['Smiles','gcnscore','SVMscore','XGBscore','gatscore']]
    print(df.columns)
    df['mean'] = df[['gcnscore','SVMscore','XGBscore','gatscore']].mean(axis=1)
    df['std'] = df[['gcnscore','SVMscore','XGBscore','gatscore']].std(axis=1)
    df.to_csv('/home/weili/zyh/library/24_211/screen_result/Enamine_REAL_HAC_24_CXSMILES{}_screen_result.csv'
              .format(key1),index=False)
    for i in os.listdir(to_dir_path):
        os.remove(os.path.join(to_dir_path, i))
    print(len(df), time.time() - s)


