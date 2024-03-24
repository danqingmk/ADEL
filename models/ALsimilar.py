import pandas as pd

df = pd.read_csv('/home/weili/zyh/dataset/keti/B2AR2.csv')
df = df.sort_values(by=['dock_score'], axis=0, ascending=[True])
ss1 = df.iloc[:1000]
total = (df['label'] == 1).sum()
print('total active:',total)
df = df.iloc[:585]
dfact = (df['label'] == 1).sum()
print('dock top 600 active:',dfact)
smi = df[df['label'] == 1]

data = pd.read_csv('/home/weili/zyh/dataset/keti/B2AR2/single/14-2/screen_1/screen_result.csv')
data = data.sort_values(by=['mean'], axis=0, ascending=[True])
data = data[data['mean'] >= -15]
ss2 = data.iloc[:1000]
# data = data.drop_duplicates(subset=['Smiles'],keep='first')
data = data.iloc[:585]
dataact = (data['label'] == 1).sum()
print('AL top 600 active:', dataact)
ALsmi = data[data['label'] == 1]

actsame = set(smi['Smiles']) & set(ALsmi['Smiles'])
print('same active mol number:',len(actsame))
tolsame = set(ss1['Smiles']) & set(ss2['Smiles'])
print('same mol number:',len(tolsame))