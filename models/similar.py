import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

suppl = Chem.SDMolSupplier('/home/weili/zyh/dock/mltest/5model20000/iter5.sdf')
# suppl = Chem.SDMolSupplier('/home/jianping/zyh/data_f5.sdf')
cpd_list=[]
for i in range(len(suppl)):
    try:
        smi = Chem.MolToSmiles(suppl[i])
        temp_dict=suppl[i].GetPropsAsDict()
        temp_dict['SMILES']=smi
        cpd_list.append(temp_dict)
    except Exception as e:
        print(e)
        continue
df = pd.DataFrame(cpd_list)
print(len(df))
# df.to_csv('/home/jianping/lxy/efficient-compound/2_knime.csv')
# data =df.drop_duplicates(subset=['SMILES'], keep='first')
# print(len(data))
print(df.columns)
data = df[['SMILES','r_i_docking_score']]
data.columns = ['Smiles','dock_score']

# data = pd.read_csv('/home/weili/zyh/dataset/keti/mltest/2model8000/iter2.csv')
data = data.iloc[:1000]
data = data[['Smiles']]

datas = pd.read_csv('/home/weili/zyh/dataset/keti/database.csv')
df = datas.sort_values(by=['dock_score'], axis=0, ascending=[True])
df = df.iloc[:1000]
df = df[['Smiles']]

same = set(data['Smiles']) & set(df['Smiles'])
print(len(same))
