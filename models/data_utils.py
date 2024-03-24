import os.path

from rdkit.Chem import AllChem

from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, auc
import random
import pandas as pd

from rdkit import Chem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from rdkit.Chem import rdFingerprintGenerator
def saltremover(i):
    l = i.split('.')
    d = {len(a):a for a in l }
    smile = d[max(d.keys())]
    return smile
def stand_smiles(smiles):
    try:
        smiles = AllChem.MolToSmiles(AllChem.MolFromSmiles(smiles))
    except:
        smiles = ''
    return smiles
def process(file,content,cpu=10,sep=','):

    if content == 'cano':
        data = pd.read_csv(file,sep=sep)
        start =len(data)
        data['Smiles'] = data['Smiles'].apply(saltremover)
        data['Smiles'] = data['Smiles'].apply(stand_smiles)
        output = file.split('.csv')[0] + '_pro.csv'
        if os.path.exists(output):
            pass
        else:
            data.to_csv(output,index=False)
            print('we meet some smiles which cannot revert to cano_smiles and the number is',start-len(data))
    elif content == 'descriptor':
        from padelpy import padeldescriptor
        output = file.split('.csv')[0] + '_pro.csv'
        data = pd.read_csv(output)
        data['activity'] = data['Smiles']
        data=data[['Smiles','activity']]
        smi = file.split('.csv')[0] +'.smi'
        des = file.split('.csv')[0] +'_23d.csv'
        if os.path.exists(des):
            pass
        else:
            data.to_csv(smi,index=False,sep='\t',header=None)
            padeldescriptor(mol_dir=smi, d_2d=True, d_3d=True, d_file=des,threads=50)
            # print('done 2d3d',end=' ')
    elif content == 'pubchem':
        from padelpy import padeldescriptor
        output = file.split('.csv')[0] + '_pro.csv'
        data = pd.read_csv(output,sep=sep)
        data['activity'] = data['Smiles']
        data=data[['Smiles','activity']]
        smi = file.split('.csv')[0] +'.smi'
        des = file.split('.csv')[0] +'_pubchem.csv'
        if os.path.exists(des):
            pass
        else:
            data.to_csv(smi,index=False,sep='\t',header=None)
            padeldescriptor(mol_dir=smi, fingerprints=True, d_file=des,threads=30)
            # print('done punchem',end=' ')
    if content == 'adj_23d':
        des = file.split('.csv')[0] + '_23d.csv'
        name = file.split('.csv')[0] + '_23d_adj.csv'
        if os.path.exists(name):
            pass
        else:
            des = pd.read_csv(des).iloc[:,:1445]   # save pre 1444 features
            des = des.replace([np.inf,-np.inf],np.nan)  # trans inf to nan
            des = des.dropna(thresh=int(des.shape[1]/2),axis =0)  #drop that row all 0 how ='all'
            des = des.dropna(thresh=int(des.shape[0]/2),axis =1)
            des = des.fillna(0)  # fill nan by mean col
            min_max_scaler = preprocessing.MinMaxScaler()
            adj = min_max_scaler.fit_transform(des.drop(['Name'], axis=1))  # scaler
            adj = pd.DataFrame(adj,columns=list(des.columns)[1:])
            adj['Name']=des['Name']
            adj.to_csv(name,index=False)
            # print('done scaler',end =' ')
def start(file,des= None):
    content = ['cano']
    if '2d-3d' in des:
        content.extend(['descriptor','adj_23d'])
    if 'pubchem' in des:
        content.append('pubchem')
    for pro in content:
        process(file,pro)
def statistical(y_true, y_pred, y_pro):
    c_mat = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = list(c_mat.flatten())
    se = tp / (tp + fn)
    sp = tn / (tn + fp)
    precision = tp / (tp + fp)
    acc = (tp + tn) / (tn + fp + fn + tp)
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8)
    auc_prc = auc(precision_recall_curve(y_true, y_pro, pos_label=1)[1],
                  precision_recall_curve(y_true, y_pro, pos_label=1)[0])
    auc_roc = roc_auc_score(y_true, y_pro)
    return precision, se, sp, acc, mcc, auc_prc, auc_roc




class TVT(object):
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y

        self.column = pd.Series(data=self._index(len(X)),index=self.X.index)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        X = self.X[idx]
        Y = self.Y[idx]
        return X, Y
    def _index(self,number):
        span = number//10 +1
        end = number - 9*span
        column = []
        for num in range(10):
            if num != 9:
                column.extend([num]*span)
            else:
                column.extend([num] * end)
        random.seed(42)
        random.shuffle(column)
        return column
    def set2ten(self,num):
        assert 0<=num<=9,'num [0,9]'
        test_X = self.X.loc[self.column[self.column==num].index,]
        test_Y = self.Y.loc[self.column[self.column == num].index,]
        rest_X = self.X.loc[self.column[self.column != num].index,]
        rest_Y = self.Y.loc[self.column[self.column != num].index,]
        # in this .py ,we don't add a check to confirm if all one class in r/test_Y
        return rest_X, test_X, rest_Y,test_Y

class ScaffoldSplitter():


    def _generate_scaffolds(self,dataset):
        scaffolds = {}
        for i,ind in enumerate(list(dataset.index)):
            smiles = dataset.iloc[i]
            scaffold = self._generate_scaffold(smiles)
            if scaffold not in scaffolds:
                scaffolds[scaffold] = [ind]
            else:
                scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
        scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
        scaffold_sets = [
                scaffold_set for (scaffold, scaffold_set) in sorted(
                scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
            ]
        return scaffold_sets

    def _generate_scaffold(self, smiles):

        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffoldSmiles(mol=mol)
        return scaffold

    def train_test_split(self,X,Y,frac_train=0.8,valid=False):
        frac_valid=(1-frac_train)/2 if valid else (1-frac_train)
        scaffold_sets = self._generate_scaffolds(X)
        train_cutoff = frac_train * len(X)
        valid_cutoff = (frac_train + frac_valid) * len(X)
        train_inds,valid_inds,test_inds = [],[],[]
        for scaffold_set in scaffold_sets:
            if len(train_inds) + len(scaffold_set) > train_cutoff:
                if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                    test_inds += scaffold_set
                else:
                    valid_inds += scaffold_set
            else:
                train_inds += scaffold_set
        if valid:
            X_train,Y_train = X.loc[train_inds,],Y.loc[train_inds,]
            X_valid,Y_valid =X.loc[valid_inds,],Y.loc[valid_inds,]
            X_test,Y_test=X.loc[test_inds,],Y.loc[test_inds,]
            return X_train,X_valid,X_test,Y_train,Y_valid,Y_test
        else:
            X_train, Y_train = X.loc[train_inds,], Y.loc[train_inds,]
            X_valid, Y_valid = X.loc[valid_inds,], Y.loc[valid_inds,]
            return X_train, X_valid, Y_train, Y_valid

class ClusterSplitter():

    def _tanimoto_distance_matrix(self,fp_list):
        """Calculate distance matrix for fingerprint list"""
        dissimilarity_matrix = []

        for i in range(1, len(fp_list)):
            # Compare the current fingerprint against all the previous ones in the list
            similarities = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
            # Since we need a distance matrix, calculate 1-x for every element in similarity matrix
            dissimilarity_matrix.extend([1 - x for x in similarities])
        return dissimilarity_matrix

    def _cluster_fingerprints(self,fingerprints, cutoff=0.2):
        distance_matrix = self._tanimoto_distance_matrix(fingerprints)
        clusters = Butina.ClusterData(distance_matrix, len(fingerprints), cutoff, isDistData=True)
        clusters = sorted(clusters, key=len, reverse=True)
        return clusters
    def _generate_cluster(self, smiles):

        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffoldSmiles(mol=mol)
        return scaffold
    def _generate_clusters(self,dataset,train_size=0.8):
        fingerprints=[]
        rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=5)
        for i,ind in enumerate(list(dataset.index)):
            smiles = dataset.iloc[i]
            fingerprints.append(rdkit_gen.GetFingerprint(Chem.MolFromSmiles(smiles)))
        clusters = self._cluster_fingerprints(fingerprints, cutoff=0.2)
        cluster_centers = [c[0] for c in clusters]
        sorted_clusters = []
        Singletons = []
        for cluster in clusters:
            if len(cluster) <= 1:
                Singletons.append(cluster)
                continue  # Singletons
            sorted_fingerprints = [rdkit_gen.GetFingerprint(Chem.MolFromSmiles(dataset.loc[i,])) for i in cluster]
            similarities = DataStructs.BulkTanimotoSimilarity(
                sorted_fingerprints[0], sorted_fingerprints[1:]
            )
            similarities = list(zip(similarities, cluster[1:]))
            similarities.sort(reverse=True)
            sorted_clusters.append((len(similarities), [i for _, i in similarities]))
            sorted_clusters.sort(reverse=True)
        selected_molecules = cluster_centers.copy()
        index = 0
        pending = int(len(dataset) * train_size) - len(selected_molecules)
        while pending > 0 and index < len(sorted_clusters):
            tmp_cluster = sorted_clusters[index][1]
            if sorted_clusters[index][0] > 10:
                num_compounds = int(sorted_clusters[index][0] * train_size)
            else:
                num_compounds = len(tmp_cluster)
            if num_compounds > pending:
                num_compounds = pending
            selected_molecules += [i for i in tmp_cluster[:num_compounds]]
            index += 1
            pending = int(len(dataset) * train_size) - len(selected_molecules)
        return selected_molecules

    def train_test_split(self,X,Y,frac_train=0.8,valid=False):
        train_inds = self._generate_clusters(X,train_size=frac_train)
        if valid:
            rest_inds = list(set(X.index) - set(train_inds))
            valid_inds = rest_inds[:int(len(rest_inds)/2)]
            test_inds = rest_inds[int(len(rest_inds)/2):]
            X_train, Y_train = X.loc[train_inds,], Y.loc[train_inds,]
            X_valid, Y_valid = X.loc[valid_inds,], Y.loc[valid_inds,]
            X_test, Y_test = X.loc[test_inds,], Y.loc[test_inds,]
            return X_train, X_valid, X_test, Y_train, Y_valid, Y_test
        else:
            valid_inds = list(set(X.index)-set(train_inds))
            X_train, Y_train = X.loc[train_inds,], Y.loc[train_inds,]
            X_valid, Y_valid = X.loc[valid_inds,], Y.loc[valid_inds,]
            return X_train, X_valid, Y_train, Y_valid
