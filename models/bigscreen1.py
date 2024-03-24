import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, SMILESToBigraph, RandomSplitter, EarlyStopping
from dgllife.data import MoleculeCSVDataset
from dgllife.model import GCNPredictor, GATPredictor
from gnn_utils import set_random_seed, Meter, collate_molgraphs
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import multiprocessing as mp
import time
import joblib
import torch
torch.set_num_threads(20)

def run_an_eval_epoch(model, data_loader, args):
    f = open(args['output'],'w+')
    f.write('Smiles,{}score\n'.format(args['model']))
    # print(args['output'])
    model.eval()
    # eval_metric = Meter()
    smile_list = {}
    count = 0
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            eval_metric = Meter()
            smiles, bg, labels, masks = batch_data
            smile_list[count]=smiles
            bg = bg.to(args['device'])
            atom_feats = bg.ndata.pop('h')
            bond_feats = bg.edata.pop('e')
            # transfer the data to device(cpu or cuda)
            outputs = model(bg, atom_feats)
            # smile_list.append(smiles)
            eval_metric.update(outputs, labels, torch.tensor([count]))
            roc_score = eval_metric.compute_metric('pred')[0][0]
            # if roc_score.tolist()[0][0] >= args['prop']:
            #     f.write('{},{}\n'.format(smiles[0],round(roc_score.tolist()[0][0],2)))
                # print(smiles[0],round(roc_score.tolist()[0][0],2))
            f.write('{},{}\n'.format(smiles[0], roc_score.tolist()))
            count += 1
            torch.cuda.empty_cache()
            if count % 10000 == 0:
                print(count)
        f.close()

def dlscreen(file='', sep=',', models=None,prop=0.5, smiles_col='Smiles',out_dir=None):
    # AtomFeaturizer = AttentiveFPAtomFeaturizer
    # BondFeaturizer = AttentiveFPBondFeaturizer
    # print(models)
    model_type = models.split('/')[-2]
    device = torch.device("cpu")
    args = {'model': model_type, 'device': device}
    print(model_type)
    outputs = os.path.join(out_dir,file.split('/')[-1].replace('.csv','_{}_screen.csv'.format(args['model'])))
    # print(outputs)
    if os.path.exists(outputs):
        print(outputs,'has done')
    else:
        args['output'] =outputs
        # my_df = pd.read_csv(file)
        my_df = pd.read_csv(file, sep=sep, names=['Smiles'])
        # print(my_df.columns)
        # my_dataset: MoleculeCSVDataset = csv_dataset.MoleculeCSVDataset(my_df.iloc[:, :], smiles_to_bigraph, AtomFeaturizer,
        #                                                                     BondFeaturizer, smiles_col,
        #                                                                     file.replace('.csv', '.bin'))
        smiles_to_graph = SMILESToBigraph(add_self_loop=True,
                                          node_featurizer=CanonicalAtomFeaturizer(atom_data_field='h'),
                                          edge_featurizer=CanonicalBondFeaturizer(bond_data_field='e', self_loop=True))

        my_dataset = MoleculeCSVDataset(df=my_df,
                                        smiles_to_graph=smiles_to_graph,
                                        smiles_column='Smiles',
                                        cache_file_path=file.replace('.csv', '.bin'), load=True,
                                        # task_names=args['tasks'],
                                        n_jobs=1)

        train_loader = DataLoader(my_dataset, shuffle=True, batch_size=1, collate_fn=collate_molgraphs)

        if 'gcn' in models:
            chf = models.split('_')[-2].split('.')[0]
            ghf = models.split('_')[-3]#-2
            best_model = GCNPredictor(in_feats=CanonicalAtomFeaturizer().feat_size('h'),
                                      hidden_feats=eval(ghf),
                                      classifier_hidden_feats=eval(chf),
                                      classifier_dropout=0.0,
                                      # n_tasks=len(args['tasks']),
                                      predictor_hidden_feats=128,
                                      predictor_dropout=0.0)

        elif 'gat' in models:
            chf = models.split('_')[-2].split('.')[0]
            nh = models.split('_')[-3]
            ghf = models.split('_')[-4]#-3
            best_model = GATPredictor(in_feats=CanonicalAtomFeaturizer().feat_size('h'),
                                       hidden_feats=eval(ghf),
                                       num_heads=eval(nh),
                                       classifier_hidden_feats=eval(chf))

        best_model.load_state_dict(torch.load(models, map_location=device)['model_state_dict'])
        best_model.to(device)
        # print('model screen')
        run_an_eval_epoch(best_model, train_loader, args)

def fp(X, type='ECFP4',file=''):
    smiles = np.array(X)
    count = 0
    if type == 'ECFP4':
        ms = [Chem.MolFromSmiles(smiles[i]) for i in range(len(smiles))]
        ecfpMat = np.zeros((len(ms), 1024), dtype=int)
        for i in range(len(ms)):
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(ms[i], 2, 1024)
                ecfpMat[i] = np.array(list(fp.ToBitString()))
            except:
                count += 1
                ecfpMat[i] = np.zeros((1, 1024), dtype=int)
        X = ecfpMat
        return X

def mlscreen(file='',sep=',',models='',out_dir='./',smiles_col=True):
    # df = pd.read_csv(file, sep=sep,usecols=[i for i in range(5)],header=None,engine='python')
    df = pd.read_csv(file, sep=sep, names=['Smiles'])
    # df.columns = open(file,'r').readline().strip().split(sep)[:5]
    print(df.columns)
    # df['index'] = df.index
    total = len(df)
    print(total, time.time() - s)
    # print('load model')
    modell = joblib.load(models)
    # print(models)
    type = models.split('_')[-4]#[-3]
    model_name = models.split('_')[-2]
    print(model_name)
    # print(type)
    X = df.Smiles
    X = fp(X=X,type=type)
    y_pred = modell.predict(X)
    df[model_name+'score'] = y_pred
    # df = df.drop(index=0)
    output = file.split('/')[-1].replace('.csv','_{}_screen.csv'.format(model_name))
    output = os.path.join(out_dir,output)
    df.to_csv(output,index=False)
    print(len(df), time.time() - s, 'screen',round(len(df)/total *100,2),'%')
    return total,len(df)

def split_file(file):
    if len(os.listdir(smi_path))!=0:
        for i in os.listdir(smi_path):
            os.remove(os.path.join(smi_path,i))
    symbol_num = file.split('.')[0].split('/')[-1]
    count = 0
    for j,line in enumerate(open(os.path.join(path,file)).readlines()):
        if j == 0:
            continue
        file_num = count//100000
        out_file = os.path.join(smi_path,str(symbol_num)+'_'+str(file_num)+'.csv')
        f_o = open(out_file,'a+')
        smi = line.strip().split(',')[0]
        f_o.write(smi+'\n')
        f_o.close()
        count += 1

def result_file(frfile,model):
    filename_csv = []
    frames = []
    for root, dirs, files in os.walk(out_dir):
        for file in files:
            filename_csv.append(os.path.join(root, file))
            df = pd.read_csv(os.path.join(root, file), sep=',')
            frames.append(df)
    result = pd.concat(frames)
    print(len(result))
    result.to_csv(result_dir + '/{}_{}_screen.csv'.format(frfile.split('.')[0], model.split('/')[-2]), index=False)
    for i in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, i))

path = '/home/weili/zyh/library/24_211'
file_label = 'Enamine_REAL_HAC_24_CXSMILES_{}w.smi'
model1 = '/home/weili/zyh/dataset/keti/screen_model/gcn/gcn_random_cla_0_0.03162277660168379_(256, 256)_64_8.pth'
model2 = '/home/weili/zyh/dataset/keti/screen_model/XGB/random_reg_ECFP4_9_XGB_bestModel.pkl'
model3 = '/home/weili/zyh/dataset/keti/screen_model/SVM/random_reg_ECFP4_9_SVM_bestModel.pkl'
model4 = '/home/weili/zyh/dataset/keti/screen_model/gat/gat_random_reg_0.0001_0.003162_(128, 128)_(5, 5)_64_3.pth'
sep = ','
smi_path = os.path.join(path,'temp_smi')
out_dir = os.path.join(path,'guodu')
result_dir = os.path.join(path,'result')


count = 0
num = 0
states = 307
for state in range(states):
    s = time.time()
    state = (state+211)*100
    file = file_label.format(state)
    file_path = os.path.join(path,file)
    split_file(file_path)
    file_num = len(os.listdir(smi_path))
    print(len(os.listdir(smi_path)))

    # p = mp.Pool(processes=int(file_num))
    # for file_content in os.listdir(smi_path):
    #     if '.csv' in file_content:
    #         file_path = os.path.join(smi_path, file_content)
    #         # print(file_path)
    #         param = {'file': file_path, 'sep': sep, 'models': model2, 'out_dir': out_dir}
    #         get = p.apply_async(mlscreen, kwds=param)
    #     # p.apply_async(screen, args=(file,out_dir,models))
    # p.close()
    # p.join()
    # result_file(file, model2)

    p = mp.Pool(processes=int(file_num))
    for file_content in os.listdir(smi_path):
        if '.csv' in file_content:
            file_path = os.path.join(smi_path, file_content)
            param = {'file': file_path, 'sep': sep, 'models': model3, 'out_dir': out_dir}
            get = p.apply_async(mlscreen, kwds=param)
        # p.apply_async(screen, args=(file,out_dir,models))
    p.close()
    p.join()
    result_file(file, model3)

    # p = mp.Pool(processes=int(file_num))
    # for file_content in os.listdir(smi_path):
    #     if '.csv' in file_content:
    #         file_path = os.path.join(smi_path, file_content)
    #         param = {'file': file_path, 'sep': sep, 'models': model4, 'out_dir': out_dir}
    #         get = p.apply_async(dlscreen, kwds=param)
    #     # p.apply_async(screen, args=(file,out_dir,models))
    # p.close()
    # p.join()
    # result_file(file, model4)
    #
    # p = mp.Pool(processes=int(file_num))
    # for file_content in os.listdir(smi_path):
    #     if '.csv' in file_content:
    #         file_path = os.path.join(smi_path, file_content)
    #         param = {'file': file_path, 'sep': sep, 'models': model1, 'out_dir': out_dir}
    #         get = p.apply_async(dlscreen, kwds=param)
    #     # p.apply_async(screen, args=(file,out_dir,models))
    # p.close()
    # p.join()
    # result_file(file, model1)

    time.sleep(5)
    print(time.time() - s)

