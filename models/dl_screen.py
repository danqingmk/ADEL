import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, SMILESToBigraph, RandomSplitter, EarlyStopping
from dgllife.data import MoleculeCSVDataset
from dgllife.model import GCNPredictor, GATPredictor
from gnn_utils import set_random_seed, Meter, collate_molgraphs
from torch.utils.data import DataLoader
import os
import multiprocessing as mp
import time
s =time.time()
import torch
torch.set_num_threads(50)

def run_an_eval_epoch(model, data_loader, args):
    f = open(args['output'],'w+')
    f.write('Smiles,{}score\n'.format(args['model']))
    # f.write('Smiles,gcnscore\n')
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
            if count % 100000 == 0:
                print(count)
        f.close()

def screen(file='', sep=',', models=None,prop=0.5, smiles_col='Smiles',out_dir=None):
    # AtomFeaturizer = AttentiveFPAtomFeaturizer
    # BondFeaturizer = AttentiveFPBondFeaturizer
    print(models)
    model_type = models.split('/')[-2]
    # device = torch.device("cpu")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    args = {'model': model_type, 'device': device}
    outputs = os.path.join(out_dir,file.split('/')[-1].replace('.csv','_{}_screen.csv'.format(args['model'])))
    print(outputs)
    if os.path.exists(outputs):
        print(outputs,'has done')
    else:
        args['output'] =outputs
        # my_df = pd.read_csv(file,engine='python',sep=sep)
        my_df = pd.read_csv(file)
        print(my_df.columns)
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
            print(chf)
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
            print(chf)
            nh = models.split('_')[-3]
            ghf = models.split('_')[-4]#-3
            best_model = GATPredictor(in_feats=CanonicalAtomFeaturizer().feat_size('h'),
                                       hidden_feats=eval(ghf),
                                       num_heads=eval(nh),
                                       classifier_hidden_feats=eval(chf))

        best_model.load_state_dict(torch.load(models, map_location=device)['model_state_dict'])
        best_model.to(device)
        print('model screen')
        run_an_eval_epoch(best_model, train_loader, args)

screen(models ='/home/weili/zyh/dataset/keti/mukuo/fangcha/model_save/iteration_8/gcn/gcn_random_cla_0_0.03162277660168379_(128, 128)_256_5.pth',
       file='/home/weili/zyh/dataset/keti/B2AR2.csv',prop=0.5,sep = ',',
       out_dir='/home/weili/zyh/dataset/keti/mukuo/B2AR2/fangcha/screen_8',smiles_col='Smiles')

# screen(models ='/home/weili/zyh/dataset/keti/mukuo/fangcha/model_save/iteration_7/gcn/gcn_random_cla_1e-06_0.03162277660168379_(256, 128)_64_7.pth',
#        file='/home/weili/zyh/dataset/keti/mukuo/base.csv',prop=0.5,sep = ',',
#        out_dir='/home/weili/zyh/dataset/keti/mukuo/fangcha/screen_7',smiles_col='Smiles')

# models ='/home/weili/zyh/dataset/keti/single/14-2/model_save/iteration_1/gat/gat_random_reg_1e-08_0.000316_(128, 64)_(5, 5)_256_1.pth'
# file = '/home/weili/zyh/dataset/keti/data1/'
# sep = ','
# out_dir = '/home/weili/zyh/dataset/keti/single/14-2/guodu'
# p = mp.Pool(processes=10)
# for file_content in os.listdir(file):
#     if '.csv' in file_content:
#         file_path = os.path.join(file,file_content)
#         print(file_path)
#         param = {'file': file_path, 'sep': sep, 'models': models, 'out_dir': out_dir}
#         get = p.apply_async(screen, kwds=param)
# p.close()
# p.join()
print(time.time() - s)
