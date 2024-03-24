import torch
import os
import numpy as np
import pandas as pd
from dnn_torch_utils import Meter, MyDataset, EarlyStopping, MyDNN, collate_fn, set_random_seed
from hyperopt import fmin, tpe, hp, rand, STATUS_OK, Trials, partial
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss, MSELoss
import gc
import time
s =time.time()
import warnings
from sklearn import preprocessing
from splitdater import split_dataset
from feature_create import create_des
import torch
import multiprocessing as mp

torch.set_num_threads(1)

def run_an_eval_epoch(model, data_loader, args):
    f = open(args['output'],'a+')

    f.write('Smiles,DNNscore\n')
    count = 0
    model.eval()
    # eval_metric = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            eval_metric = Meter()
            Xs, Ys, masks = batch_data
            Xs, Ys, masks = Xs.to(args['device']), Ys.to(args['device']), masks.to(args['device'])
            outputs = model(Xs)
            outputs.cpu()
            Ys.cpu()
            masks.cpu()
#            torch.cuda.empty_cache()

            eval_metric.update(outputs, Ys, torch.tensor([count]))
            roc_score = eval_metric.compute_metric('pred')[0][0]
            smiles = args['data'][args['smiles_col']].tolist()[int(Ys[0])]
            write_check = 0

            f.write('{},{}\n'.format(smiles, roc_score))
            count += 1
            torch.cuda.empty_cache()
            if count%100000 ==0:
                print(count)
        f.close()
def screen(start=None,file='', sep=',', models=None,prop=0.5, smiles_col='Smiles',out_dir=None,tasks=1):
    # my_df = pd.read_csv(file, engine='python', sep=',',skiprows=start,nrows=num_batch,header=None)
    # my_df.columns = ['Smiles', 'dock_score']
    my_df = pd.read_csv(file, engine='python', sep=',')

    device = torch.device("cuda:0")#("cuda:0")
    args = {'device': device, 'metric': 'r2','data':my_df,'smiles_col':smiles_col,'tasks':tasks}
    outputs = os.path.join(out_dir,
                           file.split('/')[-1].replace('.csv', '_DNN_screen.csv'))
    # if os.path.exists(outputs):
    #     print(outputs, 'has done')
    # else:
    #     args['output'] = outputs
    #     FP_type = models.split('/')[-1].split('_')[1]
    #     model_dir = out_dir.replace(out_dir.split('/')[-1],'model_save')
    #     print(smiles_col)
    #     data_x, data_y = create_des(my_df[smiles_col], list(range(len(my_df))), FP_type=FP_type, model_dir=model_dir)
    #
    #     dataset = MyDataset(data_x, data_y)
    #     loader = DataLoader(dataset,  collate_fn=collate_fn)
    #     inputs = data_x.shape[1]
    #     hideen_unit = (eval(models.split('/')[-1].split('_')[5]),
    #                    eval(models.split('/')[-1].split('_')[6])
    #                    ,eval(models.split('/')[-1].split('_')[7]))
    #
    #     dropout = eval(models.split('/')[-1].split('_')[4])
    #     best_model = MyDNN(inputs=inputs, hideen_units=hideen_unit, outputs=tasks,
    #                        dp_ratio=dropout, reg=False)
    #     best_model.load_state_dict(torch.load(models, map_location=device)['model_state_dict'])
    #     best_model.to(device)
    #     print('model screen')
    #     run_an_eval_epoch(best_model, loader, args)
    args['output'] = outputs
    FP_type = models.split('/')[-1].split('_')[1]
    model_dir = out_dir.replace(out_dir.split('/')[-1],'model_save')
    print(smiles_col)
    data_x, data_y = create_des(my_df[smiles_col], list(range(len(my_df))), FP_type=FP_type, model_dir=model_dir)

    dataset = MyDataset(data_x, data_y)
    loader = DataLoader(dataset,  collate_fn=collate_fn)
    inputs = data_x.shape[1]
    hideen_unit = (eval(models.split('/')[-1].split('_')[5]),
                   eval(models.split('/')[-1].split('_')[6]),
                   eval(models.split('/')[-1].split('_')[7]))

    dropout = eval(models.split('/')[-1].split('_')[4])
    best_model = MyDNN(inputs=inputs, hideen_units=hideen_unit, outputs=tasks,
                           dp_ratio=dropout, reg=False)
    best_model.load_state_dict(torch.load(models, map_location=device)['model_state_dict'])
    best_model.to(device)
    print('model screen')
    run_an_eval_epoch(best_model, loader, args)

# screen(out_dir='/home/weili/zyh/dataset/keti/medium/6model/guodu', file='/home/weili/zyh/dataset/keti/data/30w.csv',
#        models='/home/weili/zyh/dataset/keti/medium/6model/model_save/iteration_3/DNN/reg_ECFP4_random_dataset_0.0039_256_128_64_0.0001_early_stop.pth',
#        sep=',',smiles_col='Smiles',)

out_dir='/home/weili/zyh/dataset/keti/medium/6model/guodu'
file = '/home/weili/zyh/dataset/keti/data/'
models ='/home/weili/zyh/dataset/keti/medium/6model/model_save/iteration_7/DNN/reg_ECFP4_random_dataset_0.0048_256_512_64_0.0002_early_stop.pth'
sep = ','

p = mp.Pool(processes=10)
for file_content in os.listdir(file):
    if '.csv' in file_content:
        file_path = os.path.join(file,file_content)
        print(file_path)
        param = {'file': file_path, 'sep': sep, 'models': models, 'out_dir': out_dir}
        get = p.apply_async(screen, kwds=param)
p.close()
p.join()
print(time.time() - s)
