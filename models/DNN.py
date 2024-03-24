import os
import torch
import numpy as np
import pandas as pd
from dnn_torch_utils import Meter, MyDataset, EarlyStopping, MyDNN, collate_fn, set_random_seed
from hyperopt import fmin, tpe, hp, rand, STATUS_OK, Trials, partial
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss, MSELoss
import gc
import time
start_time = time.time()
import warnings
from sklearn import preprocessing
from splitdater import split_dataset
from feature_create import create_des
from data_utils import TVT


warnings.filterwarnings('ignore')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
set_random_seed(seed=42)
torch.set_num_threads(70)
epochs = 300  # training epoch
batch_size = 128
patience = 50

def get_split_index(data,num,split_type='random',random =42):
    # data_tr_x, data_va_x, data_te_x, data_tr_y, data_va_y, data_te_y = split_dataset(X, Y, split_type=split_type,
    #                                                                                  valid_need=True,random_state=random)

    data_x, data_te_x, data_y, data_te_y = data.set2ten(num)
    data_tr_x, data_va_x, data_tr_y, data_va_y = split_dataset(data_x, data_y, split_type=split_type, valid_need=False,
                                                               random_state=random, train_size=(8 / 9))
    tr = pd.DataFrame([data_tr_x, data_tr_y]).T;
    tr['group'] = 'train'
    va = pd.DataFrame([data_va_x, data_va_y]).T;
    va['group'] = 'valid'
    te = pd.DataFrame([data_te_x, data_te_y]).T;
    te['group'] = 'test'
    df = pd.concat([tr, va, te])

    return df[df.group == 'train'].index, df[df.group == 'valid'].index, df[
        df.group == 'test'].index

def run_a_train_epoch(model, data_loader, loss_func, optimizer, args):
    model.train()

    train_metric = Meter()  # for each epoch
    for batch_id, batch_data in enumerate(data_loader):
        Xs, Ys, masks = batch_data

        # transfer the data to device(cpu or cuda)
        Xs, Ys, masks = Xs.to(args['device']), Ys.to(args['device']), masks.to(args['device'])

        outputs = model(Xs)

        loss = (loss_func(outputs, Ys) * (masks != 0).float()).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        outputs.cpu()
        Ys.cpu()
        masks.cpu()
        loss.cpu()
#        torch.cuda.empty_cache()

        train_metric.update(outputs, Ys, masks)
    if args['reg']:
        rmse_score = np.mean(train_metric.compute_metric(args['metric']))  # in case of multi-tasks
        mae_score = np.mean(train_metric.compute_metric('mae'))  # in case of multi-tasks
        r2_score = np.mean(train_metric.compute_metric('r2'))  # in case of multi-tasks
        return {'rmse': rmse_score, 'mae': mae_score, 'r2': r2_score}
    else:
        roc_score = np.mean(train_metric.compute_metric(args['metric']))  # in case of multi-tasks
        prc_score = np.mean(train_metric.compute_metric('prc_auc'))  # in case of multi-tasks
        return {'roc_auc': roc_score, 'prc_auc': prc_score}

def run_an_eval_epoch(model, data_loader, args):
    model.eval()

    eval_metric = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):

            Xs, Ys, masks = batch_data
            # transfer the data to device(cpu or cuda)
            Xs, Ys, masks = Xs.to(args['device']), Ys.to(args['device']), masks.to(args['device'])

            outputs = model(Xs)

            outputs.cpu()
            Ys.cpu()
            masks.cpu()
#            torch.cuda.empty_cache()
            eval_metric.update(outputs, Ys, masks)
    if args['reg']:
        rmse_score = np.mean(eval_metric.compute_metric(args['metric']))  # in case of multi-tasks
        mae_score = np.mean(eval_metric.compute_metric('mae'))  # in case of multi-tasks
        r2_score = np.mean(eval_metric.compute_metric('r2'))  # in case of multi-tasks
        return {'rmse': rmse_score, 'mae': mae_score, 'r2': r2_score}
    else:

        roc_score = np.mean(eval_metric.compute_metric(args['metric']))  # in case of multi-tasks
        prc_score = np.mean(eval_metric.compute_metric('prc_auc'))  # in case of multi-tasks
        se = np.mean(eval_metric.compute_metric('se'), axis=0)
        sp = np.mean(eval_metric.compute_metric('sp'), axis=0)
        acc = np.mean(eval_metric.compute_metric('acc'), axis=0)
        mcc = np.mean(eval_metric.compute_metric('mcc'), axis=0)
        precision = np.mean(eval_metric.compute_metric('precision'), axis=0)
        return {'roc_auc': roc_score, 'prc_auc': prc_score, 'se': se, 'sp': sp, 'acc': acc, 'mcc': mcc,
                'pre': precision}


def get_pos_weight(Ys):
    Ys = torch.tensor(np.nan_to_num(Ys), dtype=torch.float32)
    num_pos = torch.sum(Ys, dim=0)
    num_indices = torch.tensor(len(Ys))
    return (num_indices - num_pos) / num_pos

def standardize(col):
    return (col - np.mean(col)) / np.std(col)

def all_one_zeros(series):
    if (len(series.dropna().unique()) == 2):
        flag = False
    else:
        flag = True
    return flag

def best_model_running(seed, opt_res, data, split_type='random', FP_type='ECFP4',task_type='cla',model_dir=False,difftasks=None,my_df=None):
    # construct the model based on the optimal hyper-parameters
    pd_res = []
    epochs = 300  # training epoch

    batch_size = 128
    patience = 50
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("cpu")
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    # if device == 'cuda':
    #    torch.cuda.set_device(eval(gpu_id))  # gpu device id
    reg = False if task_type=='cla' else True

    args = {'device': device, 'metric': 'rmse' if reg else 'roc_auc', 'epochs': epochs,
            'patience': patience,  'reg': reg}
    if task_type=='cla':
        while True:
            tr_indx, val_indx, te_indx = get_split_index(data,seed, split_type='random', random=seed)

            data_tr_x, data_tr_y = my_df.loc[tr_indx, 'Smiles'], my_df.loc[tr_indx, difftasks]
            data_va_x, data_va_y = my_df.loc[val_indx, 'Smiles'], my_df.loc[val_indx, difftasks]
            data_te_x, data_te_y = my_df.loc[te_indx, 'Smiles'], my_df.loc[te_indx, difftasks]


            data_tr_x, data_tr_y = create_des(data_tr_x, data_tr_y, FP_type=FP_type,model_dir=model_dir)
            data_va_x, data_va_y = create_des(data_va_x, data_va_y, FP_type=FP_type,model_dir=model_dir)
            data_te_x, data_te_y = create_des(data_te_x, data_te_y, FP_type=FP_type,model_dir=model_dir)
            if np.any(data_tr_y[difftasks].apply(all_one_zeros)) or \
                        np.any(data_va_y[difftasks].apply(all_one_zeros)) or \
                        np.any(data_te_y[difftasks].apply(all_one_zeros)):
                # print('\ninvalid random seed {} due to one class presented in the splitted {} sets...'.format(seed,
                #                                                                                               split_type))
                # print('Changing to another random seed...\n')
                seed = np.random.randint(50, 999999)
            else:

                break
    else:
        tr_indx, val_indx, te_indx = get_split_index(data, seed, split_type=split_type, random=seed)

        data_tr_x, data_tr_y = my_df.loc[tr_indx, 'Smiles'], my_df.loc[tr_indx, difftasks]
        data_va_x, data_va_y = my_df.loc[tr_indx, 'Smiles'], my_df.loc[val_indx, difftasks]
        data_te_x, data_te_y = my_df.loc[tr_indx, 'Smiles'], my_df.loc[te_indx, difftasks]
        data_tr_x, data_tr_y = create_des(data_tr_x, data_tr_y, FP_type=FP_type)
        data_va_x, data_va_y = create_des(data_va_x, data_va_y, FP_type=FP_type)
        data_te_x, data_te_y = create_des(data_te_x, data_te_y, FP_type=FP_type)

    data_tr_y = data_tr_y.values.reshape(-1, len(difftasks))
    data_va_y = data_va_y.values.reshape(-1, len(difftasks))
    data_te_y = data_te_y.values.reshape(-1, len(difftasks))



    # dataloader
    train_dataset = MyDataset(data_tr_x, data_tr_y)
    validation_dataset = MyDataset(data_va_x, data_va_y)
    test_dataset = MyDataset(data_te_x, data_te_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    inputs = data_tr_x.shape[1]
    best_model = MyDNN(inputs=inputs, hideen_units=opt_res['hidden_units'], outputs=len(difftasks),
                       dp_ratio=opt_res['dropout'], reg=reg)

    best_optimizer = torch.optim.Adadelta(best_model.parameters(), weight_decay=opt_res['l2'])
    file_nam = '%s/%s_%s_%s_%s_%.4f_%d_%d_%d_%.4f_early_stop_%d.pth' % (
    model_dir,task_type,FP_type, split_type, 'dataset', opt_res['dropout'],
    opt_res['hidden_units'][0],
    opt_res['hidden_units'][1],
    opt_res['hidden_units'][2],
    opt_res['l2'], seed)
    if reg:
        loss_func = MSELoss(reduction='none')
        stopper = EarlyStopping(mode='lower', patience=patience, filename=file_nam)
    else:

        pos_weights = get_pos_weight(data.Y.values)
        loss_func = BCEWithLogitsLoss(reduction='none', pos_weight=pos_weights.to(args['device']))
        stopper = EarlyStopping(mode='higher', patience=patience, filename=file_nam)
    best_model.to(device)

    for j in range(epochs):
        # training

        run_a_train_epoch(best_model, train_loader, loss_func, best_optimizer, args)

        # early stopping
        train_scores = run_an_eval_epoch(best_model, train_loader, args)
        val_scores = run_an_eval_epoch(best_model, validation_loader, args)
        early_stop = stopper.step(val_scores[args['metric']], best_model)
        if early_stop:
            break

        stopper.load_checkpoint(best_model)
    if task_type=='cla':
        tr_scores = run_an_eval_epoch(best_model, train_loader, args)
        tr_results = [seed,FP_type, split_type, 'tr',tr_scores['se'], tr_scores['sp'],
                      tr_scores['acc'],tr_scores['mcc'], tr_scores['pre'],
                      tr_scores['prc_auc'],tr_scores['roc_auc']]

        val_scores = run_an_eval_epoch(best_model, validation_loader, args)
        va_results = [seed,FP_type, split_type, 'va',val_scores['se'], val_scores['sp'],
                      val_scores['acc'],val_scores['mcc'], val_scores['pre'],
                      val_scores['prc_auc'],val_scores['roc_auc']]
        te_scores = run_an_eval_epoch(best_model, test_loader, args)
        te_results = [seed,FP_type, split_type, 'te',te_scores['se'], te_scores['sp'],
                      te_scores['acc'],te_scores['mcc'], te_scores['pre'],
                      te_scores['prc_auc'],te_scores['roc_auc']]
        pd_res.append(tr_results);pd_res.append(va_results), pd_res.append(te_results)
    else:
        tr_scores = run_an_eval_epoch(best_model, train_loader, args)
        val_scores = run_an_eval_epoch(best_model, validation_loader, args)
        te_scores = run_an_eval_epoch(best_model, test_loader, args)
        tr_results = [seed,FP_type, split_type, 'tr',
                      tr_scores['rmse'], tr_scores['mae'],
                      tr_scores['r2']]
        va_results = [seed,FP_type, split_type, 'va',
                      val_scores['rmse'], val_scores['mae'],
                      val_scores['r2']]
        te_results = [seed,FP_type, split_type, 'te',
                     te_scores['rmse'], te_scores['mae'],
                      te_scores['r2']]

        pd_res.append(tr_results); pd_res.append(va_results); pd_res.append(te_results)


    return pd_res

def run_dnn(X,Y,split_type='random',FP_type='ECFP4',task_type='cla',model_dir=None,file_name='',difftasks=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("cpu")
    reg = True if task_type == 'reg' else False
    args = {'device': device, 'metric': 'roc_auc' if task_type == 'cla' else 'rmse', 'epochs': epochs,
            'split_type': split_type,'reg': reg}

    my_df = pd.read_csv(file_name)

    opt_iters = 50
    random_state = 42

    hyper_paras_space = {'l2': hp.uniform('l2', 0, 0.01),
                         'dropout': hp.uniform('dropout', 0, 0.5),
                         'hidden_unit1': hp.choice('hidden_unit1', [64, 128, 256, 512]),
                         'hidden_unit2': hp.choice('hidden_unit2', [64, 128, 256, 512]),
                         'hidden_unit3': hp.choice('hidden_unit3', [64, 128, 256, 512])}

    a = 0
    # while True:
    #     a=a+1
    #     print(a)
    data = TVT(X, Y)
    tr_indx, val_indx, te_indx = get_split_index(data, 0, split_type=split_type, random=random_state)
    data_tr_x, data_tr_y = my_df.loc[tr_indx,'Smiles'],my_df.loc[tr_indx,difftasks]
    data_va_x, data_va_y = my_df.loc[val_indx,'Smiles'],my_df.loc[val_indx,difftasks]
    data_te_x, data_te_y = my_df.loc[te_indx,'Smiles'],my_df.loc[te_indx,difftasks]


    data_tr_x, data_tr_y = create_des(data_tr_x, data_tr_y, FP_type=FP_type, model_dir=model_dir)
    data_va_x, data_va_y = create_des(data_va_x, data_va_y, FP_type=FP_type, model_dir=model_dir)
    data_te_x, data_te_y = create_des(data_te_x, data_te_y, FP_type=FP_type, model_dir=model_dir)

    data_tr_y = data_tr_y.values.reshape(-1,len(difftasks))
    data_va_y = data_va_y.values.reshape(-1,len(difftasks))
    data_te_y = data_te_y.values.reshape(-1,len(difftasks))

    # dataloader

    train_dataset = MyDataset(data_tr_x, data_tr_y)
    validation_dataset = MyDataset(data_va_x, data_va_y)
    test_dataset = MyDataset(data_te_x, data_te_y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    inputs = data_tr_x.shape[1]

    if task_type == 'cla':
        pos_weight = get_pos_weight(my_df[difftasks].values)
    else:
        pos_weight = None

    def hyper_opt(hyper_paras):
        hidden_units = [hyper_paras['hidden_unit1'], hyper_paras['hidden_unit2'], hyper_paras['hidden_unit3']]
        my_model = MyDNN(inputs=inputs, hideen_units=hidden_units, dp_ratio=hyper_paras['dropout'],
                         outputs=len(difftasks), reg=reg)
        optimizer = torch.optim.Adadelta(my_model.parameters(), weight_decay=hyper_paras['l2'])
        if model_dir:
            file_nam = '%s/%s_%s_%s_%s_%.4f_%d_%d_%d_%.4f_early_stop.pth' % (model_dir,task_type,FP_type,split_type,'dataset', hyper_paras['dropout'],
                                                                  hyper_paras['hidden_unit1'],
                                                                  hyper_paras['hidden_unit2'],
                                                                  hyper_paras['hidden_unit3'],
                                                                  hyper_paras['l2'])
        else:
            file_nam = file_name +'.pth'
        if reg:
            loss_func = MSELoss(reduction='none')
            stopper = EarlyStopping(mode='lower', patience=patience, filename=file_nam)
        else:
            loss_func = BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight.to(args['device']))
            stopper = EarlyStopping(mode='higher', patience=patience, filename=file_nam)
        my_model.to(device)
        for i in range(epochs):
            # training
            run_a_train_epoch(my_model, train_loader, loss_func, optimizer, args)

            # early stopping
            val_scores = run_an_eval_epoch(my_model, validation_loader, args)

            early_stop = stopper.step(val_scores[args['metric']], my_model)

            if early_stop:
                break
        stopper.load_checkpoint(my_model)
        val_scores = run_an_eval_epoch(my_model, validation_loader, args)

        feedback = val_scores[args['metric']] if reg else (1 - val_scores[args['metric']])

        my_model.cpu()
        torch.cuda.empty_cache()
        gc.collect()
        return feedback

    trials = Trials()  # 通过Trials捕获信息
    print('reg',reg)
    opt_res = fmin(hyper_opt, hyper_paras_space, algo=tpe.suggest, max_evals=opt_iters, trials=trials)

    # construct the model based on the optimal hyper-parameters
    hidden_unit1_ls = [64, 128, 256, 512]
    hidden_unit2_ls = [64, 128, 256, 512]
    hidden_unit3_ls = [64, 128, 256, 512]
    opt_hidden_units = [hidden_unit1_ls[opt_res['hidden_unit1']], hidden_unit2_ls[opt_res['hidden_unit2']],
                        hidden_unit3_ls[opt_res['hidden_unit3']]]
    opt_res = {'l2': opt_res['l2'], 'dropout': opt_res['dropout'], 'hidden_units': opt_hidden_units}

    para_file = str(model_dir).replace('model_save', 'param_save') + '/%s_%s_%s_%s' % (
    split_type, task_type, FP_type, 'DNN.param')
    if not os.path.exists(str(model_dir).replace('model_save', 'param_save')):
        os.makedirs(str(model_dir).replace('model_save', 'param_save'))
    print(os.path.exists(str(model_dir).replace('model_save', 'param_save')))
    f = open(para_file, 'w')
    f.write('%s' % opt_res)
    f.close()
    print('para file has done')

    best_model = MyDNN(inputs=inputs, hideen_units=opt_res['hidden_units'], outputs=len(difftasks),
                       dp_ratio=opt_res['dropout'], reg=reg)
    best_model_file = '%s/%s_%s_%s_%s_%.4f_%d_%d_%d_%.4f_early_stop.pth' % (
        model_dir, task_type, FP_type, split_type, 'dataset', opt_res['dropout'],
        opt_res['hidden_units'][0],
        opt_res['hidden_units'][1],
        opt_res['hidden_units'][2],
        opt_res['l2'])
    best_model.load_state_dict(torch.load(best_model_file, map_location=device)['model_state_dict'])
    best_model.to(device)

    tr_scores = run_an_eval_epoch(best_model, train_loader, args)
    val_scores = run_an_eval_epoch(best_model, validation_loader, args)
    te_scores = run_an_eval_epoch(best_model, test_loader, args)

    if task_type == 'cla':
        pd_res = []
        tr_results = [0, FP_type, split_type, 'tr', tr_scores['se'], tr_scores['sp'],
                      tr_scores['acc'], tr_scores['mcc'], tr_scores['pre'],
                      tr_scores['prc_auc'], tr_scores['roc_auc']]
        va_results = [0, FP_type, split_type, 'va', val_scores['se'], val_scores['sp'],
                      val_scores['acc'], val_scores['mcc'], val_scores['pre'],
                      val_scores['prc_auc'], val_scores['roc_auc']]
        te_results = [0, FP_type, split_type, 'te', te_scores['se'], te_scores['sp'],
                      te_scores['acc'], te_scores['mcc'], te_scores['pre'],
                      te_scores['prc_auc'], te_scores['roc_auc']]
        pd_res.append(tr_results)
        pd_res.append(va_results), pd_res.append(te_results)
        para_res = pd.DataFrame(pd_res, columns=['seed','FP_type', 'split_type', 'type',
                                                  'se', 'sp', 'acc', 'mcc',
                                                 'precision', 'auc_prc', 'auc_roc'])

    else:
        pd_res = []
        tr_results = [0, FP_type, split_type, 'tr',
                      tr_scores['rmse'], tr_scores['mae'],
                      tr_scores['r2']]
        va_results = [0, FP_type, split_type, 'va',
                      val_scores['rmse'], val_scores['mae'],
                      val_scores['r2']]
        te_results = [0, FP_type, split_type, 'te',
                      te_scores['rmse'], te_scores['mae'],
                      te_scores['r2']]
        pd_res.append(tr_results)
        pd_res.append(va_results)
        pd_res.append(te_results)
        para_res = pd.DataFrame(pd_res, columns=['seed','FP_type', 'split_type', 'type','rmse', 'mae', 'r2'])

    for i in range(9):
        item = best_model_running((i + 1), opt_res, data, split_type=split_type, FP_type=FP_type, task_type=task_type,
                                 model_dir=model_dir, difftasks=difftasks, my_df=my_df)
        pd_res.extend(item)
    if not reg:

        best_res = pd.DataFrame(pd_res, columns=['seed', 'FP_type', 'split_type', 'type',
                                                 'se', 'sp',
                                                 'acc', 'mcc', 'precision', 'auc_prc', 'auc_roc'])

        pd1 = para_res[['seed','FP_type', 'split_type', 'type',
                        'precision', 'se', 'sp',
                        'acc', 'mcc', 'auc_prc', 'auc_roc']]
        best_res = pd.concat([pd1, best_res], ignore_index=True)
        # best_res = best_res.groupby(['FP_type', 'split_type', 'type'])['acc', 'mcc', 'auc_prc', 'auc_roc'].mean()
    else:
        best_res = pd.DataFrame(pd_res, columns=['seed', 'FP_type', 'split_type', 'type',
                                                 'rmse', 'mae', 'r2'])

        pd1 = para_res[['seed','FP_type', 'split_type', 'type',
                        'rmse', 'r2', 'mae']]
        best_res = pd.concat([pd1, best_res], ignore_index=True)
    result_dir = model_dir.replace('model_save', 'result_save')
    para_name, best_name = os.path.join(result_dir,
                                        '_'.join([split_type, 'DNN', FP_type, 'para.csv'])), os.path.join(
        result_dir, '_'.join([split_type, 'DNN', FP_type, 'best.csv']))
    para_res.to_csv(para_name, index=False)
    best_res.to_csv(best_name, index=False)

    return para_res, best_res

# file = '/data/jianping/bokey/OCAICM/dataset/aurorab/aurorab.csv'
# df = pd.read_csv(file)
# # para = eval(open('/data/jianping/bokey/file_anyway/DNN_scaffold_MACCS.parameter','r').readline().strip())
# a,b =para_dnn(df['Smiles'],df['activity'],
#               opt_res={'l2': 0.00030709542487378596, 'dropout': 0.23899660688719013, 'hidden_units': [256, 256, 128]},
#               split_type='cluster',FP_type='ECFP4',task_type = 'cla'
#               ,model_dir='/data/jianping/bokey/file_anyway/',file_name=file,difftasks=['activity'])
# print(a,b)