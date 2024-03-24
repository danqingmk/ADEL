import time
import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss, MSELoss
from dgl import backend as F
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, SMILESToBigraph, RandomSplitter, EarlyStopping
from dgllife.data import MoleculeCSVDataset
from dgllife.model import GCNPredictor, GATPredictor
from hyperopt import fmin, tpe, hp, Trials
from gnn_utils import set_random_seed, Meter, collate_molgraphs
from dgllife.model.gnn import GCN

epochs = 300
patience = 50
batch_size = 128*15
start_time = time.time()
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
set_random_seed(seed=42)
torch.set_num_threads(100)

def get_pos_weight(my_dataset):
    num_pos = F.sum(my_dataset.labels, dim=0)
    num_indices = F.tensor(len(my_dataset.labels))
    return (num_indices - num_pos) / num_pos

def train_epoch(model, data_loader, loss_func, optimizer, args):
    model.train()
    train_metric = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, masks = batch_data

        bg = bg.to(args['device'])

        atom_feats = bg.ndata.pop('h')
        bond_feats = bg.edata.pop('e')
        labels, masks, atom_feats, bond_feats = labels.to(args['device']), masks.to(args['device']), \
            atom_feats.to(args['device']), bond_feats.to(args['device'])

        outputs = model(bg, atom_feats)

        loss = (loss_func(outputs, labels) * (masks != 0).float()).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        outputs.cpu()
        labels.cpu()
        masks.cpu()
        atom_feats.cpu()
        bond_feats.cpu()
        loss.cpu()
        # torch.cuda.empty_cache()

        train_metric.update(outputs, labels, masks)

    if args['metric'] == 'rmse':
        rmse_score = np.mean(train_metric.compute_metric(args['metric']))  # in case of multi-tasks
        mae_score = np.mean(train_metric.compute_metric('mae'))  # in case of multi-tasks
        r2_score = np.mean(train_metric.compute_metric('r2'))  # in case of multi-tasks
        return {'rmse': rmse_score, 'mae': mae_score, 'r2': r2_score}
    else:
        roc_score = np.mean(train_metric.compute_metric(args['metric']))  # in case of multi-tasks
        prc_score = np.mean(train_metric.compute_metric('prc_auc'))  # in case of multi-tasks

        return {'roc_auc': roc_score, 'prc_auc': prc_score}


def eval_epoch(model, data_loader, args):
    model.eval()
    eval_metric = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data

            bg = bg.to(args['device'])

            atom_feats = bg.ndata.pop('h')
            bond_feats = bg.edata.pop('e')

            labels, masks, atom_feats, bond_feats = labels.to(args['device']), masks.to(args['device']), \
                atom_feats.to(args['device']), bond_feats.to(args['device'])

            outputs = model(bg, atom_feats)

            outputs.cpu()
            labels.cpu()
            masks.cpu()
            atom_feats.cpu()
            bond_feats.cpu()
            # loss.cpu()
            torch.cuda.empty_cache()

            eval_metric.update(outputs, labels, masks)

    if args['metric'] == 'rmse':
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

def best_model_running(seed, opt_res, data, args,file_name,split_type='random', model_name ='gcn',task_type='cla',model_dir=False,my_df=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    number_workers = 0

    # smiles_to_graph = SMILESToBigraph(add_self_loop=True,
    #                                   node_featurizer=CanonicalAtomFeaturizer(atom_data_field='h'),
    #                                   edge_featurizer=CanonicalBondFeaturizer(bond_data_field='e', self_loop=True))
    # my_dataset = MoleculeCSVDataset(df=my_df,
    #                                 smiles_to_graph=smiles_to_graph,
    #                                 smiles_column='Smiles',
    #                                 cache_file_path=file_name.replace('.csv', '_graph.bin'),
    #                                 task_names=args['tasks'],
    #                                 n_jobs=30)
    if task_type == 'cla':
        pos_weight = get_pos_weight(data)
    else:
        pos_weight = None

    train_set, val_set, test_set = RandomSplitter.train_val_test_split(data,
                                                                       frac_train=0.8, frac_val=0.1, frac_test=0.1,
                                                                       random_state=42)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_molgraphs, num_workers=number_workers)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_molgraphs, num_workers=number_workers)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_molgraphs, num_workers=number_workers)

    if model_name == 'gcn':
        best_model = GCNPredictor(in_feats=CanonicalAtomFeaturizer().feat_size('h'),
                                  hidden_feats=opt_res['gcn_hidden_feats'],
                                  classifier_hidden_feats=opt_res['classifier_hidden_feats'],
                                  classifier_dropout=0.0,
                                  # n_tasks=len(args['tasks']),
                                  predictor_hidden_feats=128,
                                  predictor_dropout=0.0)
        best_model_file_name = '%s/%s_%s_%s_%s_%s_%s_%s_%s.pth' % (model_dir, 'gcn', 'random', 'cla',
                                                    opt_res['l2'], opt_res['lr'],
                                                    opt_res['gcn_hidden_feats'],
                                                    opt_res['classifier_hidden_feats'], seed)
    elif model_name == 'gat':
        best_model = GATPredictor(in_feats=CanonicalAtomFeaturizer().feat_size('h'),
                                  hidden_feats=opt_res['gat_hidden_feats'],
                                  num_heads=opt_res['num_heads'],
                                  classifier_hidden_feats=opt_res['classifier_hidden_feats'],
                             # alphas=[opt_res['alpha']] * opt_res['num_gnn_layers'],
                             # residuals=[opt_res['residual']] * opt_res['num_gnn_layers'],
                             # predictor_hidden_feats=opt_res['predictor_hidden_feats'],
                             #      n_tasks=len(args['tasks'])
                                  )
        # best_model_file_name = '%s/%s_%s_%s_%s_%.6f_%s_%s_%s_%s.pth' % (model_dir, 'gat', split_type, task_type,
        #                                                            opt_res['l2'], opt_res['lr'],
        #                                                            [opt_res['gnn_hidden_feats']] *
        #                                                            opt_res['num_gnn_layers'],
        #                                                            [opt_res['num_heads']] * opt_res[
        #                                                                'num_gnn_layers'],
        #                                                            [float('{:.6f}'.format(i)) for i in
        #                                                             [opt_res['alpha']] * opt_res[
        #                                                                 'num_gnn_layers']],
        #                                                            opt_res['predictor_hidden_feats'])
        best_model_file_name = '%s/%s_%s_%s_%s_%.6f_%s_%s_%s_%s.pth' % (model_dir, 'gat', split_type, task_type,
                                                                opt_res['l2'], opt_res['lr'],
                                                                opt_res['gat_hidden_feats'],
                                                                opt_res['num_heads'],
                                                                opt_res['classifier_hidden_feats'], seed)

    optimizer = torch.optim.Adam(best_model.parameters(), lr=opt_res['lr'], weight_decay=opt_res['l2'])

    if task_type == 'reg':
        loss_func = MSELoss(reduction='none')
        stopper = EarlyStopping(mode='lower', patience=patience, filename=best_model_file_name)
    else:
        loss_func = BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight.to(args['device']))
        stopper = EarlyStopping(mode='higher', patience=patience, filename=best_model_file_name)

    best_model.to(args['device'])

    for i in range(epochs):
        # training
        train_epoch(best_model, train_loader, loss_func, optimizer, args)

        # early stopping
        val_scores = eval_epoch(best_model, val_loader, args)
        early_stop = stopper.step(val_scores[args['metric']], best_model)
        print('best', seed, val_scores)

        if early_stop:
            break
    stopper.load_checkpoint(best_model)

    tr_scores = eval_epoch(best_model, train_loader, args)
    val_scores = eval_epoch(best_model, val_loader, args)
    te_scores = eval_epoch(best_model, test_loader, args)
    result_one = pd.concat([pd.DataFrame([tr_scores],index=['tr']),pd.DataFrame([val_scores],index=['va']),pd.DataFrame([te_scores],index=['te'])])
    result_one['type'] = result_one.index
    result_one['split'] = split_type
    result_one['model'] = model_name
    result_one['seed'] = seed
    if task_type == 'cla':
        result_one.columns = ['auc_roc','auc_prc', 'se', 'sp', 'acc', 'mcc', 'precision', 'type', 'split','model','seed']
    else:
        result_one.columns = ['rmse', 'mae', 'r2', 'type', 'split', 'model', 'seed']

    return result_one

def run_gnn(task_type='cla', file_name=None, model_name=None, split_type=None,model_dir=None,difftasks=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device run for GCN:', device)

    args = {'device': device, 'tasks': difftasks, 'metric': 'roc_auc' if task_type == 'cla' else 'rmse',
            'split_type': split_type, 'model':model_name}

    my_df = pd.read_csv(file_name)

    opt_iters = 50
    number_workers = 0
    repetitions = 9

    smiles_to_graph = SMILESToBigraph(add_self_loop=True,
                                      node_featurizer=CanonicalAtomFeaturizer(atom_data_field='h'),
                                      edge_featurizer=CanonicalBondFeaturizer(bond_data_field='e', self_loop=True))
    my_dataset = MoleculeCSVDataset(df=my_df,
                                    smiles_to_graph=smiles_to_graph,
                                    smiles_column='Smiles',
                                    cache_file_path=file_name.replace('.csv', '_graph.bin'),
                                    task_names=args['tasks'], load=True,
                                    n_jobs=50)
    if task_type == 'cla':
        pos_weight = get_pos_weight(my_dataset)
    else:
        pos_weight = None

    train_set, val_set, test_set = RandomSplitter.train_val_test_split(my_dataset,
                                                                         frac_train=0.8, frac_val=0.1, frac_test=0.1,
                                                                         random_state=42)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_molgraphs, num_workers=number_workers)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_molgraphs, num_workers=number_workers)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_molgraphs, num_workers=number_workers)


    gcn_hyperparameters = {
        'gnn_hidden_feats': hp.choice('gnn_hidden_feats', [32, 64, 128, 256]),
        'predictor_hidden_feats': hp.choice('predictor_hidden_feats', [16, 32, 64, 128, 256, 512, 1024]),
        'num_gnn_layers': hp.choice('num_gnn_layers', [1, 2, 3, 4, 5]),
        'residual': hp.choice('residual', [True, False]),
        'batchnorm': hp.choice('batchnorm', [True, False]),
        'dropout': hp.uniform('dropout', low=0., high=0.6)}

    hyper_paras_space = {'gcn': dict(l2=hp.choice('l2', [0, 10 ** -8, 10 ** -6, 10 ** -4]),
                                     lr=hp.choice('lr', [10 ** -2.5, 10 ** -3.5, 10 ** -1.5]),
                                     num_gnn_layers=hp.choice('num_gnn_layers', [1, 2, 3, 4, 5]),
                                     gcn_hidden_feats=hp.choice('gcn_hidden_feats',
                                                                [[128, 128], [256, 256], [128, 64], [256, 128]]),
                                     classifier_hidden_feats=hp.choice('classifier_hidden_feats', [128, 64, 256])),
                         'gat': dict(l2=hp.choice('l2', [0, 10 ** -8, 10 ** -6, 10 ** -4]),
                                     lr=hp.choice('lr', [10 ** -2.5, 10 ** -3.5, 10 ** -1.5]),
                                     gat_hidden_feats=hp.choice('gat_hidden_feats',
                                                                [[128, 128], [256, 256], [128, 64], [256, 128]]),
                                     classifier_hidden_feats=hp.choice('classifier_hidden_feats', [128, 64, 256]),
                                     num_heads=hp.choice('num_heads', [[4, 4],[4, 5],[3, 3],[5, 5],[4, 3]]),
                                     # alpha=hp.uniform('alpha', low=0., high=1),
                                     # predictor_hidden_feats=hp.choice('predictor_hidden_feats', [16, 32, 64, 128, 256]),
                                     # num_gnn_layers=hp.choice('num_gnn_layers', [1, 2, 3, 4, 5]),
                                     # residual=hp.choice('residual', [True, False]),
                                     # dropout=hp.uniform('dropout', low=0., high=0.6)
                                     )}
    hyper_paras_space = hyper_paras_space[args['model']]

    def hyper_opt(hyper_paras):
        if model_name == 'gcn':
            model = GCNPredictor(in_feats=CanonicalAtomFeaturizer().feat_size('h'),
                             hidden_feats=hyper_paras['gcn_hidden_feats'],
                             classifier_hidden_feats=hyper_paras['classifier_hidden_feats'],
                             classifier_dropout=0.0,
                             n_tasks=len(args['tasks']),
                             predictor_hidden_feats=128,
                             predictor_dropout=0.0)
            model_file_name = '%s/%s_%s_%s_%s_%s_%s_%s.pth' % (model_dir, 'gcn',split_type, task_type,
                                                             hyper_paras['l2'], hyper_paras['lr'],
                                                             hyper_paras['gcn_hidden_feats'],
                                                             hyper_paras['classifier_hidden_feats'])
        elif model_name == 'gat':
            model = GATPredictor(in_feats=CanonicalAtomFeaturizer().feat_size('h'),
                                 hidden_feats=hyper_paras['gat_hidden_feats'],
                                 num_heads=hyper_paras['num_heads'],
                                 classifier_hidden_feats=hyper_paras['classifier_hidden_feats'],
                             # feat_drops=[hyper_paras['dropout']] * hyper_paras['num_gnn_layers'],
                             # attn_drops=[hyper_paras['dropout']] * hyper_paras['num_gnn_layers'],
                             # alphas=[hyper_paras['alpha']] * hyper_paras['num_gnn_layers'],
                             # residuals=[hyper_paras['residual']]* hyper_paras['num_gnn_layers'],
                             # predictor_hidden_feats=hyper_paras['predictor_hidden_feats'],
                             # predictor_dropout=hyper_paras['dropout'],
                             # n_tasks=len(args['tasks'])
                                 )
            # model_file_name = '%s/%s_%s_%s_%s_%.6f_%s_%s_%s_%s.pth' % (model_dir, 'gat', split_type, task_type,
            #                                                                  hyper_paras['l2'], hyper_paras['lr'],
            #                                                                  [hyper_paras['gat_hidden_feats']] *hyper_paras['num_gnn_layers'],
            #                                                                  [hyper_paras['num_heads']] * hyper_paras['num_gnn_layers'],
            #                                                                  [float('{:.6f}'.format(i)) for i in[hyper_paras['alpha']] * hyper_paras['num_gnn_layers']],
            #                                                                  hyper_paras['predictor_hidden_feats'])
            model_file_name = '%s/%s_%s_%s_%s_%.6f_%s_%s_%s.pth' % (model_dir, 'gat', split_type, task_type,
                                                                    hyper_paras['l2'], hyper_paras['lr'],
                                                                    hyper_paras['gat_hidden_feats'],
                                                                    hyper_paras['num_heads'],
                                                                    hyper_paras['classifier_hidden_feats'])

        optimizer = torch.optim.Adam(model.parameters(), lr=hyper_paras['lr'], weight_decay=hyper_paras['l2'])

        if task_type == 'reg':
            loss_func = MSELoss(reduction='none')
            stopper = EarlyStopping(mode='lower', patience=patience, filename=model_file_name)
        else:
            loss_func = BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight.to(args['device']))
            stopper = EarlyStopping(mode='higher', patience=patience,filename=model_file_name)

        model.to(args['device'])

        for i in range(epochs):
            # training
            train_epoch(model, train_loader, loss_func, optimizer, args)

            # early stopping
            val_scores = eval_epoch(model, val_loader, args)
            early_stop = stopper.step(val_scores[args['metric']], model)

            if early_stop:
                break
        stopper.load_checkpoint(model)

        val_scores = eval_epoch(model, val_loader, args)

        feedback = val_scores[args['metric']] if task_type == 'reg' else (1 - val_scores[args['metric']])
        model.cpu()
        torch.cuda.empty_cache()
        gc.collect()
        return feedback

    # start hyper-parameters optimization
    trials = Trials()
    opt_res = fmin(hyper_opt, hyper_paras_space, algo=tpe.suggest, max_evals=opt_iters, trials=trials)
    print('the best hyper-parameters settings for ' + 'activity ' + args['split_type'] + model_name + ' are:  ', opt_res)

    l2_ls = [0, 10 ** -8, 10 ** -6, 10 ** -4]
    lr_ls = [10 ** -2.5, 10 ** -3.5, 10 ** -1.5]
    hidden_feats_ls = [(128, 128), (256, 256), (128, 64), (256, 128)]
    classifier_hidden_feats_ls = [128, 64, 256]
    num_heads_ls = [(4, 4),(4, 5),(3, 3),(5, 5),(4, 3)]
    gnn_hidden_feats = [32, 64, 128, 256]
    num_heads = [4, 6, 8]
    predictor_hidden_feats = [16, 32, 64, 128, 256]
    num_gnn_layers = [1, 2, 3, 4, 5]
    residual = [True, False]

    if model_name == 'gcn':
        param = {'l2': l2_ls[opt_res['l2']], 'lr': lr_ls[opt_res['lr']],
                 'gcn_hidden_feats': hidden_feats_ls[opt_res['gcn_hidden_feats']],
                 'classifier_hidden_feats': classifier_hidden_feats_ls[opt_res['classifier_hidden_feats']]}
    # elif model_name == 'gat':
    #     param = {'hidden_feats': gnn_hidden_feats[opt_res['gat_hidden_feats']]*num_gnn_layers[opt_res['num_gnn_layers']],
    #              'num_heads': num_heads[opt_res['num_heads']]*num_gnn_layers[opt_res['num_gnn_layers']],
    #              'predictor_hidden_feats': predictor_hidden_feats[opt_res['predictor_hidden_feats']],
    #              'residual': residual[opt_res['residual']]}
    elif model_name == 'gat':
        param = {'l2': l2_ls[opt_res['l2']], 'lr': lr_ls[opt_res['lr']],
                 'gat_hidden_feats': hidden_feats_ls[opt_res['gat_hidden_feats']],
                 'num_heads': num_heads_ls[opt_res['num_heads']],
                 'classifier_hidden_feats': classifier_hidden_feats_ls[opt_res['classifier_hidden_feats']]}

    param_file = str(model_dir).replace('model_save', 'param_save') + '/%s_%s_%s' % (
        args['split_type'], task_type, '{}.param'.format(model_name))
    if not os.path.exists(str(model_dir).replace('model_save', 'param_save')):
        os.makedirs(str(model_dir).replace('model_save', 'param_save'))
    print(os.path.exists(str(model_dir).replace('model_save', 'param_save')))
    f = open(param_file, 'w')
    f.write('%s' % param)
    f.close()
    print('para file has done')

    opt_res = eval(open(param_file,'r').readline().strip())

    if model_name == 'gcn':
        best_model = GCNPredictor(in_feats=CanonicalAtomFeaturizer().feat_size('h'),
                              hidden_feats=opt_res['gcn_hidden_feats'],
                              classifier_hidden_feats=opt_res['classifier_hidden_feats'],
                              classifier_dropout=0.0,
                              n_tasks=len(args['tasks']),
                              predictor_hidden_feats=128,
                              predictor_dropout=0.0)
        best_model_file = '%s/%s_%s_%s_%s_%s_%s_%s.pth' % (model_dir, 'gcn',split_type, task_type,
                                                           opt_res['l2'], opt_res['lr'], opt_res['gcn_hidden_feats'],
                                                           opt_res['classifier_hidden_feats'])
    elif model_name == 'gat':
        best_model = GATPredictor(in_feats=CanonicalAtomFeaturizer().feat_size('h'),
                                  hidden_feats=opt_res['gat_hidden_feats'],
                                  num_heads=opt_res['num_heads'],
                                  classifier_hidden_feats=opt_res['classifier_hidden_feats'],
                             # feat_drops=[opt_res['dropout']] * opt_res['num_gnn_layers'],
                             # attn_drops=[opt_res['dropout']] * opt_res['num_gnn_layers'],
                             # alphas=[opt_res['alpha']] * opt_res['num_gnn_layers'],
                             # residuals=[opt_res['residual']]* opt_res['num_gnn_layers'],
                             # predictor_hidden_feats=opt_res['predictor_hidden_feats'],
                             # predictor_dropout=opt_res['dropout'],agg_modes=None,activations=None,
                             # n_tasks=len(args['tasks'])
                                  )
        # best_model_file = '%s/%s_%s_%s_%s_%.6f_%s_%s_%s_%s.pth' % (model_dir, 'gat', split_type, task_type,
        #                                                                  opt_res['l2'], opt_res['lr'],
        #                                                                  [opt_res['gat_hidden_feats']] *
        #                                                                  opt_res['num_gnn_layers'],
        #                                                                  [opt_res['num_heads']] * opt_res[
        #                                                                      'num_gnn_layers'],
        #                                                                  [float('{:.6f}'.format(i)) for i in
        #                                                                   [opt_res['alpha']] * opt_res[
        #                                                                       'num_gnn_layers']],
        #                                                                  opt_res['predictor_hidden_feats'])
        best_model_file = '%s/%s_%s_%s_%s_%.6f_%s_%s_%s.pth' % (model_dir, 'gat', split_type, task_type,
                                                                    opt_res['l2'], opt_res['lr'],
                                                                    opt_res['gat_hidden_feats'],
                                                                    opt_res['num_heads'],
                                                                    opt_res['classifier_hidden_feats'])

    best_model.load_state_dict(torch.load(best_model_file, map_location=device)['model_state_dict'])
    best_model.to(device)

    tr_scores = eval_epoch(best_model, train_loader, args)
    val_scores = eval_epoch(best_model, val_loader, args)
    te_scores = eval_epoch(best_model, test_loader, args)
    print('training set:', tr_scores)
    print('validation set:', val_scores)
    print('test set:', te_scores)

    if task_type == 'cla':
        record = {'auc_roc':[tr_scores['roc_auc'],val_scores['roc_auc'],te_scores['roc_auc']],
                  'auc_prc':[tr_scores['prc_auc'],val_scores['prc_auc'],te_scores['prc_auc']],
              'se':[tr_scores['se'],val_scores['se'],te_scores['se']],
              'sp':[tr_scores['sp'],val_scores['sp'],te_scores['sp']],
              'acc':[tr_scores['acc'],val_scores['acc'],te_scores['acc']],
              'mcc':[tr_scores['mcc'],val_scores['mcc'],te_scores['mcc']],
              'precision':[tr_scores['pre'],val_scores['pre'],te_scores['pre']]}
    else:
        record = {'rmse':[tr_scores['rmse'],val_scores['rmse'],te_scores['rmse']],
                  'r2':[tr_scores['r2'],val_scores['r2'],te_scores['r2']],
                  'mae':[tr_scores['mae'],val_scores['mae'],te_scores['mae']]}
    param = {k:[v,v,v] for k,v in param.items()}
    record = {k:v for d in [param,record] for k,v in d.items()}
    best_res = pd.DataFrame(record,index=['tr','va','te'])
    best_res['type'] = best_res.index
    best_res['split'] = args['split_type']
    best_res['model'] = model_name
    best_res['seed'] = 0

    if task_type == 'cla':
        para_res = best_res[[ 'auc_roc',
            'auc_prc', 'se', 'sp', 'acc', 'mcc', 'precision', 'type', 'split',
            'model','seed']]
    else:
        para_res = best_res[['rmse','mae','r2','type','split','model','seed']]
    for seed in range(1,repetitions+1):
        res_best = best_model_running(seed, opt_res, my_dataset, args, file_name, split_type=args['split_type'], model_name=model_name, task_type=task_type,
                       model_dir=model_dir, my_df=my_df)
        para_res = pd.concat([para_res,res_best],ignore_index=True)
    result_dir = model_dir.replace('model_save', 'result_save')
    para_name, best_name = os.path.join(result_dir,
                                        '_'.join([args['split_type'], model_name,'para.csv'])), os.path.join(
        result_dir, '_'.join([args['split_type'], model_name,'best.csv']))
    para_res.to_csv(best_name, index=False)
    best_res.to_csv(para_name, index=False)
    print(para_res.groupby(['split','type'])['rmse','r2','mae'].mean(),best_res)
    return


# if __name__ == '__main__':
#     file_name = '/home/weili/zyh/dataset/keti/test/test.csv'
#     model_dir = '/home/weili/zyh/dataset/keti/test/'
#     difftasks = ['pValue']
#     run_gnn(task_type='reg', file_name=file_name, model_name='gat', split_type='random', model_dir=model_dir, difftasks=difftasks)
