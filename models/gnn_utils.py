import time

import numpy as np
import torch
import torch.nn.functional as F
import dgl
import random
import os
import datetime
from dgllife.utils import one_hot_encoding
from dgllife.utils import SMILESToBigraph, ScaffoldSplitter, RandomSplitter
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, auc, mean_absolute_error, r2_score


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子

    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def split_dataset(dataset, split_type=None, train_ratio=None, val_ratio=None, test_ratio=None, random_state=42):
    """Split the dataset
    Return: train_set---Training subset
            val_set---Validation subset
            test_set---Test subset
    其中，'scaffold_decompose' using rdkit.Chem.AllChem.MurckoDecompose
         'scaffold_smiles' using rdkit.Chem.Scaffolds.MurckoScaffold.MurckoScaffoldSmiles
    """
    if split_type == 'scaffold_decompose':
        train_set, val_set, test_set = ScaffoldSplitter.train_val_test_split(
            dataset, frac_train=train_ratio, frac_val=val_ratio, frac_test=test_ratio,
            scaffold_func='decompose')

    elif split_type == 'scaffold_smiles':
        train_set, val_set, test_set = ScaffoldSplitter.train_val_test_split(
            dataset, frac_train=train_ratio, frac_val=val_ratio, frac_test=test_ratio,
            scaffold_func='smiles')

    elif split_type == 'random':
        train_set, val_set, test_set = RandomSplitter.train_val_test_split(
            dataset, frac_train=train_ratio, frac_val=val_ratio, frac_test=test_ratio,
            random_state=random_state)
    else:
        return ValueError("Expect the splitting method from ['random'、'scaffold_decompose'、'scaffold_smiles'], "
                          "got {}".format(split_type))

    return train_set, val_set, test_set


def chirality(atom):
    """Get Chirality information for an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list of 3 boolean values
        这3个布尔值分别表示原子是否有一个手性标签R、原子是否有一个手性标签S和原子是否是一个可能的手性中心
    """
    try:
        return one_hot_encoding(atom.GetProp('_CIPCode'), ['R', 'S']) + \
               [atom.HasProp('_ChiralityPossible')]
    except:
        return [False, False] + [atom.HasProp('_ChiralityPossible')]

def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader.

    Parameters
    ----------
    data : list of 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and a binary
        mask indicating the existence of labels.

    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels.
    """
    assert len(data[0]) in [3, 4], \
        'Expect the tuple to be of length 3 or 4, got {:d}'.format(len(data[0]))
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if len(data[0]) == 3:
        # mask is None
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)

    return smiles, bg, labels, masks


class Meter(object):
    """Track and summarize model performance on a dataset for (multi-label) binary classification."""
    def __init__(self):
        self.mask = []
        self.y_pred = []
        self.y_true = []

    def update(self, y_pred, y_true, mask):
        """Update for the result of an iteration

        Parameters
        ----------
        y_pred : float32 tensor
            Predicted molecule labels with shape (B, T),
            B for batch size and T for the number of tasks
        y_true : float32 tensor
            Ground truth molecule labels with shape (B, T)
        mask : float32 tensor
            Mask for indicating the existence of ground
            truth labels with shape (B, T)
        """

        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())
        self.mask.append(mask.detach().cpu())

    def roc_precision_recall_score(self):
        """Compute AUC_PRC for each task.
        Returns
        -------
        list of float
            rmse for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_pred = torch.sigmoid(y_pred)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            precision, recall, _thresholds = precision_recall_curve(task_y_true, task_y_pred, pos_label=1)
            scores.append(auc(recall, precision))
        return scores

    def roc_auc_score(self):
        """Compute roc-auc score for each task.

        Returns
        -------
        list of float
            roc-auc score for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # Todo: support categorical classes
        # This assumes binary case only
        y_pred = torch.sigmoid(y_pred)  # 求得为正例的概率

        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()

            scores.append(roc_auc_score(task_y_true, task_y_pred))
        return scores

    def l1_loss(self, reduction):
        """Compute l1 loss for each task.

        Returns
        -------
        list of float
            l1 loss for all tasks
        reduction : str
            * 'mean': average the metric over all labeled data points for each task
            * 'sum': sum the metric over all labeled data points for each task
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            scores.append(F.l1_loss(task_y_true, task_y_pred, reduction=reduction).item())
        return scores

    def rmse(self):
        """Compute RMSE for each task.

        Returns
        -------
        list of float
            rmse for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            scores.append(np.sqrt(F.mse_loss(task_y_pred, task_y_true).cpu().item()))
        return scores

    def mae(self):
        """Compute mae for each task.

        Returns
        -------
        list of float
            mae for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            scores.append(mean_absolute_error(task_y_true, task_y_pred))
        return scores

    def se(self):
        """Compute se score for each task.

                Returns
                -------
                list of float
                    se score for all tasks
                """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # Todo: support categorical classes
        # This assumes binary case only
        y_pred = torch.sigmoid(y_pred)  # 求得为正例的概率
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()

            task_y_pred = [1 if i >= 0.5 else 0 for i in task_y_pred]
            c_mat = confusion_matrix(task_y_true, task_y_pred)

            tn, fp, fn, tp = list(c_mat.flatten())
            se = tp / (tp + fn)
            scores.append(se)

        return scores
    def precision(self):
        """Compute precision score for each task.

                Returns
                -------
                list of float
                    precision score for all tasks
                """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # Todo: support categorical classes
        # This assumes binary case only
        y_pred = torch.sigmoid(y_pred)  # 求得为正例的概率
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()

            task_y_pred = [1 if i >=0.5 else 0 for i in task_y_pred]
            c_mat = confusion_matrix(task_y_true, task_y_pred)

            tn, fp, fn, tp = list(c_mat.flatten())
            precision = tp / (tp + fp)
            scores.append(precision)

        return scores
    def sp(self):
        """Compute sp score for each task.

                Returns
                -------
                list of float
                    sp score for all tasks
                """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # Todo: support categorical classes
        # This assumes binary case only
        y_pred = torch.sigmoid(y_pred)  # 求得为正例的概率
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()

            task_y_pred = [1 if i >=0.5 else 0 for i in task_y_pred]
            c_mat = confusion_matrix(task_y_true, task_y_pred)

            tn, fp, fn, tp = list(c_mat.flatten())

            sp = tn / (tn + fp)

            scores.append(sp)

        return scores
    def acc(self):
        """Compute acc score for each task.

                Returns
                -------
                list of float
                    acc score for all tasks
                """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # Todo: support categorical classes
        # This assumes binary case only
        y_pred = torch.sigmoid(y_pred)  # 求得为正例的概率
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()

            task_y_pred = [1 if i >=0.5 else 0 for i in task_y_pred]
            c_mat = confusion_matrix(task_y_true, task_y_pred)

            tn, fp, fn, tp = list(c_mat.flatten())

            acc = (tp + tn) / (tn + fp + fn + tp)

            scores.append(acc)

        return scores
    def mcc(self):
        """Compute mcc score for each task.

                Returns
                -------
                list of float
                    mcc score for all tasks
                """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # Todo: support categorical classes
        # This assumes binary case only
        y_pred = torch.sigmoid(y_pred)  # 求得为正例的概率
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            task_y_pred = [1 if i >=0.5 else 0 for i in task_y_pred]
            c_mat = confusion_matrix(task_y_true, task_y_pred)
            tn, fp, fn, tp = list(c_mat.flatten())
            mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8)
            scores.append(mcc)

        return scores
    def r2(self):
        """Compute r2 score for each task.

        Returns
        -------
        list of float
            r2 score for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            scores.append(r2_score(task_y_true, task_y_pred))
        return scores

    def pred(self):

        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)

        # y_pred = torch.sigmoid(y_pred)  # 求得为正例的概率

        return y_pred
    def compute_metric(self, metric_name, reduction='mean'):
        """Compute metric for each task.

        Parameters
        ----------
        metric_name : str
            Name for the metric to compute.
        reduction : str
            Only comes into effect when the metric_name is l1_loss.
            * 'mean': average the metric over all labeled data points for each task
            * 'sum': sum the metric over all labeled data points for each task

        Returns
        -------
        list of float
            Metric value for each task
        """
        assert metric_name in ['roc_auc', 'l1', 'rmse', 'prc_auc', 'mae', 'r2','se','sp','acc','mcc','pred','precision'], \
            'Expect metric name to be "roc_auc", "l1", "rmse", "prc_auc", "mae", "r2","pred" got {}'.format(metric_name)  # assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常
        assert reduction in ['mean', 'sum']
        if metric_name == 'roc_auc':
            return self.roc_auc_score()
        if metric_name == 'l1':
            return self.l1_loss(reduction)
        if metric_name == 'rmse':
            return self.rmse()
        if metric_name == 'prc_auc':
            return self.roc_precision_recall_score()
        if metric_name == 'se':
            return self.se()

        if metric_name == 'precision':
            return self.precision()
        if metric_name == 'sp':
            return self.sp()
        if metric_name == 'acc':
            return self.acc()
        if metric_name == 'mcc':
            return self.mcc()
        if metric_name == 'mae':
            return self.mae()
        if metric_name == 'r2':
            return self.r2()
        if metric_name == 'mcc':
            return self.mcc()
        if metric_name == 'pred':
            return self.pred()

class EarlyStopping(object):
    """Early stop performing

    Parameters
    ----------
    mode : str
        * 'higher': Higher metric suggests a better model
        * 'lower': Lower metric suggests a better model
    patience : int
        Number of epochs to wait before early stop if the metric stops getting improved
    filename : str or None
        Filename for storing the model checkpoint
        """
    def __init__(self, mode='higher', patience=10, filename=None):
        if filename is None:
            dt = datetime.datetime.now()
            filename = '{}_early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(dt.date(), dt.hour, dt.minute, dt.second)

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        return (score > prev_best_score)

    def _check_lower(self, score, prev_best_score):
        return (score < prev_best_score)

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):  # 当前模型如果是更优模型，则保存当前模型
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        """Save model when the metric on the validation set gets improved."""
        torch.save({'model_state_dict': model.state_dict()}, self.filename)

    def load_checkpoint(self, model):
        """Load model saved with early stopping"""
        model.load_state_dict(torch.load(self.filename)['model_state_dict'])


