import torch
import numpy as np
import pandas as pd
import time
import argparse
import matplotlib.pyplot as plt
import pickle

from utils import print_results
from experiment import Experiment
from data_loaders import load_dataset

torch.cuda.is_available() #check cuda

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-num_experiments", type=int, default=1, nargs="?", help="Number of iterations.")
parser.add_argument("-epochs", type=int, default=10000, nargs="?", help="Number of iterations.")
parser.add_argument("-patience", type=int, default=50, nargs="?",help="patience for early stop.")
parser.add_argument("-batch_size", type=int, default=4096, nargs="?", help="Batch size.")
parser.add_argument("-lr", type=float, default=3e-4, nargs="?", help="Learning rate.")
parser.add_argument("-core_shape", type=str, default='20,20,20', help="shape of the core tensor in last layer")
parser.add_argument("-core2_shape", type=str, default='10,10,10', help="shape of the core tensor in last layer")
parser.add_argument("-core3_shape", type=str, default='5,5,5', help="shape of the core tensor in last layer")
parser.add_argument("-device", type=str, default='cuda', help="cuda or cpu")
parser.add_argument("-val_ratio", type=float, default=0.1, nargs="?", help="validation split ratio")
parser.add_argument("-model", type=str, default='ML1', choices=["ML1", "ML2", "ML3"])
parser.add_argument("-dataset", type=str, default='fun', help="dataset name", choices=['fun', 'eps', 'chars'])
parser.add_argument("-lambda_l1", type=float, default=0, help="strength of L1 regularization")
parser.add_argument("-lambda_l2", type=float, default=0, help="strength of L2 regularization")
parser.add_argument("-regularization", type=str, default=None, help="L1 or L2")

args = parser.parse_args()

args.core_shape = list(map(int, args.core_shape.split(',')))
args.core2_shape = list(map(int, args.core2_shape.split(',')))
args.core3_shape = list(map(int, args.core3_shape.split(',')))
print(vars(args))

# For running multiple experiments with different seed
num_experiments = 1
metric_dict_train = {'rmse':[], 'mae':[], 'mape':[]}
metric_dict_test = {'rmse':[], 'mae':[], 'mape':[]}

for i in range(args.num_experiments):
    print('\n\n\n\n\n\n')
    print(f'{args.dataset}, EXPERIMENT {i+1}, core_shape={args.core_shape}, core2_shape={args.core2_shape}, core3_shape={args.core3_shape}')

    # load dataset
    tr_idxs, tr_vals, te_idxs, te_vals, data_shape = load_dataset(args.dataset, seed=i)

    ex = Experiment(num_iterations = args.epochs,
                    batch_size = args.batch_size,
                    learning_rate = args.lr, 
                    data_shape = data_shape,
                    core_shape = tuple(args.core_shape),
                    core2_shape = tuple(args.core2_shape),
                    device = args.device,
                    model = args.model,
                    patience = args.patience,
                    tr_idxs = tr_idxs,
                    tr_vals = tr_vals,
                    val_ratio = args.val_ratio,
                    regularization = args.regularization,
                    lambda_l1 = args.lambda_l1,
                    lambda_l2 = args.lambda_l2)

    dic = ex.train_and_eval()

    _, mse, rmse, mae, mape = ex.evaluate(dic['model'], tr_idxs, tr_vals)
    metric_dict_train['rmse'].append(rmse)
    metric_dict_train['mae'].append(mae)
    metric_dict_train['mape'].append(mape)
    
    _, mse, rmse, mae, mape = ex.evaluate(dic['model'], te_idxs, te_vals)
    metric_dict_test['rmse'].append(rmse)
    metric_dict_test['mae'].append(mae)
    metric_dict_test['mape'].append(mape)

print_results(metric_dict_train, metric_dict_test)
