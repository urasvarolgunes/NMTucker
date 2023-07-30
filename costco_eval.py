import tensorflow.compat.v1 as tf
import keras as k
import pandas as pd
from sklearn.model_selection import train_test_split
from costco_utils import get_metrics, set_session, create_costco, transform, mape_keras, print_results, print_num_params
from data_loaders import load_dataset
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-num_experiments", type=int, default=1, nargs="?", help="Number of iterations.")
parser.add_argument("-rank", type=int, default=20, help= "rank")
parser.add_argument("-patience", type=int, default=20, help= "early stop patience")
parser.add_argument("-dataset", type=str, default='fun', help="dataset name", choices=['fun', 'eps', 'chars'])
args = parser.parse_args()


# For running multiple experiments with different seed
num_experiments = 1
metric_dict_train = {'rmse':[], 'mae':[], 'mape':[]}
metric_dict_test = {'rmse':[], 'mae':[], 'mape':[]}

for i in range(args.num_experiments):
    print('\n\n\n\n\n')
    print(f'RANK {args.rank}, EXPERIMENT {i+1}')
    # load data
    tr_idxs, tr_vals, te_idxs, te_vals, shape = load_dataset(args.dataset, seed=i)
    
    lr = 3e-4
    epochs = 10000
    batch_size = 4096
    seed = i
    verbose = 1
    rank = args.rank
    nc = rank
    set_session(device_count={"GPU": 0}, seed=seed)
    optim = k.optimizers.Adam(lr=lr)

    model = create_costco(shape, rank, nc)

    print_num_params()

    model.compile(optim, loss=["mse"], metrics=["mae", mape_keras])
    hists = model.fit(
        x=transform(tr_idxs),
        y=tr_vals,
        verbose=verbose,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[k.callbacks.EarlyStopping(
        monitor="val_mean_absolute_error", 
        patience=args.patience, 
        restore_best_weights=True)],
        )

    train_metrics = get_metrics(model, transform(tr_idxs), tr_vals)
    for metric_name, val in train_metrics.items():
        metric_dict_train[metric_name].append(val)
    
    test_metrics = get_metrics(model, transform(te_idxs), te_vals)
    for metric_name, val in test_metrics.items():
        metric_dict_test[metric_name].append(val)
    
    # results for current experiment
    print({'train': train_metrics, 'test': test_metrics})

# final summary of all experiments
print_results(metric_dict_train, metric_dict_test)
    
