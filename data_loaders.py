import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(name, seed):
    print(f'loading {name} dataset..')
    
    if name == 'fun':
        df = pd.read_csv('data/fundamentals.csv')
        dtrain, dtest = train_test_split(df, test_size=0.2, random_state=seed)
        shape = (32, 180, 19)
    
    elif name == 'eps':
        df = pd.read_csv('data/eps.csv')
        dtrain, dtest = train_test_split(df, test_size=0.2, random_state=seed)
        shape = (32, 180, 173)

    elif name == 'chars':
        df = pd.read_csv('data/chars.csv')
        dtrain, dtest = train_test_split(df, test_size=0.2, random_state=seed)
        shape = (60, 100, 202)

    tr_idxs = dtrain.values[:, 0:3]
    tr_vals = dtrain.values[:, -1]
    te_idxs = dtest.values[:, 0:3]
    te_vals = dtest.values[:, -1]

    return tr_idxs, tr_vals, te_idxs, te_vals, shape