# NMTucker: Non-linear Matryoshka Tucker Decomposition for Financial Time Series Imputation

This repository is the official PyTorch implementation of NMTucker.

NMTucker1 example
```
python nmtucker_eval.py -core_shape=40,40,40 -num_experiments=2 -dataset=fun
```

NMTucker2 example
```
python nmtucker_eval.py -model=ML2 -dataset=chars -core_shape=80,80,80 -core2_shape=36,36,36
```

NMTucker3 example
```
python nmtucker_eval.py -model=ML3 -dataset=fun -core_shape=70,70,70 -core2_shape=50,50,50 -core3_shape=35,35,35
```

NMTucker-L1 example
```
python nmtucker_eval.py -model=ML1 -dataset=fun -core_shape=40,40,40 -regularization=L1 -lambda_l1=1e-6
```

Requirements:

pytorch == 1.13.1
