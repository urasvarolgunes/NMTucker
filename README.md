# NMTucker: Non-linear Matryoshka Tucker Decomposition for Financial Time Series Imputation (ICAIF '23)

This repository contains the official PyTorch implementation of NMTucker.

Paper: https://dl.acm.org/doi/abs/10.1145/3604237.3626909

<br />

### Usage
- NMTucker1 example
```
python nmtucker_eval.py -dataset=fun -core_shape=40,40,40
```

- NMTucker2 example
```
python nmtucker_eval.py -model=ML2 -dataset=fun -core_shape=80,80,80 -core2_shape=36,36,36
```

- NMTucker3 example
```
python nmtucker_eval.py -model=ML3 -dataset=fun -core_shape=70,70,70 -core2_shape=50,50,50 -core3_shape=35,35,35
```

- NMTucker-L1 example
```
python nmtucker_eval.py -model=ML1 -dataset=fun -core_shape=40,40,40 -regularization=L1 -lambda_l1=1e-6
```

### Requirements:
- pytorch == 1.13.1


<br />

If you find this work useful, please cite our paper:

```
@inproceedings{10.1145/3604237.3626909,
author = {Varolgunes, Uras and Zhou, Dan and Yu, Dantong and Uddin, Ajim},
title = {NMTucker: Non-linear Matryoshka Tucker Decomposition for Financial Time Series Imputation},
year = {2023},
isbn = {9798400702402},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3604237.3626909},
doi = {10.1145/3604237.3626909},
booktitle = {Proceedings of the Fourth ACM International Conference on AI in Finance},
pages = {516–523},
series = {ICAIF '23}
}
```
