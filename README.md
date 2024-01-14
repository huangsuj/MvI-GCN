# SAFL-GCN
Source code of MvI-GCN, which is proposed in this paper:
MvI-GCN: Efficient Multi-view Infinite-order Graph Convolutional Network.

### Requirement
- Python == 3.9.12
- PyTorch == 1.11.0
- Numpy == 1.21.5
- Scikit-learn == 1.1.0
- Scipy == 1.8.0
- Texttable == 1.6.4
- Tensorly == 0.7.0
- Tqdm == 4.64.0

### Usage
    python args.py
- --device: gpu num or cpu.
- --path: path of datasets.
- --dataset: name of datasets.
- --seed: random seed
- --fix_seed: fix the seed.
- --n_repeated: number of repeated times.
- --lr: learning rate.
- --weight_decay: weight decay.
- --ratio: ratio of labeled samples
- --num_epoch: number of training epochs
- --alpha: the hyper-parameters according to this paper.
- --hdim: number of hidden dimensions.


### Datasets Descriptions
- Original datasets are stored in **/datasets**.

### Quick Running
run `python train.py`
