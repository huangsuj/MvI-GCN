
import os
import pdb
import time
import random
import torch
import numpy as np
import scipy.io as sio
import scipy.sparse as ss
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from utils import *
import torch.nn.functional as F


def load_data(args, device, j, r):
    data = sio.loadmat(args.path + args.dataset + '.mat')
    features = data['X']
    concat_feature = torch.Tensor().to(device)
    adj_list = []
    labels = data['Y'].flatten()
    labels = labels - min(set(labels))
    idx_labeled, idx_unlabeled = generate_partition(labels=labels, ratio=args.ratio, seed=args.shuffle_seed, r=r)
    labels = torch.from_numpy(labels).long()
    num_classes = len(np.unique(labels))

    labels_one_hot = torch.eye(num_classes)[labels, :]
    labels_one_hot_mask = torch.zeros_like(labels_one_hot)
    labels_one_hot_mask[idx_labeled, :] = labels_one_hot[idx_labeled, :]
    IN = torch.eye(features[0][0].shape[0]).to(device)

    for i in range(features.shape[1]):
        feature = features[0][i].astype(float)
        ID = torch.eye(features[0][i].shape[1]).to(device)
        lada = args.alpha[j]
        begin_time = time.time()
        feature = torch.from_numpy(feature).float().to(device)
        feature = F.normalize(feature, p=2.0, dim=1)
        A = F.relu(feature @ feature.t())
        A = A - torch.diag_embed(torch.diag(A)) + torch.eye(feature.shape[0]).to(device)
        degree_ = A.sum(1)
        dgree_inv1 = torch.pow(degree_, -0.5)
        dgree_inv11 = torch.diag(dgree_inv1)

        U = lada * (dgree_inv11 @ feature)
        V = feature.t() @ dgree_inv11
        adj = IN + U @ (ID - V @ U).inverse() @ V

        cost_time = time.time() - begin_time
        print("#########",cost_time)
        adj_list.append(adj)
        feature = feature.to(device)
        concat_feature = torch.cat((concat_feature, feature), dim=1)
    concat_feature = concat_feature.to(device)

    hidden_dims = [concat_feature.shape[1]] + args.hdim + [num_classes]
    print('the number of sample:' + str(concat_feature.shape[0]))

    return adj_list, concat_feature, labels, idx_labeled, idx_unlabeled, hidden_dims


def generate_partition(labels, ratio, seed, r):
    each_class_num = count_each_class_num(labels)
    labeled_each_class_num = {} ## number of labeled samples for each class
    total_num = round(ratio[r] * len(labels))
    for label in each_class_num.keys():
        labeled_each_class_num[label] = max(round(each_class_num[label] * ratio[r]), 1) # min is 1

    # index of labeled and unlabeled samples
    p_labeled = []
    p_unlabeled = []
    index = [i for i in range(len(labels))]
    # print(index)
    if seed >= 0:
        random.seed(seed)
        random.shuffle(index)
    labels = labels[index]
    for idx, label in enumerate(labels):
        if (labeled_each_class_num[label] > 0):
            labeled_each_class_num[label] -= 1
            p_labeled.append(index[idx])
            total_num -= 1
        else:
            p_unlabeled.append(index[idx])
    return p_labeled, p_unlabeled


def count_each_class_num(labels):
    '''
        Count the number of samples in each class
    '''
    count_dict = {}
    for label in labels:
        if label in count_dict.keys():
            count_dict[label] += 1
        else:
            count_dict[label] = 1
    return count_dict