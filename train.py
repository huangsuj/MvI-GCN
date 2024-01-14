"""
@Project   : HGCN-Net
@Time      : 2021/11/7
@Author    : Zhihao Wu
@File      : train.py
"""
import os
import warnings
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from args import parameter_parser
from utils import *
from Dataloader import load_data
from model import MvGCN
def train(args, device, j, r):
    begin_time = time.time()
    adj_list, co_feature, labels, idx_labeled, idx_unlabeled, hidden_dims = load_data(args, device, j, r)

    n = co_feature.shape[0]
    num_classes = len(np.unique(labels))
    labels = labels.to(device)
    num_view = len(adj_list)

    total_para = 0

    co_feature = F.normalize(co_feature, 1)
    model = MvGCN(n, num_classes, num_view, hidden_dims, args.dropout, adj_list, device).to(device)
    total_para += sum(x.numel() for x in model.parameters())
    print("Total number of paramerters in networks is {}  ".format(total_para / 1e6))

    loss_function1 = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    Loss_list = []
    ACC_list = []
    F1_list = []

    begin_time = time.time()
    with tqdm(total=args.num_epoch, desc="Training") as pbar:
        for epoch in range(args.num_epoch):
            model.train()
            output = model(co_feature, adj_list)
            output = F.log_softmax(output, dim=1)
            optimizer.zero_grad()
            loss = loss_function1(output[idx_labeled], labels[idx_labeled])
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                output = model(co_feature, adj_list)
                pred_labels = torch.argmax(output, 1).cpu().detach().numpy()
                ACC, P, R, F1 = get_evaluation_results(labels.cpu().detach().numpy()[idx_unlabeled], pred_labels[idx_unlabeled])
                pbar.set_postfix({'Loss': '{:.4f}'.format(loss.item()),
                                  'ACC': '{:.2f}'.format(ACC * 100), 'F1': '{:.2f}'.format(F1 * 100)})
                pbar.update(1)
                Loss_list.append(float(loss.item()))
                ACC_list.append(ACC)
                F1_list.append(F1)
    cost_time = time.time() - begin_time
    model.eval()
    output = model(co_feature, adj_list)
    print("Evaluating the model")
    pred_labels = torch.argmax(output, 1).data.cpu().numpy()

    ACC, P, R, F1 = get_evaluation_results(labels.cpu().detach().numpy()[idx_unlabeled], pred_labels[idx_unlabeled])
    print("------------------------")
    print("ACC:   {:.2f}".format(ACC * 100))
    print("P:   {:.2f}".format(P * 100))
    print("R:   {:.2f}".format(R * 100))
    print("F1 :   {:.2f}".format(F1 * 100))
    print("Running Time {:.4f}".format(cost_time))
    print("------------------------")

    return ACC, P, R, F1, cost_time, Loss_list, ACC_list, F1_list


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = parameter_parser()
    device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)
    if args.fix_seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    tab_printer(args)
    for j in range(len(args.alpha)):
        for r in range(len(args.ratio)):
            train(args, device, j, r)

