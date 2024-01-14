
from texttable import Texttable
from sklearn import metrics
import torch.nn as nn
import torch.nn.functional as F

def tab_printer(args):
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())


def get_evaluation_results(labels_true, labels_pred):
    ACC = metrics.accuracy_score(labels_true, labels_pred)
    P = metrics.precision_score(labels_true, labels_pred, average='macro')
    R = metrics.recall_score(labels_true, labels_pred, average='macro')
    F1 = metrics.f1_score(labels_true, labels_pred, average='macro')

    return ACC, P, R, F1

class MLP(nn.Module):
    def __init__(self, input_d, hidden_d, output_d, num_layers, dropout, norm, init_activate):
        super().__init__()

        self.init_activate = init_activate
        self.norm = norm
        self.dropout = dropout
        self.layers = nn.ModuleList([])
        self.num_layers = num_layers

        if num_layers == 1:
            self.layers.append(nn.Linear(input_d, output_d))
        elif num_layers > 1:
            self.layers.append(nn.Linear(input_d, hidden_d))
            for k in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_d, hidden_d))
            self.layers.append(nn.Linear(hidden_d, output_d))

        self.norm_cnt = num_layers - 1 + int(init_activate)  # how many norm layers we have
        if norm == "batch":
            self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_d) for _ in range(self.norm_cnt)])
        elif norm == "layer":
            self.norms = nn.ModuleList([nn.LayerNorm(hidden_d) for _ in range(self.norm_cnt)])

        self.reset_params()

    def reset_params(self):
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight.data)
            # nn.init.constant_(layer.bias.data, 0)


    def activate(self, x):
        if self.norm != "none":
            x = self.norms[self.cur_norm_idx](x)  # use the last norm layer
            self.cur_norm_idx += 1
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        return x

    def forward(self, x):
        self.cur_norm_idx = 0

        if self.init_activate:
            x = self.activate(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.num_layers == 1:
                x = F.relu(x)
            if i != len(self.layers) - 1:  # do not activate in the last layer
                x = self.activate(x)

        return x