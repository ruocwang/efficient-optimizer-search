"""
Graph Attention Networks in DGL using SPMV optimization.
Multiple heads are also batched together for faster training.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import argparse
import numpy as np
import networkx as nx
import time
import torch
import torch.nn.functional as F
import dgl
from dgl.data import register_data_args
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset

from .gat import GAT
from .utils import EarlyStopping

from utils.logging import log_and_print, print_program


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels)


def main(args, opt_func, terminator, device, fast=False, verbose=True):
    if fast:
        acc = np.random.rand() * 0.4
        return acc, acc, 'finished', 10, ''
    # load and preprocess dataset
    data_path = '/nfs/data/ruocwang/projects/crash/automl/comp-graph-search/data/gnn'
    if args.dataset == 'cora':
        assert False, "warning, be mindful not to override existing cora dataset from the other codebase"
        data = CoraGraphDataset(raw_dir=data_path, verbose=False)
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset(raw_dir=data_path, verbose=False)
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset(raw_dir=data_path, verbose=False)
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    g = data[0]
    g = g.int().to(device)

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    num_feats = features.shape[1]
    n_classes = data.num_classes
    n_edges = data.graph.number_of_edges()

    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()
    # create model
    heads = ([args.num_heads] * (args.num_layers-1)) + [args.num_out_heads]
    model = GAT(g,
                args.num_layers,
                num_feats,
                args.num_hidden,
                n_classes,
                heads,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.negative_slope,
                args.residual)

    if args.early_stop:
        stopper = EarlyStopping(patience=args.patience)
    model.to(device)
    loss_fcn = torch.nn.CrossEntropyLoss()

    # # use optimizer
    optimizer = opt_func(model.parameters())
    # optimizer = torch.optim.Adam(
    #     model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.num_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        train_acc = accuracy(logits[train_mask], labels[train_mask])

        if args.fastmode:
            val_acc = accuracy(logits[val_mask], labels[val_mask])
        else:
            val_acc = evaluate(model, features, labels, val_mask)
            if args.early_stop:
                if stopper.step(val_acc, model):
                    break

        if verbose: log_and_print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"" ValAcc {:.4f} | ETputs(KTEPS) {:.2f}".
                                  format(epoch, np.mean(dur), loss.item(), train_acc, val_acc, n_edges / np.mean(dur) / 1000))

    if args.early_stop:
        model.load_state_dict(stopper.saved_ckpt)
    best_valid_acc = stopper.best_score
    best_test_acc  = evaluate(model, features, labels, test_mask)
    if verbose: log_and_print("Test Accuracy {:.4f}".format(best_test_acc))

    status = 'finished'
    log_str = ''
    return best_valid_acc, best_test_acc, status, epoch, log_str


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--num-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=8,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    args = parser.parse_args()
    print(args)

    main(args)