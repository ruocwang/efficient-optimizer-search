from __future__ import division
from __future__ import print_function
from copy import deepcopy

import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .utils import load_data, accuracy
except:
    from utils import load_data, accuracy
try:
    from .models import GAT, SpGAT
except:
    from models import GAT, SpGAT
try:
    from utils.logging import log_and_print, print_program
except:
    log_and_print = print



def train(epoch, model, optimizer, features, adj, labels, idx_train, idx_val, verbose=False):
    ## eval on training set
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    ## eval on validation set
    # Evaluate validation set performance separately,
    # deactivates dropout during validation run.
    model.eval()
    output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    if verbose and epoch % 100 == 0:
        log_and_print('Epoch: {:04d}'.format(epoch+1) + ' '
                      'loss_train: {:.4f}'.format(loss_train.data.item()) + ' '
                      'acc_train: {:.4f}'.format(acc_train.data.item()) + ' '
                      'loss_val: {:.4f}'.format(loss_val.data.item()) + ' '
                      'acc_val: {:.4f}'.format(acc_val.data.item()) + ' '
                      'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item(), acc_val.item()


def compute_test(model, features, adj, labels, idx_test, verbose=False):
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    # if verbose:
    #     log_and_print("Test set results:" + ' '
    #                   "loss= {:.4f}".format(loss_test.data.item()) + ' '
    #                   "accuracy= {:.4f}".format(acc_test.data.item()))
    
    return acc_test.item()

# Train model
def main(args, opt_func, terminator, device, fast=False, verbose=True):
    ## loading everything up
    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data(verbose=False)

    # Model and optimizer
    if args.sparse:
        model = SpGAT(nfeat=features.shape[1], 
                    nhid=args.hidden, 
                    nclass=int(labels.max()) + 1, 
                    dropout=args.dropout, 
                    nheads=args.nb_heads, 
                    alpha=args.alpha)
    else:
        model = GAT(nfeat=features.shape[1], 
                    nhid=args.hidden, 
                    nclass=int(labels.max()) + 1, 
                    dropout=args.dropout, 
                    nheads=args.nb_heads, 
                    alpha=args.alpha)
    model.cuda()

    # optimizer = optim.Adam(model.parameters(), 
    #                        lr=args.lr, 
    #                        weight_decay=args.weight_decay)
    optimizer = opt_func(model.parameters())

    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

    
    ## training
    t_total = time.time()
    bad_counter = 0
    best_acc_val = 0
    best_loss_val = None
    best_epoch = 0
    # best_model_state = None
    for epoch in range(args.num_epochs):
        loss_value, acc_val = train(epoch, model, optimizer, features, adj, labels, idx_train, idx_val, verbose=verbose)

        if best_loss_val is None or loss_value < best_loss_val: ## better loss -> original impls, much better for all optimizers
        # if acc_val > best_acc_val: ## better valid acc
            best_acc_val = acc_val
            best_loss_val = loss_value
            best_epoch = epoch
            # best_model_state = deepcopy(model.state_dict())
            best_acc_test = compute_test(model, features, adj, labels, idx_test, verbose=verbose)

            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

    if verbose:
        log_and_print("Optimization Finished!")
        log_and_print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Restore best model
    if verbose:
        log_and_print('Loading {}th epoch'.format(best_epoch))
    # model.load_state_dict(best_model_state)

    # Testing
    # best_acc_test = compute_test(model, features, adj, labels, idx_test, verbose=verbose)

    return deepcopy(best_acc_val), deepcopy(best_acc_test), 'finished', deepcopy(epoch), ''









if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    parser.add_argument('--num_epochs', type=int, default=10000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
    parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=100, help='Patience')

    args = parser.parse_args()

    main(args, None, None, None, None, verbose=True)