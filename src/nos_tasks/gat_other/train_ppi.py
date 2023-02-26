"""
Graph Attention Networks (PPI Dataset) in DGL using SPMV optimization.
Multiple heads are also batched together for faster training.
Compared with the original paper, this code implements
early stopping.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from sklearn.metrics import f1_score
from .gat import GAT
from dgl.data.ppi import PPIDataset
from dgl.dataloading import GraphDataLoader

from utils.logging import log_and_print, print_program


def evaluate(feats, model, subgraph, labels, loss_fcn):
    with torch.no_grad():
        model.eval()
        model.g = subgraph
        for layer in model.gat_layers:
            layer.g = subgraph
        output = model(feats.float())
        loss_data = loss_fcn(output, labels.float())
        predict = np.where(output.data.cpu().numpy() >= 0., 1, 0)
        score = f1_score(labels.data.cpu().numpy(),
                         predict, average='micro')
        return score, loss_data.item()


def main(args, opt_func, terminator, device, fast, verbose=False):
    batch_size = args.batch_size
    cur_step = 0
    patience = args.patience
    best_score = -1
    best_loss = 10000
    # define loss function
    loss_fcn = torch.nn.BCEWithLogitsLoss()
    # create the dataset
    data_path = '/nfs/data/ruocwang/projects/crash/automl/comp-graph-search/data/gnn'
    train_dataset = PPIDataset(raw_dir=data_path, mode='train', verbose=False)
    valid_dataset = PPIDataset(raw_dir=data_path, mode='valid', verbose=False)
    test_dataset  = PPIDataset(raw_dir=data_path, mode='test', verbose=False)
    train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size)
    valid_dataloader = GraphDataLoader(valid_dataset, batch_size=batch_size)
    test_dataloader  = GraphDataLoader(test_dataset, batch_size=batch_size)
    g = train_dataset[0]
    n_classes = train_dataset.num_labels
    num_feats = g.ndata['feat'].shape[1]
    g = g.int().to(device)
    heads = ([args.num_heads] * (args.num_layers-1)) + [args.num_out_heads]
    # define the model
    model = GAT(g,
                args.num_layers,
                num_feats,
                args.num_hidden,
                n_classes,
                heads,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.alpha,
                args.residual)
    
    # define the optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = opt_func(model.parameters())


    model = model.to(device)
    best_model_state_dict = None
    for epoch in range(args.num_epochs):
        model.train()
        loss_list = []
        for batch, subgraph in enumerate(train_dataloader):
            subgraph = subgraph.to(device)
            model.g = subgraph
            for layer in model.gat_layers:
                layer.g = subgraph
            logits = model(subgraph.ndata['feat'].float())
            loss = loss_fcn(logits, subgraph.ndata['label'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        loss_data = np.array(loss_list).mean()
        # log_and_print("Epoch {:05d} | Loss: {:.4f}".format(epoch + 1, loss_data))
        if epoch % 5 == 0:
            score_list = []
            val_loss_list = []
            for batch, subgraph in enumerate(valid_dataloader):
                subgraph = subgraph.to(device)
                score, val_loss = evaluate(subgraph.ndata['feat'], model, subgraph, subgraph.ndata['label'], loss_fcn)
                score_list.append(score)
                val_loss_list.append(val_loss)
            mean_score = np.array(score_list).mean()
            mean_val_loss = np.array(val_loss_list).mean()
            if verbose: log_and_print("Val F1-Score: {:.4f} ".format(mean_score))
            # early stop
            if mean_score > best_score or best_loss > mean_val_loss:
                if mean_score > best_score and best_loss > mean_val_loss:
                    val_early_loss = mean_val_loss
                    val_early_score = mean_score
                best_model_state_dict = deepcopy(model.state_dict())
                best_score = np.max((mean_score, best_score))
                best_loss = np.min((best_loss, mean_val_loss))
                cur_step = 0
            else:
                cur_step += 1
                if cur_step == patience:
                    break
    
    #### testing on the best model
    model.load_state_dict(best_model_state_dict)
    test_score_list = []
    for batch, subgraph in enumerate(test_dataloader):
        subgraph = subgraph.to(device)
        score, test_loss = evaluate(subgraph.ndata['feat'], model, subgraph, subgraph.ndata['label'], loss_fcn)
        test_score_list.append(score)
    best_test_score = np.array(test_score_list).mean()
    if verbose: log_and_print("Test F1-Score: {:.4f}".format(best_test_score))


    ## TODO under dev ##
    status = 'finished'
    log_str = ''
    best_valid_acc = best_score
    best_test_acc  = best_test_score
    return best_valid_acc, best_test_acc, status, epoch, log_str



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--num-epochs", type=int, default=400,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=6,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=3,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=256,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=True,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=0,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0,
                        help="weight decay")
    parser.add_argument('--alpha', type=float, default=0.2,
                        help="the negative slop of leaky relu")
    parser.add_argument('--batch-size', type=int, default=2,
                        help="batch size used for training, validation and test")
    parser.add_argument('--patience', type=int, default=10,
                        help="used for early stop")
    args = parser.parse_args()
    print(args)

    main(args)