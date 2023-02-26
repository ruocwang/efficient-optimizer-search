from syslog import LOG_INFO
import torch
import torch.nn as nn


class NoneScheduler():
    def __init__(self, optimizer):
        self.optimizer = optimizer
    
    def step(self):
        return None


class QuadraticLoss():
    def __init__(self):
        pass

    def __call__(self, logits, targets):
        if logits.shape != targets.shape:
            logits = logits.flatten()
        assert logits.shape == targets.shape
        return torch.sum((logits - targets) ** 2)


class DLRTargetedLoss():
    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def __call__(self, x, y, y_target): ## y_target would be t
        x_sorted, ind_sorted = x.sort(dim=1)
        u = torch.arange(x.shape[0])

        loss = -(x[u, y] - x[u, y_target]) / (x_sorted[:, -1] - .5 * (
            x_sorted[:, -3] + x_sorted[:, -4]) + 1e-12)

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        else:
            assert False, self.reduction


def load_scheduler(optimizer, config):
    num_epochs = config['num_epochs']
    
    if 'scheduler' not in config:
        scheduler = NoneScheduler(optimizer)  
    elif config['scheduler'] == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:
        print(f'ERROR: UNSUPPORTED SCHEDULER: {config["scheduler"]}'); exit(1)
        
    return scheduler


def load_criterion(name):
    if   name == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    elif name == 'CrossEntropyLossIndiv':
        criterion = nn.CrossEntropyLoss(reduction="none")
    
    elif name == 'DLRTargetedLoss':
        criterion = DLRTargetedLoss(reduction="none")
    elif name == 'DLRTargetedLossIndiv':
        criterion = DLRTargetedLoss(reduction="none")

    elif name == 'NLLLoss':
        criterion = nn.NLLLoss()
    elif name == 'QuadraticLoss':
        criterion = QuadraticLoss()
    
    else:
        print(f'ERROR: UNSUPPORTED LOSS_FN: {"criterion"}'); exit(1)
        
    return criterion


def load_optimizer(params, program, features, T, config, lr=None, prefix_fn=None, update=False):
    from nos.opt_learned import LEARNED_OPT
    
    lr = config['lr'] if lr is None else lr
    momentum  = 0
    momentum2 = 0
    weight_decay = 0
    
    if 'momentum' in config:
        momentum = config['momentum']
    if 'momentum2' in config:
        momentum2 = config['momentum2']
    if 'momentum3' in config:
        momentum3 = config['momentum3']
    if 'weight_decay' in config:
        weight_decay = config['weight_decay']

    opt = LEARNED_OPT(params, program=program,
                    weight_decay=weight_decay,
                    lr=lr, momentum=momentum, momentum2=momentum2, momentum3=momentum3,
                    features=features, T=T, prefix_fn=prefix_fn, update=update)
    return opt
