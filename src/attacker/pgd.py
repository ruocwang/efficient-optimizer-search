import math
from foolbox import PyTorchModel
from numpy import gradient
from sympy import E
import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam
from nos.train_utils_zoo import DLRTargetedLoss


def normalize_Linf(x):
    ndims = 3 ## image
    t = x.abs().view(x.shape[0], -1).max(1)[0]
    return x / (t.view(-1, *([1] * ndims)) + 1e-12)

def get_random_perturb(x, eps):
    t = 2 * torch.rand(x.shape).to(x.device).detach() - 1
    return eps * torch.ones_like(x).detach() * normalize_Linf(t)


# performs Linf-constraint PGD attack w/o noise
# @epsilon: radius of Linf-norm ball
def Linf_PGD(x_in, y_true, net, criterion, opt_func, step, eps, random_start=False, inds_to_attack=None, n_target_classes=-1, use_best_adv=True):
    if eps == 0: return x_in

    ## ignore misclassified ones
    if inds_to_attack is not None:
        x_in_full = x_in.clone()
        x_in = x_in[inds_to_attack]
        y_true = y_true[inds_to_attack]

    ## eval mode
    training = False
    if not isinstance(net, PyTorchModel) and net.training:
        training = True
        net.eval()

    ## random restart
    if random_start:
        x_adv = x_in.clone() + get_random_perturb(x_in, eps)
        x_adv.clamp_(0, 1)
    else:
        x_adv = x_in.clone()
    x_adv = x_adv.requires_grad_()

    ## create optimizer (i.e. reinit for each round)
    # optimizer = Linf_SGD([x_adv], lr=0.007)
    optimizer = opt_func([x_adv])

    x_best_adv = x_adv.clone()
    out = net(x_adv)

    for sid in range(step):
        optimizer.zero_grad()
        if not isinstance(net, PyTorchModel): net.zero_grad()

        # out = net(x_adv)
        loss_raw = -criterion(out, y_true)

        if len(loss_raw.shape) >= 1: ## loss_indiv
            loss = loss_raw.sum()
        else:
            loss = loss_raw
        loss.backward()

        try:
            optimizer.step(cur_loss_pos=-loss_raw)
        except:
            optimizer.step()


        diff = x_adv - x_in

        ## clip(image)
        if torch.isnan(x_adv).sum() > 0:
            x_adv.detach().copy_(x_in)
            break
        diff.clamp_(-eps, eps)
        x_adv.detach().copy_((diff + x_in).clamp_(0, 1))

        out = net(x_adv)
        ## record best adv
        pred = out.detach().max(dim=-1)[1] == y_true
        # ind_pred = (pred == 0).nonzero().squeeze() ## misclassified ones
        ind_pred = pred == 0
        x_best_adv[ind_pred] = x_adv[ind_pred].clone()

    if not isinstance(net, PyTorchModel): net.zero_grad()
    if training: net.train()
    
    if inds_to_attack is not None:
        if use_best_adv:
            x_in_full[inds_to_attack] = x_best_adv
        else:
            x_in_full[inds_to_attack] = x_adv
        x_adv = x_in_full
    return x_adv


def Linf_PGD_targeted(x_in, y_true, net, criterion, opt_func, step, eps, random_start=False, inds_to_attack=None, n_target_classes=9):
    assert isinstance(criterion, DLRTargetedLoss), "must use dlr loss for targted attack"

    x_adv_final = x_in.clone()

    # print(inds_to_attack.sum())

    for target_class in range(2, n_target_classes + 2): # 2 - 10
        
        ## get y_target (z_t)
        output = net(x_in)
        y_target = output.sort(dim=1)[1][:, -target_class]

        ## create loss function with z_t
        dlr_loss_t = lambda x, y: criterion(x, y, y_target[inds_to_attack])

        ## targeted attack single run
        x_adv = Linf_PGD(x_adv_final, y_true, net, dlr_loss_t, opt_func, step, eps, random_start=random_start, inds_to_attack=inds_to_attack)

        ## eval x_adv
        with torch.no_grad():
            inds_to_attack = net(x_adv).max(dim=-1).indices == y_true
            x_adv_final[~inds_to_attack] = x_adv[~inds_to_attack] ## successed ones
            if inds_to_attack.sum() == 0:
                break
            # print(inds_to_attack.sum())

    return x_adv_final




















# performs L2-constraint PGD attack w/o noise
# @epsilon: radius of L2-norm ball
# def L2_PGD(x_in, y_true, net, steps, eps):
#     if eps == 0:
#         return x_in
#     training = net.training
#     if training:
#         net.eval()
#     x_adv = x_in.clone().requires_grad_()
#     optimizer = Adam([x_adv], lr=0.01)
#     ## TODO under dev ##
#     # optimizer.program.reset_memory_state() ## everytime an optimizer is initialized, reset memory state
#     eps = torch.tensor(eps).view(1,1,1,1).cuda()
#     #print('====================')
#     for _ in range(steps):
#         optimizer.zero_grad()
#         net.zero_grad()
#         out, _ = net(x_adv)
#         loss = -F.cross_entropy(out, y_true)
#         loss.backward()
#         #print(loss.item())
#         optimizer.step()
#         diff = x_adv - x_in
#         norm = torch.sqrt(torch.sum(diff * diff, (1, 2, 3)))
#         norm = norm.view(norm.size(0), 1, 1, 1)
#         norm_out = torch.min(norm, eps)
#         diff = diff / norm * norm_out
#         x_adv.detach().copy_((diff + x_in).clamp_(0, 1))
#     net.zero_grad()
#     # reset to the original state
#     if training :
#         net.train()
#     return x_adv

