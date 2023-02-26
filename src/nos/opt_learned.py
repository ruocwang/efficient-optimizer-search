from sqlite3 import paramstyle
import torch
from torch.optim.optimizer import Optimizer, required

from utils.logging import print_program
from .opt_adam import Adam
from .opt_rmsprop import RMSprop


class LEARNED_OPT(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}

        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the 
        parameters, gradient, velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, momentum2=0, momentum3=0,
                 dampening=0, weight_decay=0, nesterov=False,
                 program=None, features=None, T=None, prefix_fn=None, update=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if momentum2 < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum2))
        if momentum3 < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum3))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, momentum2=momentum2, momentum3=momentum3,
                        dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(LEARNED_OPT, self).__init__(params, defaults)
        
        
        #### wrc modified
        self.update = update
        self.program = program
        ## for NAG
        self.grad_prev_flat = []
        ## updated spaces
        self.prefix_fn = prefix_fn
        self.features = features
        self.T = T ## total steps
        self.program.reset_memory_state() ## reset state everytime an optimizer is freshly initialized
        self.prog_str = print_program(self.program)
        ## Adam and RMSProp
        self.adam = Adam(params, lr=0.001*3)
        self.rmsprop = RMSprop(params, lr=0.01*3)
        ## Dynamic Decay
        batch_size = params[0].shape[0]
        self.n_iter_2 = max(int(0.22 * self.T), 1)
        self.n_iter_min = max(int(0.06 * self.T), 1)
        self.size_decr = max(int(0.03 * self.T), 1)
        self.thr_decr = 0.75 # rho
        self.device = params[0].device
        self.dd_dict = {
            'loss_steps': torch.zeros((self.T, batch_size)).to(self.device), ## tensor(step, batch)
            'loss_best': torch.zeros((batch_size)).to(self.device),
            'reduced_last_check': torch.zeros((batch_size)).to(self.device), ## tensor(batch) of 0/1
            'loss_best_last_check': torch.zeros((batch_size)).to(self.device), ## tensor(batch)
            'step_size': torch.ones((batch_size)).to(self.device), ## * lr removed, dup
            'counter3': 0,
            'k': self.n_iter_2,
        }
        ####
        self.num_steps = 0


    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = torch.zeros(x.shape[1]).to(self.device)
        for counter5 in range(k):
          t += (x[j - counter5] > x[j - counter5 - 1]).float()

        return (t <= k * k3 * torch.ones_like(t)).float()


    def __setstate__(self, state):
        super(LEARNED_OPT, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)


    @torch.no_grad()
    def step(self, cur_loss_pos=None, closure=None, **kwargs):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        #### fetch function args
        self.num_steps += 1
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        ## params for const optimization
        const_params = None
        if 'const_params' in kwargs:
            const_params = kwargs['const_params']

        for gid, group in enumerate(self.param_groups):
            weight_decay = group['weight_decay']
            momentum  = group['momentum']
            momentum2 = group['momentum2']
            momentum3 = group['momentum3']
            # dampening = group['dampening']
            nesterov  = group['nesterov']
            
            assert not nesterov
            assert momentum  > 0, "always compute 1stMom, it's the program's job to decide whether to use it"
            assert momentum2 > 0, "always compute 2ndMom, it's the program's job to decide whether to use it"
            assert momentum3 > 0, "always compute 3rdMom, it's the program's job to decide whether to use it"

            for pid, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                
                ## get gradient
                d_p = p.grad

                ## weight decay
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                ## 1st momentum
                if momentum != 0 and 'M1()' in self.prog_str:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state: ## init
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1)
                    mom_1st = buf

                if momentum != 0 and 'M1p()' in self.prog_str:
                    param_state = self.state[p]
                    if 'momentum_p_buffer' not in param_state: ## init
                        buf = param_state['momentum_p_buffer'] = torch.clone(d_p).detach() * (1 - momentum)
                    else:
                        buf = param_state['momentum_p_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - momentum) ## (mu * state + (1 - mu) * g)
                    mom_1st_p = buf

                ## 2nd momentum
                if momentum2 != 0 and 'M2()' in self.prog_str:
                    param_state = self.state[p]
                    if 'momentum2_buffer' not in param_state: ## init
                        buf = param_state['momentum2_buffer'] = torch.clone(d_p ** 2).detach()
                    else:
                        buf = param_state['momentum2_buffer']
                        buf.mul_(momentum2).addcmul_(d_p, d_p, value=1 - momentum2)
                    mom_2nd = buf
                
                if momentum2 != 0 and 'M2p()' in self.prog_str:
                    param_state = self.state[p]
                    if 'momentum2_p_buffer' not in param_state: ## init
                        buf = param_state['momentum2_p_buffer'] = torch.clone(d_p ** 2).detach() * (1 - momentum2)
                    else:
                        buf = param_state['momentum2_p_buffer']
                        buf.mul_(momentum2).addcmul_(d_p, d_p, value=1 - momentum2)
                    mom_2nd_p = buf

                ## 3rd momentum
                if momentum3 != 0 and 'M3()' in self.prog_str:
                    param_state = self.state[p]
                    if 'momentum3_buffer' not in param_state: ## init
                        buf = param_state['momentum3_buffer'] = torch.clone(d_p ** 3).detach()
                    else:
                        buf = param_state['momentum3_buffer']
                        buf.mul_(momentum3).addcmul_(d_p, d_p, value=1 - momentum3)
                    mom_3rd = buf

                if momentum3 != 0 and 'M3p()' in self.prog_str:
                    param_state = self.state[p]
                    if 'momentum3_p_buffer' not in param_state: ## init
                        buf = param_state['momentum3_p_buffer'] = torch.clone(d_p ** 3).detach() * (1 - momentum3)
                    else:
                        buf = param_state['momentum3_p_buffer']
                        buf.mul_(momentum3).addcmul_(d_p, d_p, value=1 - momentum3)
                    mom_3rd_p = buf


                ## step t
                if 't' in self.features.names:
                    param_state = self.state[p]
                    if 'step' not in param_state:
                        param_state['step'] = 0
                    param_state['step'] += 1
                    step = torch.tensor([[param_state['step']]]).to(p.device)
                if 'T' in self.features.names:
                    T = torch.tensor([[self.T]]).to(p.device)

                ## decay step
                if 'dd()' in self.prog_str and step > 1:
                    i = step.item() - 1
                    y1 = cur_loss_pos.detach().clone() ## current loss
                    self.dd_dict['loss_steps'][i] = y1 + 0 ## store loss at every step
                    ind = (y1 > self.dd_dict['loss_best'])
                    ## TODO: restart from previous best
                    # x_best[ind] = x_adv[ind].clone() ## store the best adversarial example
                    # grad_best[ind] = grad[ind].clone() ## store the gradient
                    self.dd_dict['loss_best'][ind] = y1[ind] + 0 ## update loss_best
                    self.dd_dict['counter3'] += 1

                    # print(self.dd_dict['loss_steps'].sum(), self.dd_dict['loss_best'].sum())

                    if self.dd_dict['counter3'] == self.dd_dict['k']:
                        fl_oscillation = self.check_oscillation(self.dd_dict['loss_steps'], i, self.dd_dict['k'],
                                        self.dd_dict['loss_best'], k3=self.thr_decr)
                        fl_reduce_no_impr = (1. - self.dd_dict['reduced_last_check']) * (
                            self.dd_dict['loss_best_last_check'] >= self.dd_dict['loss_best']).float()
                        fl_oscillation = torch.max(fl_oscillation, fl_reduce_no_impr)
                        self.dd_dict['reduced_last_check'] = fl_oscillation.clone()
                        self.dd_dict['loss_best_last_check'] = self.dd_dict['loss_best'].clone()
                        # print(fl_oscillation.sum(), self.dd_dict['step_size'].mean())
                        if fl_oscillation.sum() > 0:
                            ind_fl_osc = (fl_oscillation > 0)
                            self.dd_dict['step_size'][ind_fl_osc] /= 2.0

                            ## TODO: restart from previous best
                            # x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                            # grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

                        self.dd_dict['k'] = max(self.dd_dict['k'] - self.size_decr, self.n_iter_min)
                        self.dd_dict['counter3'] = 0


                #### program ####
                ## make inputs to the nos program
                orig_shape = d_p.shape
                
                batch = {}
                d_p_flat = d_p.flatten().unsqueeze(0) # flatten and fake a batch dimension
                batch['W'] = p.flatten().unsqueeze(0)
                batch['G'] = d_p_flat
                batch['t'] = step.float()
                batch['T'] = T.float()
                batch['lr'] = group['lr']
                batch['momentum'] = momentum
                batch['momentum2'] = momentum2
                batch['momentum3'] = momentum3


                if 'Gp()' in self.prog_str:
                    if pid < len(self.grad_prev_flat):
                        d_p_prev_flat = self.grad_prev_flat[pid]
                    else: ## first visit
                        d_p_prev_flat = torch.zeros_like(d_p_flat)
                    batch['Gp'] = d_p_prev_flat
                
                if 'M1()' in self.prog_str:
                    mom_1st_flat = mom_1st.flatten().unsqueeze(0)
                    batch['M1'] = mom_1st_flat
                if 'M1p()' in self.prog_str:
                    mom_1st_p_flat = mom_1st_p.flatten().unsqueeze(0)
                    batch['M1p'] = mom_1st_p_flat
                if 'M2()' in self.prog_str:
                    mom_2nd_flat = mom_2nd.flatten().unsqueeze(0)
                    assert (mom_2nd_flat < 0).sum() == 0
                    batch['M2'] = mom_2nd_flat
                if 'M2p()' in self.prog_str:
                    mom_2nd_p_flat = mom_2nd_p.flatten().unsqueeze(0)
                    assert (mom_2nd_p_flat < 0).sum() == 0
                    batch['M2p'] = mom_2nd_p_flat
                if 'M3()' in self.prog_str:
                    mom_3rd_flat = mom_3rd.flatten().unsqueeze(0)
                    batch['M3'] = mom_3rd_flat
                if 'M3p()' in self.prog_str:
                    mom_3rd_p_flat = mom_3rd_p.flatten().unsqueeze(0)
                    batch['M3p'] = mom_3rd_p_flat

                if 'Adam()' in self.prog_str:
                    adam_update = self.adam.get_update(gid, pid)
                    batch['Adam'] = adam_update.flatten().unsqueeze(0)
                if 'RMSprop()' in self.prog_str:
                    rmsprop_update = self.rmsprop.get_update(gid, pid)
                    batch['RMSprop'] = rmsprop_update.flatten().unsqueeze(0)
                
                ## AA
                if 'dd()' in self.prog_str:
                    batch['dd'] = self.dd_dict['step_size'].view(-1,1,1,1).repeat(1,3,32,32).flatten()

                ## constrained optimization
                if const_params is not None:
                    batch['lamb'] = const_params['lamb']
                    batch['L'] = const_params['L']
                    batch['Gr'] = const_params['Gr']


                #### main
                update_flatten = self.program.execute_on_batch(batch)
                ####

                if 'Gp' in self.features.names:
                    if pid < len(self.grad_prev_flat):
                        self.grad_prev_flat[pid] = d_p_flat
                    else:
                        assert pid == len(self.grad_prev_flat)
                        self.grad_prev_flat.append(d_p_flat)
                
                update = update_flatten.squeeze(0).reshape(orig_shape)
                
                if self.prefix_fn is not None: ## projection for Linf attack
                    update = self.prefix_fn(update)
                
                ## times learning rate
                if self.update: # p = update, the program produces the full update
                    p.sub_(p).add_(update)
                else: # p = p - lr*g
                    p.add_(update, alpha=-group['lr']) ## -lr

        return loss
