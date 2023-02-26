import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required


class SignSGD(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SignSGD, self).__init__(params, defaults)
        ## TODO under dev ##
        self.T = 100 ## total steps

    def __setstate__(self, state):
        super(SignSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                # d_p = p.grad
                d_p = torch.sign(p.grad)
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf


                ## TODO under dev ##
                param_state = self.state[p]
                if 'step' not in param_state:
                    param_state['step'] = 0
                param_state['step'] += 1
                step = torch.tensor([[param_state['step']]]).float().to(p.device)
                T = torch.tensor([[self.T]]).float().to(p.device)

                ## TODO under dev ##
                update = torch.log(torch.abs(torch.sign(d_p) + self.cosine(step, T)))
                # p.add_(d_p, alpha=-group['lr'])
                p.add_(update, alpha=-group['lr'])

        return loss

    ## TODO under dev ##
    def cosine(self, t, T): ## cosine decay
        n = 0.5
        rate = 0.5 * (1 + torch.cos(2*np.pi*n*t/T))
        return rate
