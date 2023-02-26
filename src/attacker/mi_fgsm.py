## modified from sign_pgd
import torch
from torch.optim.optimizer import Optimizer, required


class MI_FGSM(Optimizer):
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
        super(MI_FGSM, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MI_FGSM, self).__setstate__(state)
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
            
            ## TODO under dev ##
            momentum = 1.0
            
            assert not nesterov
            assert dampening == 0
            assert weight_decay == 0

            for p in group['params']:
                if p.grad is None:
                    continue
                
                ## normalized grad (l1)
                d_p = p.grad
                d_p /= torch.abs(d_p).mean()
                # d_p = torch.sign(p.grad)

                
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(d_p).detach()
                    else:
                        ## m = mu * m + normed(grad)
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                d_p = torch.sign(d_p)
                p.add_(d_p, alpha=-group['lr'])

        return loss
