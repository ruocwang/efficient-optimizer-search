import torch
import torch.nn.functional as F
import numpy as np
import sys
import sympy as sp


from collections import OrderedDict
from .neural_functions import init_neural_function
## base classes
from .library_functions import LibraryFunction, BinaryFunction, UnaryFunction, Operand


######## util functions ########
class features():
    def __init__(self):
        self.names = OrderedDict({'W':None, 'G':None, 'Gp':None, 'Gr':None,
                                  'M1':None, 'M2':None, 'M3':None,
                                  'M1p':None, 'M2p':None, 'M3p':None,
                                  'Adam':None, 'RMSprop':None,
                                  'dd':None, 't':None, 'T':None,
                                  'momentum':None, 'momentum2':None, 'momentum3':None})

class const_features():
    def __init__(self):
        self.names = OrderedDict({'W':None, 'G':None, 'Gp':None, 'Gr':None,
                                  'M1':None, 'M2':None, 'M3':None,
                                  't':None, 'T':None, 'lr':None, 'lamb':None, 'L':None})


######## Operants (feature selection) ######## (operands must not be shared since each task has its own features)
#### gradients ####
class CIFAR_Grad(Operand):
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units)

    @staticmethod
    def get_name(): return 'G'
    def execute_sym(self): return sp.symbols(self.get_name(), real=True)


class CIFAR_Grad2(Operand):
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units)

    def execute_on_batch(self, batch, deter=False):
        return batch['G'] ** 2

    @staticmethod
    def get_name(): return 'G2'
    def execute_sym(self): return sp.symbols(self.get_name(), real=True)


class CIFAR_Grad3(Operand):
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units)

    def execute_on_batch(self, batch, deter=False):
        return batch['G'] ** 3

    @staticmethod
    def get_name(): return 'G3'
    def execute_sym(self): return sp.symbols(self.get_name(), real=True)


class CIFAR_GradPrev(Operand):
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units)

    @staticmethod
    def get_name(): return 'Gp'
    def execute_sym(self): return sp.symbols(self.get_name(), real=True)


#### wrc modified
class CIFAR_GradRegu(Operand):
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units)

    @staticmethod
    def get_name(): return 'Gr'
    def execute_sym(self): return sp.symbols(self.get_name(), real=True)
####


class CIFAR_Mom1(Operand):
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units)

    @staticmethod
    def get_name(): return 'M1'
    def execute_sym(self): return sp.symbols(self.get_name(), real=True)

class CIFAR_Mom1p(Operand): ## mu * state + (1 - mu) * g
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units)

    @staticmethod
    def get_name(): return 'M1p'
    def execute_sym(self): return sp.symbols(self.get_name(), real=True)

class CIFAR_Mom2(Operand):
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units)

    @staticmethod
    def get_name(): return 'M2'
    def execute_sym(self): return sp.symbols(self.get_name(), real=True)

class CIFAR_Mom2p(Operand): ## init differs
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units)

    @staticmethod
    def get_name(): return 'M2p'
    def execute_sym(self): return sp.symbols(self.get_name(), real=True)

class CIFAR_Mom3(Operand):
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units)

    @staticmethod
    def get_name(): return 'M3'
    def execute_sym(self): return sp.symbols(self.get_name(), real=True)

class CIFAR_Mom3p(Operand):
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units)

    @staticmethod
    def get_name(): return 'M3p'
    def execute_sym(self): return sp.symbols(self.get_name(), real=True)


## adaptive-norm momentums
class CIFAR_AdaMomBase(Operand):
    def __init__(self, input_size, output_size, num_units, feat_name, mom_name):
        super().__init__(input_size, output_size, num_units)
        self.feat_name = feat_name
        self.mom_name  = mom_name

    def execute_on_batch(self, batch, deter=False):
        m = batch[self.feat_name]
        beta = batch[self.mom_name]
        step = batch['t']
        return m / (1 - beta ** step)

    def execute_sym(self): return sp.symbols(self.get_name(), real=True)

class CIFAR_AdaMom1(CIFAR_AdaMomBase):
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units, 'M1', 'momentum')

    @staticmethod
    def get_name(): return 'AM1'

class CIFAR_AdaMom1p(CIFAR_AdaMomBase):
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units, 'M1p', 'momentum')

    @staticmethod
    def get_name(): return 'AM1p'
    
class CIFAR_AdaMom2(CIFAR_AdaMomBase):
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units, 'M2', 'momentum2')

    @staticmethod
    def get_name(): return 'AM2'

class CIFAR_AdaMom2p(CIFAR_AdaMomBase):
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units, 'M2p', 'momentum2')

    @staticmethod
    def get_name(): return 'AM2p'

class CIFAR_AdaMom3(CIFAR_AdaMomBase):
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units, 'M3', 'momentum3')

    @staticmethod
    def get_name(): return 'AM3'

class CIFAR_AdaMom3p(CIFAR_AdaMomBase):
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units, 'M3p', 'momentum3')

    @staticmethod
    def get_name(): return 'AM3p'

## signed inputs
class CIFAR_SignGrad(Operand):
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units)

    def execute_on_batch(self, batch, deter=False): return torch.sign(batch['G'])

    @staticmethod
    def get_name(): return 'SG'
    def execute_sym(self): return sp.sign(sp.symbols('G', real=True))

class CIFAR_SignMom1(Operand):
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units)

    def execute_on_batch(self, batch, deter=False):
        return torch.sign(batch['M1'])

    @staticmethod
    def get_name(): return 'SM1'
    def execute_sym(self): return sp.sign(sp.symbols('M1', real=True))

class CIFAR_SignMom1p(Operand):
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units)

    def execute_on_batch(self, batch, deter=False):
        return torch.sign(batch['M1p'])

    @staticmethod
    def get_name(): return 'SM1p'
    def execute_sym(self): return sp.sign(sp.symbols('M1p', real=True))
## no need for sign adamom

## scaled signed inputs


#### number ####
class N1(Operand):
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units)

    def execute_on_batch(self, batch, deter=False):
        return torch.ones_like(batch['G'])

    @staticmethod
    def get_name(): return '1'
    def execute_sym(self): return 1


class N2(Operand):
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units)

    def execute_on_batch(self, batch, deter=False):
        return torch.ones_like(batch['G']) * 2

    @staticmethod
    def get_name(): return '2'
    def execute_sym(self): return 2


#### distribution ####
class Noise(Operand):
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units)

    def execute_on_batch(self, batch, deter=False):
        data = batch['G']
        if deter:
            cpu_rng_state = torch.get_rng_state()
            torch.manual_seed(400)
        noise = torch.normal(mean=torch.zeros_like(data), std=0.01 * torch.ones_like(data))
        if deter: torch.set_rng_state(cpu_rng_state)
        return noise

    @staticmethod
    def get_name(): return 'Noise'
    def execute_sym(self): return sp.symbols(self.get_name(), real=True)


#### weights ####
class W4(Operand):
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units)
    
    def execute_on_batch(self, batch, deter=False):
        return batch["W"] * 10e-4
    
    @staticmethod
    def get_name(): return 'W4'
    def execute_sym(self): return sp.symbols(self.get_name(), real=True)


class W3(Operand):
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units)
    
    def execute_on_batch(self, batch, deter=False):
        return batch['W'] * 10e-3
    
    @staticmethod
    def get_name(): return 'W3'
    def execute_sym(self): return sp.symbols(self.get_name(), real=True)


class W2(Operand):
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units)
    
    def execute_on_batch(self, batch, deter=False):
        return batch['W'] * 10e-2
    
    @staticmethod
    def get_name(): return 'W2'
    def execute_sym(self): return sp.symbols(self.get_name(), real=True)


class W1(Operand):
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units)
    
    def execute_on_batch(self, batch, deter=False):
        return batch['W'] * 10e-1
    
    @staticmethod
    def get_name(): return 'W1'
    def execute_sym(self): return sp.symbols(self.get_name(), real=True)


#### optimizers ####
class Adam(Operand):
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units)
    
    @staticmethod
    def get_name(): return 'Adam'
    def execute_sym(self): return sp.symbols(self.get_name(), real=True)
    def reset_memory_state(self): return None ## no need, cuz rmsprop is reset everytime learned_opt is created


class RMSprop(Operand):
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units)
    
    @staticmethod
    def get_name(): return 'RMSprop'
    def execute_sym(self): return sp.symbols(self.get_name(), real=True)
    def reset_memory_state(self): return None ## no need, cuz rmsprop is reset everytime learned_opt is created


#### decay operands ####
class DecayFunction(Operand): ## base
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units)

    def comp_decay_rate(self): raise NotImplementedError

    def execute_on_batch(self, batch, deter=False):
        t = batch['t']
        T = batch['T']
        decay_rate = self.comp_decay_rate(t, T)
        decay_rate_mat = torch.zeros_like(batch['G']).fill_(decay_rate.item())
        return decay_rate_mat

    def execute_sym(self): return sp.symbols(self.get_name(), real=True)


class LinearDecay(DecayFunction):
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units)

    def comp_decay_rate(self, t, T): return (1 - t/T)
    
    @staticmethod
    def get_name(): return 'ld'


class CosineDecay(DecayFunction):
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units)
        self.n = 0.5 ## which is "cd" in Quoc paper

    def comp_decay_rate(self, t, T):
        # t = min(t, T) ## capped
        import rlcompleter, pdb; pdb.Pdb.complete=rlcompleter.Completer(locals()).complete; pdb.set_trace()
        return 0.5 * (1 + torch.cos(2*np.pi*self.n*t/T))
    
    @staticmethod
    def get_name(): return 'cd'


class RestartDecay(DecayFunction):
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units)
        self.n = 20 ## default rd20, don't know how many of these Quoc used

    def comp_decay_rate(self, t, T):
        # t = min(t, T) ## capped
        return 0.5 * (1 + torch.cos(np.pi*((t*self.n)%T)/T))
    
    @staticmethod
    def get_name(): return 'rd'


class AnnealedNoiseDecay(DecayFunction):
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units)

    def comp_decay_rate(self, t, T):
        # t = min(t, T) ## capped
        return torch.normal(0, 1/((1 + t)**0.55))
    
    @staticmethod
    def get_name(): return 'ep'


class DynamicDecay(Operand): ## dynamic decay used in AutoAttack
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units)

    @staticmethod
    def get_name(): return 'dd'
    def execute_sym(self): return sp.symbols(self.get_name(), real=True)
    def reset_memory_state(self): return None


#### decay min version
class LinearDecayMin(DecayFunction):
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units)

    def comp_decay_rate(self, t, T):
        # t = min(t, T) ## capped
        return (1 - t/T) + 0.01
    
    @staticmethod
    def get_name(): return 'ldm'


class CosineDecayMin(DecayFunction):
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units)
        self.n = 0.5 ## which is "cd" in Quoc paper

    def comp_decay_rate(self, t, T):
        # t = min(t, T) ## capped
        return 0.5 * (1 + torch.cos(2*np.pi*self.n*t/T)) + 0.03
    
    @staticmethod
    def get_name(): return 'cdm'


class RestartDecayMin(DecayFunction):
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units)
        self.n = 20 ## default rd20, don't know how many of these Quoc used

    def comp_decay_rate(self, t, T):
        # t = min(t, T) ## capped
        return 0.5 * (1 + torch.cos(np.pi*((t*self.n)%T)/T)) + 0.03
    
    @staticmethod
    def get_name(): return 'rdm'


class AnnealedNoiseDecayMin(DecayFunction):
    def __init__(self, input_size, output_size, num_units):
        super().__init__(input_size, output_size, num_units)

    def comp_decay_rate(self, t, T):
        # t = min(t, T) ## capped
        return torch.normal(0, 1/((1 + t)**0.55)) + 0.03
    
    @staticmethod
    def get_name(): return 'epm'



######## binary ########
class AddFunction(BinaryFunction):
    def __init__(self, input_size, output_size, num_units, function1=None, function2=None):
        super().__init__(input_size, output_size, num_units, function1=function1, function2=function2)

    def mapping(self, x, y): return x + y
    
    @staticmethod
    def get_name(): return "Add"
    def symfun(self, sym1, sym2): return self.mapping(sym1, sym2)

    
class SubFunction(BinaryFunction):
    def __init__(self, input_size, output_size, num_units, function1=None, function2=None):
        super().__init__(input_size, output_size, num_units, function1=function1, function2=function2)

    def mapping(self, x, y): return x - y
    
    @staticmethod
    def get_name(): return "Sub"
    def symfun(self, sym1, sym2): return self.mapping(sym1, sym2)


class MulFunction(BinaryFunction):
    def __init__(self, input_size, output_size, num_units, function1=None, function2=None):
        super().__init__(input_size, output_size, num_units, function1=function1, function2=function2)

    def mapping(self, x, y): return x * y
    
    @staticmethod
    def get_name(): return "Mul"
    def symfun(self, sym1, sym2): return self.mapping(sym1, sym2)
    

class DivFunction(BinaryFunction):
    def __init__(self, input_size, output_size, num_units, function1=None, function2=None):
        super().__init__(input_size, output_size, num_units, function1=function1, function2=function2)

    def mapping(self, x, y):
        try:
            return x / (y + 1e-8)
        except:
            import pdb; pdb.set_trace()
    
    @staticmethod
    def get_name(): return "Div"
    def symfun(self, sym1, sym2): return sym1 / (sym2 + 1e-8)


class PowFunction(BinaryFunction):
    def __init__(self, input_size, output_size, num_units, function1=None, function2=None):
        super().__init__(input_size, output_size, num_units, function1=function1, function2=function2)

    def mapping(self, x, y):
        return x ** y # element-wise
    
    @staticmethod
    def get_name(): return "Pow"
    def symfun(self, sym1, sym2): return self.mapping(sym1, sym2)








######## unary ########
class SqrtFunction(UnaryFunction):

    def __init__(self, input_size, output_size, num_units, fxn=None):
        super().__init__(input_size, output_size, num_units, fxn=fxn)

    def mapping(self, x):
        return torch.sqrt(torch.abs(x))
    
    @staticmethod
    def get_name(): return 'Sqrt'
    def symfun(self, sym): return sp.sqrt(sp.sqrt(sym**2))


class SignFunction(UnaryFunction):

    def __init__(self, input_size, output_size, num_units, fxn=None):
        super().__init__(input_size, output_size, num_units, fxn=fxn)

    def mapping(self, x): return torch.sign(x)
    
    @staticmethod
    def get_name(): return 'Sign'
    def symfun(self, sym): return sp.sign(sym)


class ExpFunction(UnaryFunction):

    def __init__(self, input_size, output_size, num_units, fxn=None):
        super().__init__(input_size, output_size, num_units, fxn=fxn)

    def mapping(self, x):
        return torch.exp(x)
    
    @staticmethod
    def get_name(): return 'Exp'
    def symfun(self, sym): return sp.exp(sym)


class NegFunction(UnaryFunction):
    def __init__(self, input_size, output_size, num_units, fxn=None):
        super().__init__(input_size, output_size, num_units, fxn=fxn)

    def mapping(self, x):
        return -x
    
    @staticmethod
    def get_name(): return '-'
    def symfun(self, sym): return -sym


class LogFunction(UnaryFunction):
    def __init__(self, input_size, output_size, num_units, fxn=None):
        super().__init__(input_size, output_size, num_units, fxn=fxn)

    def mapping(self, x): return torch.log(torch.abs(x))
    
    @staticmethod
    def get_name(): return 'Log'
    def symfun(self, sym): return sp.log(sp.sqrt(sym**2))


class ClipFunction(UnaryFunction): ## base class

    def __init__(self, input_size, output_size, num_units, fxn=None):
        super().__init__(input_size, output_size, num_units, fxn=fxn)

    def mapping(self, x):
        return torch.clamp(x, -self.bound, self.bound)
    def symfun(self, sym): return sp.symbols(self.get_name(), cls=sp.Function, real=True)(sym)


class Clip5Function(ClipFunction):
    def __init__(self, input_size, output_size, num_units, fxn=None):
        super().__init__(input_size, output_size, num_units, fxn=fxn)
        self.bound = 1e-5

    @staticmethod
    def get_name(): return 'Clip5'
    

class Clip4Function(ClipFunction):
    def __init__(self, input_size, output_size, num_units, fxn=None):
        super().__init__(input_size, output_size, num_units, fxn=fxn)
        self.bound = 1e-4

    @staticmethod
    def get_name(): return 'Clip4'
    

class Clip3Function(ClipFunction):
    def __init__(self, input_size, output_size, num_units, fxn=None):
        super().__init__(input_size, output_size, num_units, fxn=fxn)
        self.bound = 1e-3

    @staticmethod
    def get_name(): return 'Clip3'

class Clip0Function(ClipFunction):
    def __init__(self, input_size, output_size, num_units, fxn=None):
        super().__init__(input_size, output_size, num_units, fxn=fxn)
        self.bound = 1

    @staticmethod
    def get_name(): return 'Clip0'



class DropFunction(UnaryFunction): ## base class
    def __init__(self, input_size, output_size, num_units, fxn=None):
        super().__init__(input_size, output_size, num_units, fxn=fxn)

    def mapping(self, x): return F.dropout(x, p=self.dropout_rate)
    def symfun(self, sym): return sp.symbols(self.get_name(), cls=sp.Function, real=True)(sym)


class Drop1Function(DropFunction):
    def __init__(self, input_size, output_size, num_units, fxn=None):
        super().__init__(input_size, output_size, num_units, fxn=fxn)
        self.dropout_rate = 0.1

    def mapping(self, x): return F.dropout(x, p=self.dropout_rate)

    @staticmethod
    def get_name(): return 'Drop1'


class Drop3Function(DropFunction):
    def __init__(self, input_size, output_size, num_units, fxn=None):
        super().__init__(input_size, output_size, num_units, fxn=fxn)
        self.dropout_rate = 0.3

    def mapping(self, x): return F.dropout(x, p=self.dropout_rate)

    @staticmethod
    def get_name(): return 'Drop3'


class Drop5Function(DropFunction):
    def __init__(self, input_size, output_size, num_units, fxn=None):
        super().__init__(input_size, output_size, num_units, fxn=fxn)
        self.dropout_rate = 0.5

    def mapping(self, x): return F.dropout(x, p=self.dropout_rate)

    @staticmethod
    def get_name(): return 'Drop5'


## wrc added
class MomentumFunction(UnaryFunction):
    def __init__(self, input_size, output_size, num_units, fxn=None):
        super().__init__(input_size, output_size, num_units, fxn=fxn)
        self.mu = 0.9
        self.state = None

    def mapping(self, x):
        if self.state is None:
            new_state = x
        else:
            new_state = self.state * self.mu + x
        self.state = new_state.detach().clone()

        return new_state
    
    @staticmethod
    def get_name(): return 'Mom'
    def symfun(self, sym): return sp.symbols(self.get_name(), cls=sp.Function, real=True)(sym)
    def reset_memory_state(self):
        self.state = None
        self.submodules['fxn'].reset_memory_state()