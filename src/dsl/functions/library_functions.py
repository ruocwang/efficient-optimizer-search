import torch
import torch.nn.functional as F
import sympy as sp

from .neural_functions import init_neural_function


#### base class
class LibraryFunction:

    def __init__(self, submodules, input_type, output_type, input_size, output_size, num_units, name="", has_params=False):
        self.submodules = submodules
        self.input_type = input_type
        self.output_type = output_type
        self.input_size = input_size
        self.output_size = output_size
        self.num_units = num_units
        self.name = name
        self.has_params = has_params

        if self.has_params:
            assert "init_params" in dir(self)
            self.init_params()

    def get_submodules(self):
        return self.submodules

    def set_submodules(self, new_submodules):
        self.submodules = new_submodules

    def get_typesignature(self):
        return self.input_type, self.output_type

    def reset_memory_state(self): return None


class StartFunction(LibraryFunction):

    def __init__(self, input_type, output_type, input_size, output_size, num_units):
        self.program = init_neural_function(input_type, output_type, input_size, output_size, num_units)
        submodules = { 'program' : self.program } ## will be override in get_all_children()
        super().__init__(submodules, input_type, output_type, input_size, output_size, num_units, name="Start")

    def execute_on_batch(self, batch, deter=False):
        if deter:
            assert not list(batch.values())[0].is_cuda, "determnistic update only support cpu computations"
        return self.submodules["program"].execute_on_batch(batch, deter=deter) # (batch, seq_len, dim)

    def execute_sym(self): return self.submodules["program"].execute_sym()

    def reset_memory_state(self): self.submodules["program"].reset_memory_state()


#### dummy placeholders
class SimpleITE(LibraryFunction):
    def __init__(self): pass
class ITE(LibraryFunction):
    def __init__(self): pass
class FoldFunction(LibraryFunction):
    def __init__(self): pass

        
## base class for binary functions
class BinaryFunction(LibraryFunction):
    def __init__(self, input_size, output_size, num_units, function1=None, function2=None):
        if function1 is None:
            function1 = init_neural_function("atom", "atom", input_size, output_size, num_units)
        if function2 is None:
            function2 = init_neural_function("atom", "atom", input_size, output_size, num_units)
        submodules = { "function1" : function1, "function2" : function2 }
        super().__init__(submodules, "atom", "atom", input_size, output_size, num_units, name=self.get_name())

    def mapping(self):
        raise NotImplementedError

    def execute_on_batch(self, batch, deter=False):
        predicted_function1 = self.submodules["function1"].execute_on_batch(batch, deter=deter)
        predicted_function2 = self.submodules["function2"].execute_on_batch(batch, deter=deter)
        return self.mapping(predicted_function1, predicted_function2)

    def execute_sym(self):
        sym1 = self.submodules["function1"].execute_sym()
        sym2 = self.submodules["function2"].execute_sym()
        return self.symfun(sym1, sym2)

    def reset_memory_state(self):
        self.submodules["function1"].reset_memory_state()
        self.submodules["function2"].reset_memory_state()


## base class for unary functions
class UnaryFunction(LibraryFunction):
    def __init__(self, input_size, output_size, num_units, fxn=None):
        if fxn is None:
            fxn = init_neural_function("atom", "atom", input_size, output_size, num_units)
        submodules = { "fxn" : fxn }
        super().__init__(submodules, "atom", "atom", input_size, output_size, num_units, name=self.get_name())
        
    def mapping(self):
        raise NotImplementedError
    
    def execute_on_batch(self, batch, deter=False):
        predicted_function = self.submodules["fxn"].execute_on_batch(batch, deter=deter)
        if deter:
            cpu_rng_state = torch.get_rng_state()
            torch.manual_seed(400)
        res = self.mapping(predicted_function)
        if deter: torch.set_rng_state(cpu_rng_state)
        return res

    def execute_sym(self):
        sym = self.submodules["fxn"].execute_sym()
        return self.symfun(sym)

    def reset_memory_state(self): self.submodules["fxn"].reset_memory_state()


## base class for operands    
class Operand(LibraryFunction):
    def __init__(self, input_size, output_size, num_units):
        super().__init__({}, "atom", "atom", input_size, output_size, num_units, name=self.get_name(), has_params=False)

    def execute_on_batch(self, batch, deter=False):
        return batch[self.get_name()]