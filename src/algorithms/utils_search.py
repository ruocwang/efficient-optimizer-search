from datetime import datetime
import torch
import time
import numpy as np
import sympy as sp
from utils.logging import log_and_print, print_program, print_program_dict


#### dictionaries
class Visit():
    def __init__(self, features, update):
        self.visited_programs = {}
        self.evaled_programs  = {} # (best_lr, sample_score)
        
        self.dim = 25
        self.angle_thresh = 0.15
        self.features = features
        self.probe = self.get_random_probe() ## for checking duplicated programs
        self.probe_semantic, self.grad = self.get_random_semantic_probe()
        self.update = update


    def get_debug_probe(self, scale=1):
        probe = {}
        for name in self.features.names:
            probe[name] = np.round(torch.randn(1), 2) * scale

            ## no need to do G2/3, as they are handled in program executions
            if name in ['RMSprop', 'Adam']:
                probe[name] = probe['G']
            elif name in ['M2', 'M2p']:
                probe[name] = probe[name] ** 2
            elif name == 't':
                probe[name] = torch.tensor(3.0)
            elif name == 'T':
                probe[name] = torch.tensor(100.0)
            elif name == 'momentum':
                probe[name] = torch.tensor(0.9)
            elif name in ['momentum2', 'momentum3']:
                probe[name] = torch.tensor(0.999)

        return probe


    def get_random_probe(self, scale=1):
        probe = {}
        for name in self.features.names:
            probe[name] = torch.randn(self.dim) * scale
            ## no need to do G2/3, as they are handled in program executions
            if name in ['M2', 'M2p']:
                probe[name] = probe[name] ** 2
            elif name == 't':
                probe[name] = torch.tensor(30.0)
            elif name == 'T':
                probe[name] = torch.tensor(100.0)
            elif name == 'lr':
                probe[name] = 1.0
            elif name == 'L':
                probe[name] = 1400
            elif name == 'lamb':
                probe[name] = 0.
            elif name == 'momentum':
                probe[name] = torch.tensor(0.9)
            elif name in ['momentum2', 'momentum3']:
                probe[name] = torch.tensor(0.999)

        return probe


    def get_random_semantic_probe(self):
        probe = self.get_random_probe(scale=1/60)
        probe_semantic = copy.deepcopy(probe)
        grad = probe_semantic['G'] ## match the norm of real grad
        for name in self.features.names:
            if name in ['W']:
                probe_semantic[name] = grad.clone().data.fill_(1e-3)
            if name in ['Gp', 'M1', 'M1p', 'RMSprop', 'Adam']:
                probe_semantic[name] = grad.clone()
            if name in ['M2', 'M2p']:
                probe_semantic[name] = grad.clone() ** 2
            if name in ['M3']:
                probe_semantic[name] = grad.clone() ** 3
            # if name in ['t']: no need
            # if name in ['T']: no need
        grad = grad.clone()
        return probe_semantic, grad

    
    def semantic_valid(self, program):
        ## sign
        program.reset_memory_state()
        encode = program.execute_on_batch(self.probe_semantic)
        valid_sign = (encode < 0).sum() != 0 and (encode > 0).sum() != 0
        ## not const
        probe_semantic, _ = self.get_random_semantic_probe()
        program.reset_memory_state()
        encode2 = program.execute_on_batch(probe_semantic)
        valid_const = abs(encode2 - encode).sum() > 1e-6
        ## not const symbolic
        valid_const_sym = False
        sym_str = str(program.execute_sym())

        for name in self.features.names:
            if name in ['T', 't']: continue
            if name in sym_str: valid_const_sym = True
        return valid_sign and valid_const and valid_const_sym
    
    
    def descent_valid(self, program):
        cos_avg = 0
        num_trials = 25
        for _ in range(num_trials):
            probe_semantic, grad = self.get_random_semantic_probe()
            program.reset_memory_state()
            update = program.execute_on_batch(probe_semantic)
            if self.update: ## reverse back to gradient updates
                update = -(update - probe_semantic['W'])
            cos = torch.cosine_similarity(update, grad, dim=0)
            cos_avg += cos
        cos_avg /= num_trials
        return (cos_avg > self.angle_thresh).item()


    def is_evaluated(self, target_program):
        target_program.reset_memory_state()
        target_encode = target_program.execute_on_batch(self.probe, deter=True)

        for prog_str in self.evaled_programs:
            encode = self.evaled_programs[prog_str]["encode"]
            try:
                diff = abs(encode - target_encode).mean()
            except:
                print('error!')
                import pdb; pdb.set_trace()
            if diff < 1e-6:
                return self.evaled_programs[prog_str]
        return None


    #### sympy version
    def is_evaluated_sympy(self, target_program):
        target_prog_str = print_program(target_program)
        ## check for mathematical equivalency
        sym_target = sp.powsimp(target_program.execute_sym())
        for prog_str in self.evaled_programs:
            sym_key = self.evaled_programs[prog_str]["sympy"]
            
            try:
                if sp.re(sp.powsimp(sym_key - sym_target)) == 0:
                    return self.evaled_programs[prog_str]
            except:
                log_and_print('DEBUG: sympy error occurs')
                log_and_print(f'DEBUG: Q: {target_prog_str}')
                log_and_print(f'DEBUG: K: {prog_str}')
                continue
        return None


    def record_eval(self, program, best_lr, sample_score):
        prog_str = print_program(program, ignore_constants=True)
        sym = program.execute_sym()
        program.reset_memory_state()
        encode = program.execute_on_batch(self.probe, deter=True)
        self.evaled_programs[prog_str] = {
            "encode": encode,
            "sympy":  sym,
            "best_lr": best_lr,
            "sample_score": sample_score
        }


    def record_eval_mp(self, program): # for mp
        prog_str = print_program(program, ignore_constants=True)
        sym = program.execute_sym()
        program.reset_memory_state()
        encode = program.execute_on_batch(self.probe, deter=True)
        self.evaled_programs[prog_str] = {
            "encode": encode,
            "sympy": sym,
            "best_lr": 'placeholder',
            "sample_score": 'placeholder'
        }


    def update_record(self, program, best_lr, sample_score): # for mp
        prog_str = print_program(program, ignore_constants=True)
        self.evaled_programs[prog_str]["best_lr"] = best_lr
        self.evaled_programs[prog_str]['sample_score'] = sample_score





#### representations for math rules
class treenode():
    def __init__(self, val=None, children=[]):
        self.children = children
        self.val = val


    @staticmethod
    def check_parenthesis_complete(str):
        count = 0
        for c in str:
            if c == '(': count += 1
            if c == ')': count -= 1
        return count == 0


    @staticmethod
    def programstr2tree(prog_str):
        prog_str = prog_str.replace(' ', '') ## remove all spaces
        prog_parenthesis_s = prog_str.find('(')
        prog_parenthesis_e = prog_str.rfind(')')
        prog_name = prog_str[:prog_parenthesis_s]
        
        if prog_parenthesis_s == prog_parenthesis_e - 1: # base case
            return treenode(prog_str[:-2]) # remove trailing ()
        
        children = []
        hit = 0
        for idx, c in enumerate(prog_str):
            if c == ',' and treenode.check_parenthesis_complete(prog_str[(prog_parenthesis_s + 1):idx]): # binary func
                children.append(treenode.programstr2tree(prog_str[(prog_parenthesis_s + 1):idx]))
                children.append(treenode.programstr2tree(prog_str[(idx + 1):prog_parenthesis_e]))
                hit += 1
        assert hit <= 1

        if hit == 0:
            children.append(treenode.programstr2tree(prog_str[(prog_parenthesis_s + 1):prog_parenthesis_e]))
        
        return treenode(prog_name, children) 


    @staticmethod
    def programstr2prefix(prog_str):
        ## convert prog_str to tree
        root = treenode.programstr2tree(prog_str)

        return treenode.prefix_generator(root.children[0]) ## remove "Start"

    @staticmethod
    def prefix_generator(root):
        operand = root.val
        res = [operand]

        if len(root.children) == 0:
            return res

        for child in root.children:
            res += treenode.prefix_generator(child)
        return res
    
    


#### prog_str -> program
import torch
import copy
import dsl
from program_graph import ProgramGraph, ProgramNode
from utils.logging import log_and_print, print_program, print_program_dict


def prog_str2program(graph, prog_str, enumeration_depth, typesig=None, input_size=None, output_size=None):
    ## convert program string to tree structure
    from .utils_search import treenode
    str_node = treenode.programstr2tree(prog_str)
    
    #construct the num_selected lists
    max_depth_copy = graph.max_depth
    graph.max_depth = enumeration_depth
    all_programs = []
    enumerated = {}
    # input_size = self.input_size if input_size is None else input_size
    # output_size = self.output_size if output_size is None else output_size
    if typesig is None:
        root = copy.deepcopy(graph.root_node)
    else:
        new_start = dsl.StartFunction(input_type=typesig[0], output_type=typesig[1], input_size=input_size, output_size=output_size, num_units=graph.max_num_units)
        root = ProgramNode(new_start, 0, None, 0, 0, 0)

    def enumerate_helper(currentnode, current_str_node):
        printedprog = print_program(currentnode.program, ignore_constants=True)
        assert not enumerated.get(printedprog)
        enumerated[printedprog] = True
        
        if graph.is_fully_symbolic(currentnode.program): # a finished program
            all_programs.append({
                    "program" : copy.deepcopy(currentnode.program),
                    "struct_cost" : currentnode.cost,
                    "depth" : currentnode.depth
                })
        elif currentnode.depth < enumeration_depth: # an unfinished program (will be grown further)
            ## TODO under dev ## check for any type/size-matches
            all_children = graph.get_specific_children(currentnode, current_str_node, in_enumeration=True)
            for childnode in all_children: ## enumerate all possible children
                if not enumerated.get(print_program(childnode.program, ignore_constants=True)):
                    enumerate_helper(childnode, current_str_node)

    enumerate_helper(root, str_node)
    graph.max_depth = max_depth_copy

    assert len(all_programs) == 1
    return all_programs[0]




#### metrics for evaluating optimizers
class Metric():
    def __init__(self, metric_name, num_steps):
        self.metric_name = metric_name
        self.bound = {
            "final_valid_acc":0,
            "final_test_acc":0,
            
            "final_train_loss":0,
            "final_valid_loss":0,
            "final_test_loss":0,
            
            "cumm_train_loss":0,
            "cumm_valid_loss":0,
            "cumm_test_loss":0,
            
            "avg_train_loss":0,
            "avg_valid_loss":0,
            "avg_test_loss":0,
            
            "avg_distortion":0,
            "success_rate": 0,
            
            "accuracy":0,
            "spearmanr":0,
            "pearson":0,
            "matthews_correlation":0,
        }
        
        assert metric_name in self.bound.keys()
        
        for metric in self.bound:
            if 'acc' in metric:
                self.bound[metric] = 0.2
            if 'spearmanr' in metric or 'pearson' in metric:
                self.bound[metric] = 0.2
            if 'matthews_correlation' in metric:
                self.bound[metric] = 0.2
            if 'loss' in metric:
                self.bound[metric] = 15
            if 'cumm' in metric:
                self.bound[metric] *= num_steps
            if 'success_rate' in metric:
                self.bound[metric] = 0.12
            if 'distortion' in metric:
                self.bound[metric] = 1.0

        assert 'acc' in self.metric_name or 'success_rate' in self.metric_name or \
               'loss' in self.metric_name or 'distortion' in self.metric_name or \
               'spearmanr' in self.metric_name or 'pearson' in self.metric_name or \
               'matthews_correlation' in self.metric_name, \
               "must be one of them, see better_than()"
        self.reversed = 'acc' in self.metric_name or 'success' in self.metric_name or \
                        'spearmanr' in self.metric_name or 'pearson' in self.metric_name or \
                        'matthews_correlation' in self.metric_name


    def get_bound(self):
        return self.bound[self.metric_name]


    def argsort(self, values):
        sorted_inds = np.argsort(values) ## < by default
        if self.reversed:
            return np.flip(sorted_inds).copy()
        else:
            return sorted_inds


    def better_than(self, left, right): # (nan, acc)
        if np.isnan(left):  left = self.get_bound()
        if np.isnan(right): right = self.get_bound()
            
        less_than = self._less_than(left, right)
        if self.reversed: ## here
            return not less_than
        else:
            return less_than


    def best_of_dict(self, item_dict, key):
        if self.reversed: ## valid acc
            return max(item_dict, key=key)
        else:
            return min(item_dict, key=key)


    def _less_than(self, left, right):
        if np.isinf(right):
            return True
        return left < right
