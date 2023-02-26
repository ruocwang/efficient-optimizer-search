import copy
import random
import time
import numpy as np
import torch
import dsl

from .core import ProgramLearningAlgorithm
from program_graph import ProgramGraph, ProgramNode
from utils.logging import log_and_print, print_program, print_program_dict
from utils.training_zoo import load_training_task
from algorithms.utils_search import Visit


class RANDOM_NOS(ProgramLearningAlgorithm):

    def __init__(self):
        pass


    def run(self, graph, search_config, verbose=False, args=None):
        assert isinstance(graph, ProgramGraph)

        current = copy.deepcopy(graph.root_node)
        best_program = None
        best_programs_list = []
        num_program_trained = 0
        num_program_sampled = 0 ## valid, unique programs sampled

        features = getattr(dsl, f"{search_config['dataset']}_feat")
        visit = Visit(features)
        metric = search_config['metric']
        budget = args.budget
        topN = args.topN # topN programs to report
        
        _, _, execute_and_train_with_grid_search = load_training_task(args)


        candidate_pool = [] ## used to find topN programs to report
        while len(candidate_pool) < budget:
            ## randomly pick a child and generate/sample a full program from it
            sample = self.random_sample(graph, current)
            assert graph.is_fully_symbolic(sample.program)
            
            ## duplicate programs
            if visit.is_evaluated(sample.program): continue

            ## valid checker
            if not visit.semantic_valid(sample.program) or not visit.descent_valid(sample.program): continue

            ## eval generated/sampled program
            log_and_print("Train: {}".format(print_program(sample.program, ignore_constants=(not verbose))))
            all_scores, best_lr = execute_and_train_with_grid_search(sample.program, search_config, args=args, verbose=False)
            visit.record_eval(sample.program, best_lr, all_scores)
            num_program_sampled += 1

            if all_scores is None: ## early stopped programs do not consume our budget
                continue
            else:
                sample_score = np.mean(all_scores)

            num_program_trained += 1
            candidate_pool.append([copy.deepcopy(sample.program), sample_score])
            
            log_and_print("Total program trained/sampled/cached {}/{}".format(num_program_trained, num_program_sampled))
            log_and_print(f'//////// {search_config["metric_name"]}: {sample_score} {best_lr}')


        #### sort all discovered programs (from entire candidate pool)
        log_and_print(f"total number of programs evaluated {len(candidate_pool)}")
        log_and_print(f"<<<< Top {topN} programs >>>>")
        sorted_inds = torch.sort(torch.tensor([it[1] for it in candidate_pool]),
                                 descending=metric.reversed).indices.numpy()
        top_inds = np.flip(sorted_inds[:topN]) ## flip so that the best is at the bottom
        for idx in top_inds:
            sample_program, sample_score = candidate_pool[idx]
            sample_f_score = sample_score
            ## record best program (top N actually)
            best_program = copy.deepcopy(sample_program)
            # best_value = sample_f_score
            best_programs_list.append({
                    "program" : best_program,
                    "struct_cost" : -1,
                    "score" : sample_score,
                    "path_cost" : sample_f_score,
                    "time" : -1
                })
            print_program_dict(best_programs_list[-1])


        return best_programs_list


    #### randomly generate a full program by expanding from program_node
    #### bias towards paths with higher cost
    def random_sample(self, graph, program_node):
        assert isinstance(program_node, ProgramNode)
        while not graph.is_fully_symbolic(program_node.program):
            children = graph.get_all_children(program_node, in_enumeration=True)
            program_node = random.choices(children)[0]
        return program_node
