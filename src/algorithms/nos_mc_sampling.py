#### it should never early exist
import copy
import random
import time
import numpy as np
import torch
import dsl.functions as dsl_func

from .core import ProgramLearningAlgorithm
from program_graph import ProgramGraph, ProgramNode
from utils.logging import log_and_print, print_program, print_program_dict
from utils.training_zoo import load_training_task
from algorithms.utils_search import Visit
from collections import defaultdict


class MC_SAMPLING_NOS(ProgramLearningAlgorithm):

    def __init__(self, num_mc_samples=10):
        self.num_mc_samples = num_mc_samples # number of mc samples before choosing a child


    def run(self, graph, search_config, algo_config, verbose=False, args=None):
        self.args = args
        self.algo_config = algo_config
        assert isinstance(graph, ProgramGraph)
        

        current = copy.deepcopy(graph.root_node)
        best_program = None
        best_programs_list = []
        num_program_trained = 0
        num_program_sampled = 0 ## valid, unique programs sampled
        num_program_cached  = 0
        start_time = time.time()

        features = getattr(dsl_func, f"{search_config['dataset']}_feat")
        visit = Visit(features, args.update)
        metric = search_config['metric']
        budget = args.budget
        topN = args.topN # topN programs to report
    
        _, _, execute_and_train_with_grid_search = load_training_task(args)
        

        candidate_pool = [] ## used to find topN programs to report, and it controls the budget
        # program_list = [] ## TODO under dev ## rebuttal
        terminate = False
        while not graph.is_fully_symbolic(current.program):
            #### get all children of current
            children = graph.get_all_children(current, in_enumeration=True)
            children_mapping = { print_program(child.program, ignore_constants=True) : child for child in children }
            children_scores = { key : [] for key in children_mapping.keys() }
            children_visited = defaultdict(dict)
            costs = np.array([child.cost for child in children])
            costs = [c if c!=0 else min(costs[costs != 0]) for c in costs] ## give full programs a chance

            #### eval all children
            mc_effective_idx = 0
            while_break_cnt = 0
            while mc_effective_idx < self.num_mc_samples:

                #### sampler
                while_break_cnt += 1
                if while_break_cnt > 4500:
                    log_and_print('mc sampling threshold exceeded, exit the inner while loop')
                    break

                ## randomly pick a child and generate/sample a full program from it
                child = random.choices(children, weights=costs)[0]
                sample = self.mc_sample(graph, child)
                if not graph.is_fully_symbolic(sample.program): continue ## this is possible due to constrained generation

                ## valid checker
                if not visit.semantic_valid(sample.program) or not visit.descent_valid(sample.program): continue


                ######## main entry
                ## eval generated/sampled program
                cached_outcome = visit.is_evaluated(sample.program)

                if cached_outcome is not None: ## already evaluated, but need to record score
                    log_and_print('Visited')
                    num_program_cached += 1
                    best_lr = cached_outcome["best_lr"]
                    all_scores = cached_outcome['sample_score']
                else:
                    log_and_print("Train: {}".format(print_program(sample.program, ignore_constants=(not verbose))))
                    all_scores, best_lr, _ = execute_and_train_with_grid_search(sample.program, search_config, args=args, verbose=(False, True))
                    visit.record_eval(sample.program, best_lr, all_scores)
                    num_program_sampled += 1

                if all_scores is None: ## early stopped programs do not consume our budget
                    continue
                else:
                    sample_score = np.mean(all_scores)
                ######## 


                #### record results
                if metric.better_than(sample_score, metric.get_bound()) or not args.clip_score: ## only score valid programs
                    child_prog_str = print_program(child.program, ignore_constants=True)
                    sampl_prog_str = print_program(sample.program, ignore_constants=True)
                    if self.algo_config['dup'] or sampl_prog_str not in children_visited[child_prog_str]: ## remove duplicate
                        children_scores[child_prog_str].append(sample_score)
                        children_visited[child_prog_str][sampl_prog_str] = True

                if not cached_outcome: # newly evaluated, not early-stopped programs
                    candidate_pool.append([sample.program, sample_score])
                    num_program_trained += 1
                    mc_effective_idx += 1
                
                log_and_print("Total program trained/sampled/cached {}/{}/{}"\
                              .format(num_program_trained, num_program_sampled, num_program_cached))
                log_and_print(f'-----> {search_config["metric_name"]}: {sample_score} {best_lr}')
                
                ## termination criterion
                if len(candidate_pool) >= budget or num_program_sampled > int(budget * 3.5):
                    terminate = True
                    break

            if terminate: break
            #### (Naive) selection operation
            ## score each child
            score_lowerbound = 0 if metric.reversed else float('inf')
            new_children_scores = {}
            for key,val in children_scores.items():
                if len(val) > 0 and not graph.is_fully_symbolic(children_mapping[key].program): ## has child and not fully symbolized
                    val = sorted(val, reverse=metric.reversed)[:self.algo_config['topn']]
                    new_children_scores[key] = sum(val)/len(val)
                else:
                    new_children_scores[key] = score_lowerbound
            children_scores = new_children_scores
            ## pick the best one and move on
            best_child_name = metric.best_of_dict(children_scores, children_scores.get)
            current = children_mapping[best_child_name]
            current_avg_f_score = children_scores[best_child_name]
            for key,val in children_scores.items():
                log_and_print("Avg score {:.4f} for child {}".format(val,key))
            log_and_print("SELECTING {} as best child node\n".format(best_child_name))
            log_and_print("DEBUG: time since start is {:.3f}\n".format(time.time()-start_time))

            
            #### eval program on the main path
            log_and_print("CURRENT program has avg fscore {:.4f}: {}".format(
                current_avg_f_score, print_program(current.program, ignore_constants=(not verbose))))


        #### sort all discovered programs (from entire candidate pool)
        log_and_print(f"total number of programs evaluated {len(candidate_pool)}")
        log_and_print(f"<<<< Top {topN} programs >>>>")
        sorted_inds = torch.sort(torch.tensor([it[1] for it in candidate_pool]),
                                 descending=metric.reversed).indices.numpy()
        top_inds = np.flip(sorted_inds[:topN]) ## flip so that the best is at the bottom
        for idx in top_inds:
            sample_program, sample_score = candidate_pool[idx]
            sample_f_score = sample_score
            best_program = copy.deepcopy(sample_program)
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
    def mc_sample(self, graph, program_node):
        assert isinstance(program_node, ProgramNode)
        while not graph.is_fully_symbolic(program_node.program):
            children = graph.get_all_children(program_node, in_enumeration=True)
            if self.algo_config['sample_prob'] == 'cost':
                weights = [child.cost for child in children]
            elif self.algo_config['sample_prob'] == 'uniform':
                weights = None
            
            if len(children) == 0: ## cannot finish the program within max length
                break

            program_node = random.choices(children, weights=weights)[0]

        return program_node
