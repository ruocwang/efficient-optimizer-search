#### it should never early exist
from collections import defaultdict
import copy
import os
import random
import time
import numpy as np
import torch
from algorithms.task_manager import MANAGER
import dsl.functions as dsl_func

from .core import ProgramLearningAlgorithm, ProgramNodeFrontier
from program_graph import ProgramGraph, ProgramNode
from utils.logging import log_and_print, print_program, print_program_dict
from utils.training_zoo import load_training_task
from algorithms.utils_search import Visit


#### pickle template ####
import pickle
def save_data(state, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(state, f)
def load_data(path):
    with open(path, 'rb') as f:
        state = pickle.load(f)
    return state
########################


class MC_SAMPLING_MP_NOS(ProgramLearningAlgorithm):

    def __init__(self, num_mc_samples=10):
        self.num_mc_samples = num_mc_samples # number of mc samples before choosing a child


    def run(self, graph, search_config, algo_config, verbose=False, args=None):
        self.args = args
        self.algo_config = algo_config
        assert isinstance(graph, ProgramGraph)


        current = copy.deepcopy(graph.root_node)
        best_program = None
        best_programs_list = []
        num_program_trained = 0  ## fully trained programs
        num_program_sampled = 0 ## also include early stopped programs
        num_program_cached  = 0
        start_time = time.time()

        features = getattr(dsl_func, f"{search_config['dataset']}_feat")
        visit = Visit(features, args.update)
        metric = search_config['metric']
        budget = args.budget
        topN = args.topN # topN programs to report


        candidate_pool = [] ## used to find topN programs to report, and it controls the budget
        traverse_level = 0
        while not graph.is_fully_symbolic(current.program):
            #### get all children of current
            children = graph.get_all_children(current, in_enumeration=True)
            children_mapping = { print_program(child.program, ignore_constants=True) : child for child in children }
            children_visited = defaultdict(dict)
            children_scores = { key : [] for key in children_mapping.keys() }
            costs = np.array([child.cost for child in children])
            costs = [c if c!=0 else min(costs[costs != 0]) for c in costs] ## give full programs are a chance
            traverse_level += 1


            ######## both trained samples and early stopped samples are returned
            max_concurrent_jobs = 32 if 'const' not in args.config else 80
            cur_level_ckpt_path = os.path.join(args.resume_ckpt, f'level_{traverse_level}.pth')
            if args.resume_ckpt != 'none' and os.path.exists(cur_level_ckpt_path):
                log_and_print(f'===> loading from resumed at level {traverse_level}')
                ret_samples, complete = load_data(cur_level_ckpt_path)
            else:
                curr_samples = 0
                ret_samples_all, complete_all = [], True
                while curr_samples < self.num_mc_samples:
                    ## reinit sampler closure (to update "visit")
                    def sampler():
                        while_cnt = 0
                        while True:
                            while_cnt += 1
                            if while_cnt > 4500:
                                return None, None
                            child = random.choices(children, weights=costs)[0]
                            sample = self.mc_sample(graph, child)
                            if not graph.is_fully_symbolic(sample.program): continue ## this is possible due to constrained generation

                            ## valid checker
                            if visit.semantic_valid(sample.program) and visit.descent_valid(sample.program):
                                return sample, copy.deepcopy(child)
                            else:
                                continue

                    manager = MANAGER(metric)
                    num_mc_samples = min(max_concurrent_jobs, self.num_mc_samples - curr_samples)
                    curr_samples += num_mc_samples
                    ret_samples, complete = manager.eval(num_mc_samples, sampler, visit, search_config, verbose=verbose, args=args)
                    ret_samples_all += ret_samples
                    complete_all = complete and complete_all
                    if not complete: break
                
                ret_samples = ret_samples_all
                complete = complete_all
                save_data([ret_samples, complete], os.path.join(args.ckpt_path, f'level_{traverse_level}.pth'))
            ######## return (parent_path, sample, sample_f_score, lr, status)
            ######## status is trained(s,lr)/stopped(0.2,-1)/cached(s,lr)


            #### record results (child_score for mc & candidate_pool for topN)
            for child, sample, sample_score, best_lr, status in ret_samples:
                ## record >0.2 programs to mc scores
                if metric.better_than(sample_score, metric.get_bound()):
                    child_prog_str = print_program(child.program, ignore_constants=True)
                    sampl_prog_str = print_program(sample.program, ignore_constants=True)
                    if self.algo_config['dup'] or sampl_prog_str not in children_visited[child_prog_str]: ## remove duplicate
                        children_scores[child_prog_str].append(sample_score)
                        children_visited[child_prog_str][sampl_prog_str] = True

                ## record trained programs for computing budget
                if status == 'trained':
                    log_and_print("Total program trained/sampled/cached {}/{}/{}"\
                                .format(num_program_trained, num_program_sampled, num_program_cached))
                    log_and_print(f'-----> {search_config["metric_name"]}: {sample_score} {best_lr}')
                    num_program_trained += 1
                    num_program_sampled += 1
                    candidate_pool.append([sample.program, sample_score])
                    if len(candidate_pool) >= budget: break
                elif status == 'stopped':
                    num_program_sampled += 1
                elif status == 'cached':
                    num_program_cached += 1
                else:
                    log_and_print(f'ERROR STATUS: {status}, exit')
            
            if len(candidate_pool) >= budget: break
            
            if not complete:
                log_and_print('mc sampling threshold exceeded, terminating search algo')
                break
            
            #### (Naive) selection operation
            ## score each child
            score_lowerbound = 0 if metric.reversed else float('inf')
            children_scores = { key : sum(val)/len(val) if len(val) > 0 and not graph.is_fully_symbolic(children_mapping[key].program) else score_lowerbound for key,val in children_scores.items() }
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
        log_and_print(f"\n\nTotal number of programs evaluated {len(candidate_pool)}")
        log_and_print(f"<<<< Top {topN} programs >>>>")
        sorted_inds = torch.sort(torch.tensor([it[1] for it in candidate_pool]), descending=metric.reversed).indices.numpy()
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
