from copy import deepcopy
import random
import torch
import time
import dsl.functions as dsl_func
import numpy as np
import math
from utils.logging import log_and_print, print_program

#### nos
from nos.train_utils_zoo import load_criterion, load_optimizer, load_scheduler
from torch.optim import SGD, Adam, RMSprop


class EarlyStop():
    ## metric == loss or acc
    ## step: check after how many steps/epochs
    ## limit: stop when num_fails/total > limit
    def __init__(self, min_epoch=0, max_epoch=300, warm_up=100, limit=4):
        self.state = []
        self.min_epoch = min_epoch
        self.max_epoch = max_epoch
        self.warm_up = warm_up
        self.limit = limit
        
    def check(self, best_val_acc, epoch): ## check every 10 epochs!
        self.state.append(np.round(best_val_acc, 4))
        
        if np.isnan(self.state).sum() + np.isinf(self.state).sum() > 0:
            return False
        
        if epoch <= self.min_epoch or epoch >= self.max_epoch:
            return True

        if self._no_progress():
            return False
        
        ## sagn extra: if warmup finished and not reaching 0.70
        if epoch == self.warm_up and best_val_acc < 0.70:
            return False

        return True
    
    def _no_progress(self):
        if len(self.state) < self.limit:
            return False
        ## best val acc stays the same for the latest 3 evals
        final = self.state[-1]
        for i in range(2, self.limit + 1):
            if self.state[-i] != final: return False
        return True


def execute_and_train_with_grid_search(program, config, args=None, return_raw_results=False, verbose=(True, True)):
    hierarchical = True if 'hierarchical' in config and config['hierarchical'] else False
    log_str = ''
    ## grid search
    results, log_str_gs = grid_search(program, config, args=args, verbose=verbose, hierarchical=hierarchical)
    log_str += log_str_gs
    best_hps = results['hps']
    print('best_hps:', best_hps)
    log_str += f'best_hps: {best_hps}\n'
    
    ## early stopping
    metric = config['metric']
    bound = metric.get_bound()
    if results[config['metric_name']] <= bound:
        if verbose: log_and_print(f"[WARNING] all lr produces nan loss or below bound {bound}, skip full eval")
        log_str += f"[WARNING] all lr produces nan loss or below bound {bound}, skip full eval\n"
        return None, -1, log_str
    elif args.skip_eval:
        if verbose: log_and_print(f"skip evaluation")
        log_str += f"skip evaluation\n"
        return [results[config['metric_name']]], best_hps, log_str


    ## multi-run eval using the searched lr
    all_scores = []
    all_test_scores  = []
    all_results = []
    for seed in range(config["num_seeds"]):
        results, log_str_train, _ = execute_and_train(program, config, hps=best_hps, args=args, verbose=verbose, early_stop=False, seed=seed, report_mode='search')
        log_str += log_str_train
        all_scores.append(results[config['metric_name']])
        all_test_scores.append(results['final_test_acc'])
        all_results.append(results)
    ## log if more than one seed is evaluated
    if config["num_seeds"] > 1 and verbose:
        log_and_print("==> Final Valid: {:.2f}% ±{:.2f} | Final Test: {:.2f}% ±{:.2f}\n".format(
            np.mean(all_scores)*100, np.std(all_scores)*100, np.mean(all_test_scores)*100, np.std(all_test_scores)*100))

    if return_raw_results:
        return all_results, best_hps, log_str
    else:
        return all_scores, best_hps, log_str


def make_hp_grids(hp_dict):
    name = 'lr'
    return [{name: lr} for lr in hp_dict[name]]


def grid_search(program, config, args=None, verbose=True, report_mode=None):
    log_str = ''
    num_epochs = config['num_epochs_gs'] if 'num_epochs_gs' in config else None
    metric = config['metric']
    best_value, best_results = metric.get_bound(), None
    
    for hp_dict in make_hp_grids(config['hps']):
        all_scores = []
        for seed in range(config['num_seeds_gs']):
            results, log_str_train, _ = execute_and_train(program, config, hps=hp_dict, args=args, verbose=verbose,
                                                       report_mode=report_mode,
                                                       num_epochs=num_epochs, early_stop=True, seed=seed)
            log_str += log_str_train
            all_scores.append(results[config['metric_name']])
        avg_score, std_score = np.mean(all_scores), np.std(all_scores)
        results['hps'] = hp_dict
        print(results['hps'], results[config['metric_name']])
        log_str += '{} {:.4f} ({:.4f})\n'.format(results['hps'], avg_score, std_score)
        
        if metric.better_than(avg_score, best_value):
            best_value = avg_score
            best_results = results

    #### final record
    if best_results is None: ## all nan
        best_results = results

    return best_results, log_str




def execute_and_train(program, config, hps, report_mode, args=None, early_stop=False, verbose=True, num_epochs=None, seed=0):
    log_str = ''
    #### training pipeline (generic)
    ## configs
    num_epochs      = config['num_epochs'] if num_epochs is None else num_epochs
    num_epochs_decay_max  = config['num_epochs_decay_max'] if 'num_epochs_decay_max' in config else -1
    metric_name     = config['metric_name']
    metric          = config['metric']
    features        = getattr(dsl_func, f"{config['dataset']}_feat")
    lr              = hps['lr']
    batch_size      = config['batch_size'] if 'batch_size' in config else -1
    dataset         = config['dataset']
    grad_clip       = config['grad_clip'] if 'grad_clip' in config else -1
    device = torch.cuda.current_device()

    if grad_clip > 0:
        assert 'cluster' in args.config, 'grad_clip only implemented for cluster-gat'


    if batch_size == -1: ## full batch algorithms
        step_per_epoch = 1
    elif 'num_partitions' in config: ## cluster-gat
        step_per_epoch = math.ceil(config['num_partitions'] / batch_size)
    elif dataset == 'ppi':
        step_per_epoch = math.ceil(20 / batch_size)

    if num_epochs_decay_max == -1:
        total_steps = num_epochs * step_per_epoch
    else:
        total_steps = num_epochs_decay_max * step_per_epoch

    ## model
    if args.optimizer == 'RMSprop':
        opt_func = lambda x: RMSprop(x, lr=lr)
    elif args.optimizer == 'Adam':
        opt_func = lambda x: Adam(x, lr=lr, weight_decay=config['weight_decay'])
    elif args.optimizer == 'learned_opt':
        opt_func = lambda x: load_optimizer(list(x), program, features, total_steps, config, lr=lr)
    else:
        log_and_print(f'ERROR Optimizer: {args.optimizer}'); exit(1)
    terminator = EarlyStop(max_epoch=240) if early_stop else None



    ######## main
    s = time.time()
    gat_args = config_to_args(config)
    if gat_args.dataset in ['arxiv', 'products']:
        gat_args.dataset = f'ogbn-{gat_args.dataset}'
    gat_args.seed = seed
    gat_args.n_epochs = num_epochs
    gat_args.num_epochs = num_epochs
    gat_args.base_lr = lr
    best_valid_acc, best_test_acc, status, final_epoch, log_str_run = run(gat_args, opt_func, terminator, device, verbose=verbose, fast=args.fast, grad_clip=grad_clip)
    log_str += log_str_run
    ########

    

    #### final evaluation
    results = {}
    results["final_valid_acc"] = max(best_valid_acc, metric.bound['final_valid_acc'])
    results["final_test_acc"] = max(best_test_acc, metric.bound['final_test_acc'])
    results["below_bound"] = results["final_valid_acc"] == metric.bound['final_valid_acc']
    results["status"]  = status


    #### return a score
    print('duration:', time.time() - s)
    prog_str = print_program(program)
    message = '//////// {}: {:.4f} (Test: {:.4f}) @lr={}'.format(metric_name, results[metric_name], results["final_test_acc"], lr)
    if status == 'terminated': message += f' (early stopped at {final_epoch})'
    if status == 'noenhanced': message += f' (no enhanced {final_epoch})'
    message += '\n'
    if verbose: log_and_print(f'{prog_str}\n{message}')
    log_str += message
    return results, log_str, None


class CONFIGS_ATTR():
    def __init__(self, config):
        for name in config:
            setattr(self, name, config[name])

def config_to_args(config):
    config_attr = CONFIGS_ATTR(config)
    return config_attr


def run(args, opt_func, terminator, device, fast=False, verbose=False, grad_clip=False):
    #### store random states and set seeds
    py_state = deepcopy(random.getstate())
    np_state = np.random.get_state()
    cpu_rng_state = torch.get_rng_state()
    gpu_rng_state_all = torch.cuda.get_rng_state_all() # return rng-state of all devices

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    
    if args.dataset == 'ogbn-products':
        from nos_tasks.gat_products.cluster_gat.main import run
        best_val, best_test, status, epoch, log_str = run(args, opt_func, terminator, device, fast, verbose=False, grad_clip=grad_clip)
    elif args.dataset == 'cora':
        from nos_tasks.gat_cora.train import main
        best_val, best_test, status, epoch, log_str = main(args, opt_func, terminator, device, fast, verbose=False)
    elif args.dataset in ['citeseer', 'pubmed']:
        from nos_tasks.gat_other.train import main
        best_val, best_test, status, epoch, log_str = main(args, opt_func, terminator, device, fast, verbose=False)
    elif args.dataset == 'ppi':
        from nos_tasks.gat_other.train_ppi import main
        best_val, best_test, status, epoch, log_str = main(args, opt_func, terminator, device, fast, verbose=False)


    #### restore previous random state
    random.setstate(py_state)
    np.random.set_state(np_state)
    torch.set_rng_state(cpu_rng_state)
    torch.cuda.set_rng_state_all(gpu_rng_state_all)

    return best_val, best_test, status, epoch, log_str
