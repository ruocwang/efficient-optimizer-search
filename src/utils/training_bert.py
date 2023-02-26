from copy import deepcopy
import random
import torch
import time
import dsl.functions as dsl_func
import numpy as np
import math
from utils.logging import log_and_print, print_program
from utils.utils import RANDOM_STATE, set_random_seeds, CONFIGS_ATTR

#### nos
from nos.train_utils_zoo import load_criterion, load_optimizer, load_scheduler
from torch.optim import SGD, Adam, AdamW, RMSprop


def execute_and_train_with_grid_search(program, config, args=None, return_raw_results=False, verbose=(True, True)):
    log_str = ''
    ## grid search
    results, log_str_gs = grid_search(program, config, args=args, verbose=verbose, report_mode=None)
    log_str += log_str_gs
    best_hps = results['hps']
    print('best_hps:', best_hps)
    log_str += f'best_hps: {best_hps}\n'
    
    ## early stopping
    metric = config['metric']
    bound = metric.get_bound()
    if not metric.better_than(results[config['metric_name']], bound):
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
        results, log_str_train, _ = execute_and_train(program, config, report_mode=None, hps=best_hps, args=args, verbose=verbose, early_stop=False, seed=seed)
        log_str += log_str_train
        all_scores.append(results[config['metric_name']])
        all_test_scores.append(results['final_test_acc'] if 'final_test_acc' in results else -1)
        all_results.append(results)
    ## log if more than one seed is evaluated
    assert verbose
    if config["num_seeds"] > 1 and verbose:
        message = "==> Final Valid: {:.2f}% ±{:.2f} | Final Test: {:.2f}% ±{:.2f}\n".format(
            np.mean(all_scores)*100, np.std(all_scores)*100, np.mean(all_test_scores)*100, np.std(all_test_scores)*100)
        log_and_print(message)
        log_str += message

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
            results, log_str_train, _ = execute_and_train(program, config, hps=hp_dict, args=args, verbose=verbose, report_mode=report_mode,
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
    elif args.optimizer == 'AdamW':
        opt_func = lambda x: AdamW(x, lr=lr, weight_decay=config['weight_decay'])
    elif args.optimizer == 'learned_opt':
        opt_func = lambda x: load_optimizer(list(x), program, features, total_steps, config, lr=lr)
    else:
        log_and_print(f'ERROR Optimizer: {args.optimizer}'); exit(1)


    ######## main
    s = time.time()
    bert_args = CONFIGS_ATTR(config)
    bert_args.seed = seed
    bert_args.num_train_epochs = num_epochs
    bert_args.task_name = bert_args.dataset
    best_valid_acc, best_test_acc, status, final_epoch, log_str_run = run(bert_args, opt_func)
    plot_dict = None
    log_str += log_str_run
    ########
    

    #### final evaluation
    results = {}
    results[metric_name] = max(best_valid_acc, metric.get_bound())
    results["below_bound"] = results[metric_name] == metric.get_bound()
    results["status"]  = status

    #### return a score
    print('duration:', time.time() - s)
    prog_str = print_program(program)
    message = '//////// {}: {:.4f} @lr={}'.format(metric_name, results[metric_name], lr)
    if status == 'terminated': message += f' (early stopped at {final_epoch})'
    message += '\n'
    if verbose: log_and_print(f'{prog_str}\n{message}')
    log_str += message
    return results, log_str, plot_dict


def run(args, opt_func):
    ## save current random states and then set a seed for this run
    random_state = RANDOM_STATE()
    random_state.save_state()
    set_random_seeds(args.seed)

    from nos_tasks.bert.run_glue_no_trainer import main
    best_val, best_test, status, epoch, log_str = main(args, opt_func)

    ## restore previous random state
    random_state.restore_state()

    return best_val, best_test, status, epoch, log_str

