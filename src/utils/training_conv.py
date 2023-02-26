import random
import torch
import time
import dsl.functions as dsl_func
import numpy as np
import math
from copy import deepcopy
from utils.logging import log_and_print, print_program
from utils.utils import RANDOM_STATE, set_random_seeds, CONFIGS_ATTR

#### nos
from nos.train_utils_zoo import load_optimizer
from torch.optim import SGD, Adam, RMSprop


## TODO under dev ## double check
class EarlyStop():
    ## metric == loss or acc
    ## step: check after how many steps/epochs
    ## limit: stop when num_fails/total > limit
    def __init__(self, min_step=10, max_step=50, limit=0.75):
        self.state = []
        self.min_step = min_step
        self.max_step = max_step
        self.limit = limit
        
    def check(self, val, step):
        self.state.append(val)
        
        if np.isnan(self.state).sum() + np.isinf(self.state).sum() > 0:
            return False
        
        if step < self.min_step or step > self.max_step:
            return True

        if self._count_fails() / len(self.state) > self.limit:
            return False

        return True
    
    def _running_avg_loss(self, idx):
        window = 4
        if idx == 0:
            return self.state[0]
        else:
            return np.mean(self.state[max(0, idx - window):idx])
    
    def _count_fails(self):
        cnt_fails = sum( self._running_avg_loss(i - 1) < self._running_avg_loss(i) \
                         for i in range(1, len(self.state)) )
        return cnt_fails


def execute_and_train_with_grid_search(program, config, args=None, return_raw_results=False, verbose=(True, True)):
    log_str = ''
    ## grid search
    if args.fast:
        results = {config['metric_name']:abs(np.random.random()), 'hps':config['hps']}
        results['hps'] = {name:results['hps'][name][0] for name in results['hps']}
        best_hps = results['hps']
    else:
        results, log_str_gs = grid_search(program, config, args=args, verbose=verbose, report_mode='search')
        log_str += log_str_gs
        best_hps = results['hps']
    
    ## early stopping
    metric = config['metric']
    bound = metric.get_bound()
    if not metric.better_than(results[config['metric_name']], bound):
        if verbose: log_and_print(f"[WARNING] all lr produces nan loss or below bound {bound}, skip full eval")
        log_str += f"[WARNING] all lr produces nan loss or below bound {bound}, skip full eval\n"
        return None, -1, log_str

    ## multi-run eval using the searched lr
    if args.fast:
        all_scores = [min(np.random.random() + 0.2, 0.9)]
        all_results = [results]
    else:
        all_scores = []
        all_results = []
        for seed in range(config["num_seeds"]):
            results, log_str_train, _ = execute_and_train(program, config, hps=best_hps, args=args, verbose=verbose, report_mode='search')
            log_str += log_str_train
            all_scores.append(results[config['metric_name']])
            all_results.append(results)
    
    if return_raw_results:
        return all_results, best_hps, log_str
    else:
        return all_scores, best_hps, log_str


def make_hp_grids(hp_dict):
    name = 'lr'
    return [{name: lr} for lr in hp_dict[name]]


def grid_search(program, config, args=None, verbose=True, report_mode='search'):
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
    single_step     = config['single_step'] if 'single_step' in config else False
    metric_name     = config['metric_name']
    metric          = config['metric']
    features        = getattr(dsl_func, f"{config['dataset']}_feat")
    lr              = hps['lr']
    batch_size      = config['batch_size'] if 'batch_size' in config else -1

    device = torch.cuda.current_device()

    ## dataset
    if report_mode == 'search':
        steps_per_epoch = math.ceil(int(50000 * 0.8) / batch_size)
    else:
        steps_per_epoch = math.ceil(50000 / batch_size)
    total_steps = num_epochs if single_step else num_epochs * steps_per_epoch ## T

    ## model
    if args.optimizer == 'NAG':
        opt_func = lambda x: SGD(x, lr=lr, momentum=0.9, nesterov=True)
    elif args.optimizer == 'SGD':
        opt_func = lambda x: SGD(x, lr=lr, momentum=0.9)
    elif args.optimizer == 'G':
        opt_func = lambda x: SGD(x, lr=lr)
    elif args.optimizer == 'RMSprop':
        opt_func = lambda x: RMSprop(x, lr=lr)
    elif args.optimizer == 'Adam':
        opt_func = lambda x: Adam(x, lr=lr)
    elif args.optimizer == 'learned_opt':
        opt_func = lambda x: load_optimizer(list(x), program, features, total_steps, config, lr=lr)
    else:
        log_and_print(f'ERROR Optimizer: {args.optimizer}'); exit(1)


    ######## main
    s = time.time()
    conv_args = CONFIGS_ATTR(config)
    conv_args.seed = seed
    conv_args.num_epochs = num_epochs
    conv_args.report_mode = report_mode
    conv_args.plot = args.plot
    best_valid_acc, best_test_acc, status, final_epoch, log_str_run, plot_dict = run(conv_args, opt_func)
    log_str += log_str_run
    ########


    #### final evaluation
    results = {}
    if not args.clip_score:
        results[metric_name] = best_valid_acc
        results['final_test_acc'] = best_test_acc
    else:
        results[metric_name] = max(best_valid_acc, metric.get_bound())
        results['final_test_acc'] = max(best_test_acc, metric.get_bound())
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

    from nos_tasks.conv.main import main
    (_, best_val), (_, best_test), status, epoch, log_str, plot_dict = main(args, opt_func)

    ## restore previous random state
    random_state.restore_state()

    return best_val, best_test, status, epoch, log_str, plot_dict

