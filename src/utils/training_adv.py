from copy import deepcopy
import torch
import time
import dsl.functions as dsl_func
import numpy as np
import os
from utils.logging import log_and_print, print_program
from collections import defaultdict
from foolbox import PyTorchModel
#### nos
from nos.model_zoo import load_model
from nos.dataset_zoo import load_dataset
from nos.train_utils_zoo import load_criterion, load_optimizer, load_scheduler
from torch.optim import SGD, Adam, RMSprop
from attacker.pgd import Linf_PGD, Linf_PGD_targeted
from autoattack import AutoAttack



class EarlyStop():
    ## metric == loss or acc
    ## step: check after how many steps/epochs
    ## limit: stop when num_fails/total > limit
    def __init__(self, metric, min_step=10, max_step=50, limit=0.75):
        self.state = []
        self.metric = metric
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


def get_acc(model, inputs, labels, return_idx=False):
    with torch.no_grad():
        predictions = model(inputs).argmax(axis=-1)
        accuracy = (predictions == labels).float().mean()
    if return_idx:
        return accuracy.item(), predictions == labels
    else:
        return accuracy.item()


def distance(x_adv, x, norm):
    diff = (x_adv - x).view(x.size(0), -1)
    if norm == 'L2':
        out = torch.sqrt(torch.sum(diff * diff) / x.size(0)).item()
        return out
    elif norm == 'Linf':
        out = torch.mean(torch.max(torch.abs(diff), 1)[0]).item()
        return out
    else:
        assert False, norm


def execute_and_train_with_grid_search(program, config, args=None, return_raw_results=False, verbose=(True, True)):
    log_str = ''
    ## grid search
    if args.fast:
        results = {config['metric_name']:abs(np.random.random()), 'hps':config['hps']}
        results['hps'] = {name:results['hps'][name][0] for name in results['hps']}
        best_hps = results['hps']
    else:
        results, log_str_gs = grid_search(program, config, args=args, verbose=verbose[0])
        log_str += log_str_gs
        best_hps = results['hps']


    ## early stopping
    metric = config['metric']
    bound = metric.get_bound()
    if results[config['metric_name']] <= bound:
        if verbose[0]: log_and_print(f"[WARNING] all lr produces nan loss or below bound {bound}, skip full eval")
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
            results, log_str_train = execute_and_train(program, config, hps=best_hps, args=args, verbose=verbose[1], early_stop=False)
            log_str += log_str_train
            all_scores.append(results[config['metric_name']])
            all_results.append(results)
    
    if return_raw_results:
        return all_results, best_hps, log_str
    else:
        return all_scores, best_hps, log_str



def make_hp_grids(hp_dict):
    name = 'lr'
    step = hp_dict['step'][0]
    return [{name: lr, 'step':step} for lr in hp_dict[name]]


def grid_search(program, config, args=None, verbose=True, report_mode=None):
    log_str = ''
    num_epochs = config['num_epochs_gs'] if 'num_epochs_gs' in config else None
    metric = config['metric']
    best_value, best_results = metric.get_bound(), None

    for hp_dict in make_hp_grids(config['hps']):
        results, log_str_train, _ = execute_and_train(program, config, hps=hp_dict, args=args, verbose=verbose,
                                                      report_mode=report_mode,
                                                      num_epochs=num_epochs, early_stop=False)
        log_str += log_str_train
        results['hps'] = hp_dict
        print(results['hps'], results[config['metric_name']])
        log_str += '{} {}\n'.format(results['hps'], results[config['metric_name']])

        if metric.better_than(results[config['metric_name']], best_value):
            best_value = results[config['metric_name']]
            best_results = results

    #### final record
    if best_results is None: ## all nan
        best_results = results

    return best_results, log_str




def execute_and_train(program, config, hps, args=None, early_stop=True, verbose=True,
                      report_mode=None, num_epochs=None, seed=0):
    assert not early_stop, "not necessary for adv attack"
    log_str = ''
    #### training pipeline (generic)
    ## configs
    num_epochs      = config['num_epochs'] if num_epochs is None else num_epochs
    single_step     = config['single_step'] if 'single_step' in config else False
    metric_name     = config['metric_name']
    metric          = config['metric']
    features        = getattr(dsl_func, f"{config['dataset']}_feat")
    lr              = hps['lr']

    step            = hps['step'][0] if isinstance(hps['step'], list) else hps['step']
    eps             = config['eps']
    task            = config['task']
    random_start    = config['random_start'] if 'random_start' in config else False
    norm            = 'Linf'
    n_target_classes = config['n_target_classes'] if 'n_target_classes' in config else 9

    ## dataset
    _, test_loader, _, _ = load_dataset(config['dataset'], config['batch_size'])
    search_loader = test_loader
    
    # preload iterators for efficiency (and deterministic data loading)
    cpu_rng_state = torch.get_rng_state()
    torch.manual_seed(400)
    search_loader_iter = iter(search_loader)
    torch.set_rng_state(cpu_rng_state)

    # total_steps = num_epochs if single_step else num_epochs*len(search_loader) ## T
    total_steps = step ## for adversarial attack

    ## model
    model = load_model(config['model'])
    if not isinstance(model, PyTorchModel): model.cuda()
    
    ## optimizer
    if args.optimizer == 'SignG':
        from attacker.sign_sgd import SignSGD
        opt_func = lambda x: SignSGD(x, lr=lr)
    elif args.optimizer == 'G':
        opt_func = lambda x: SGD(x, lr=lr)
    elif args.optimizer == 'MI-FGSM':
        from attacker.mi_fgsm import MI_FGSM
        opt_func = lambda x: MI_FGSM(x, lr=lr)
    elif args.optimizer == 'learned_opt':
        opt_func = lambda x: load_optimizer(x, program, features, total_steps, config, lr=lr)
    elif args.optimizer == 'learned_sign_opt':
        opt_func = lambda x: load_optimizer(x, program, features, total_steps, config, lr=lr, prefix_fn=torch.sign)
    elif args.optimizer == 'learned_log_opt':
        prefix_fn = lambda x: torch.log(torch.abs(x))
        opt_func = lambda x: load_optimizer(x, program, features, total_steps, config, lr=lr, prefix_fn=prefix_fn)
    elif args.optimizer == 'Adam':
        from torch.optim import Adam
        opt_func = lambda x: Adam(x, lr=lr)
    elif args.optimizer == 'Adamax':
        from torch.optim import Adamax
        opt_func = lambda x: Adamax(x, lr=lr)
    elif args.optimizer == 'RMSprop':
        from torch.optim import RMSprop
        opt_func = lambda x: RMSprop(x, lr=lr)
    elif args.optimizer == 'AA':
        opt_func = None
    else:
        log_and_print(f'ERROR Optimizer: {args.optimizer}'); exit(1)
    criterion   = load_criterion(config['criterion'])


    ## task
    if task == 'Linf_PGD':
        attack_f = Linf_PGD
    elif task == 'Linf_PGD_Targeted':
        attack_f = Linf_PGD_targeted
    elif task in ['AA', 'apgd-ce', 'apgd-t', 'fab', 'square']:
        adversary = AutoAttack(model, norm='Linf', eps=eps, verbose=False)
        if task != 'AA': adversary.attacks_to_run = [task] ## run specific task
        if n_target_classes != 9: adversary.apgd_targeted.n_target_classes = n_target_classes
        attack_f = adversary.run_standard_evaluation
    else:
        assert False, task


    ## perform attack task (like training)
    if not isinstance(model, PyTorchModel): model.eval()
    adv_accs, distortions, total = [], [], 0
    max_samples = num_epochs * search_loader.batch_size if single_step else exit(1)
    generated_advs, generated_ys = torch.tensor([]), torch.tensor([])
    s = time.time()
    while True:
        x, y = next(search_loader_iter)
        x, y = x.cuda(), y.cuda()

        inds_to_attack = torch.arange(x.shape[0])

        if task in ['AA', 'apgd-ce', 'apgd-t']:
            x_adv = attack_f(x, y)
            adv_acc = get_acc(model, x_adv, y) ## eval
        else:
            ## ignore already-misclassified ones
            if not isinstance(model, PyTorchModel): model.eval()
            with torch.no_grad():
                inds_to_attack = model(x).max(dim=-1).indices == y

            x_adv = attack_f(x, y, model, criterion, opt_func, step, eps,
                            random_start=random_start, inds_to_attack=inds_to_attack, n_target_classes=n_target_classes, use_best_adv=args.use_best_adv)
            adv_acc = get_acc(model, x_adv, y)

        adv_accs.append(adv_acc)
        distortions.append(distance(x_adv[inds_to_attack], x[inds_to_attack], norm=norm))
        generated_advs = torch.cat([generated_advs, x_adv.detach().cpu().clone()])
        generated_ys = torch.cat([generated_ys, y.detach().cpu().clone()])
        total += y.numel()

        if total >= max_samples: break

    ## logging
    results = defaultdict(list)
    results['avg_distortion'] = np.mean(distortions)
    results['success_rate'] = max(1 - np.mean(adv_accs), metric.bound["success_rate"])
    results["below_bound"] = results["success_rate"] == metric.bound["success_rate"]
    if verbose: print(results['avg_distortion'], results['success_rate'])

    ## save advs
    torch.save(generated_advs, os.path.join(args.ckpt_path, 'generated_advs.pth'))
    torch.save(generated_ys,   os.path.join(args.ckpt_path, 'generated_ys.pth'))

    #### return a score
    print('duration:', time.time() - s)
    message = '//////// {}: {:.4f} @lr={}'.format(metric_name, results[metric_name], lr)
    if verbose: log_and_print(message)
    log_str += message
    return results, log_str, None



def execute_and_train_ensemble(adv_list, config, args=None, early_stop=True, verbose=True, num_epochs=None):
    assert not early_stop, "not necessary for adv attack"
    log_str = ''
    #### training pipeline (generic)
    ## configs
    num_epochs      = config['num_epochs'] if num_epochs is None else num_epochs
    single_step     = config['single_step'] if 'single_step' in config else False
    metric_name     = config['metric_name']
    metric          = config['metric']
    batch_size      = config['batch_size']

    ## dataset
    _, test_loader, _, _ = load_dataset(config)
    search_loader = test_loader
    
    # preload iterators for efficiency (and deterministic data loading)
    cpu_rng_state = torch.get_rng_state()
    torch.manual_seed(400)
    search_loader_iter = iter(search_loader)
    torch.set_rng_state(cpu_rng_state)

    ## model
    model = load_model(config)
    if not isinstance(model, PyTorchModel): model.cuda()

    ## perform attack task (like training)
    if not isinstance(model, PyTorchModel): model.eval()
    adv_accs, total = [], 0
    max_samples = num_epochs * batch_size if single_step else exit(1)
    s = time.time()
    bid = 0
    while True:
        _, y = next(search_loader_iter)
        y = y.cuda()
        correct_inds_all = None
        for x_advs in adv_list:
            x_adv = x_advs[bid * batch_size: (bid + 1) * batch_size]
            x_adv = x_adv.cuda()

            _, correct_inds = get_acc(model, x_adv, y, return_idx=True)
            if correct_inds_all is None: correct_inds_all = correct_inds
            else: correct_inds_all = correct_inds_all & correct_inds

        adv_accs.append(correct_inds_all.float().mean().item())
        total += y.numel()
        bid += 1

        if total >= max_samples: break

    ## logging
    results = defaultdict(list)
    results['success_rate'] = max(1 - np.mean(adv_accs), metric.bound["success_rate"])
    results["below_bound"] = results["success_rate"] == metric.bound["success_rate"]
    if verbose: print(results['avg_distortion'], results['success_rate'])

    #### return a score
    print('duration:', time.time() - s)
    message = '//////// {}: {:.4f}'.format(metric_name, results[metric_name])
    if verbose: log_and_print(message)
    log_str += message
    return results, log_str
