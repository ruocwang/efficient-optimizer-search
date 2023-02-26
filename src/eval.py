import argparse
import os
from this import d
import hjson
import random
from pprint import pformat
import shutil
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import time

# import program_learning
import dsl
from tqdm import tqdm
from datetime import datetime
from algorithms.utils_search import Metric, Visit
from dsl.spaces.space_zoo import get_search_space
from program_graph import ProgramGraph
from utils.logging import init_logging, print_program_dict, print_program, log_and_print
from utils.utils import pick_gpu_lowest_memory, create_exp_dir, save_data, load_data
from configs.utils import update_config
from algorithms.utils_search import prog_str2program


if torch.__version__ > "1.7.1":
    print("For adversarial attack, use torch 1.7.x for reproducibility")


def parse_args():
    parser = argparse.ArgumentParser()
    #### experiment setup
    parser.add_argument('--group', type=str, help='experiment group')
    parser.add_argument('--save', type=str, required=True, help="experiment_name")
    parser.add_argument('--tag', type=str, default='none', help='extra tag for exp id')
    parser.add_argument('--log_tag', type=str, default='none', help='extra tag for log file')
    parser.add_argument('--entry', type=str, default='src', help='no need to change this')

    #### program graph
    parser.add_argument('--max_num_units', type=int, required=False, default=16,
                        help="max number of hidden units for neural programs")
    parser.add_argument('--min_num_units', type=int, required=False, default=4,
                        help="max number of hidden units for neural programs")
    parser.add_argument('--max_num_children', type=int, required=False, default=10,
                        help="max number of children for a node")
    parser.add_argument('--max_depth', type=int, required=False, default=8,
                        help="max depth of programs")
    parser.add_argument('--penalty', type=float, required=False, default=0.01,
                        help="structural penalty scaling for structural cost of edges")
    parser.add_argument('--ite_beta', type=float, required=False, default=1.0,
                        help="beta tuning parameter for if-then-else")
    #### other
    parser.add_argument('--fast', action='store_true', default=False,
                        help="fast mode for debugging the entire pipeline")
    parser.add_argument('--gpu', type=str, default='auto',
                        help='gpu device id')
    parser.add_argument('--seed', type=int, default=2,
                        help='random seed')
    parser.add_argument('--config', type=str, default="none",
                        help='config filename (prefix) for nos')
    parser.add_argument('--extra_configs', type=str, default="none",
                        help='extra configs to override config file')
    parser.add_argument('--libname', type=str, default="default",
                        help='name of optimizer library')
    parser.add_argument('--prog_str', type=str, default="none",
                        help='e.g. Start(Grad()). If provided, evaluate/execute this single program')
    parser.add_argument('--optimizer', type=str, default="learned_opt",
                        help='use to determine which optimize to use')
    parser.add_argument('--program_ckpt', type=str, default="none",
                        help='program.pth, holds a list of topN programs')
    parser.add_argument('--dsl', type=str, default="none",
                        help='which dsl to load')
    parser.add_argument('--deter', type=int, default=0,
                        help='fully deterministic mode (slow)')
    
    parser.add_argument('--update', type=int, default=0, help='produces full update instead of just gradient update')
    parser.add_argument('--clip_score', type=int, default=1, help='whether to clip scores (for ablation study')
    parser.add_argument('--plot', type=int, default=0, help='whether to generate plot data by aggressive evaluation')
    ####

    return parser.parse_args()


def evaluate_program(best_program, args):
    #### loading evaluation task
    from utils.training_zoo import load_training_task
    execute_and_train, grid_search, _ = load_training_task(args)
    
    #### Evaluate best program on test setting
    log_and_print("="*20 + "\n" + "evaluating the best discovered program")
    log_and_print(print_program(best_program['program'], ignore_constants=False))
    import sympy as sp
    log_and_print(sp.powsimp(best_program['program'].execute_sym()))
    
    eval_config = hjson.load(open(f'src/configs/task_configs/{args.config}-eval.hjson'))
    eval_config = update_config(eval_config, args.extra_configs)
    eval_config['type'] = 'eval'
    log_and_print("-"*20 + f"\neval config: {hjson.dumps(eval_config, indent=4)}\n" + "-"*20)
    metric = Metric(eval_config['metric_name'], eval_config['num_epochs'])
    eval_config['metric'] = metric


    ## learning rate grid search (use train/valid split)
    s = time.time()
    if hasattr(args, 'prog_lr'): ## program best learning rate provided in opt_library
        log_and_print('===> use stored best learning rate')
        best_hps = eval_config["hps"]
        best_hps['lr'] = args.prog_lr
    elif isinstance(eval_config['hps']['lr'], list):
        hierarchical = True if 'hierarchical' in eval_config and eval_config['hierarchical'] else False
        results, _ = grid_search(best_program['program'], eval_config, args=args, verbose=True, hierarchical=hierarchical)
        best_hps = results['hps']
    else: ## use provided learning rate
        log_and_print('===> use provided learning rate')
        best_hps = eval_config["hps"]
    log_and_print('best_hps:')
    log_and_print(str(best_hps))
    

    ## eval program for multiple runs (use train/test split)
    all_scores = []
    all_plot_dict = []
    for seed in tqdm(range(eval_config["num_seeds"])):
        results, _, plot_dict = execute_and_train(best_program['program'], eval_config,
                                hps=best_hps, args=args, early_stop=False, verbose=True, report_mode='eval', seed=seed)
        
        ## plotting
        all_plot_dict.append(plot_dict)

        ## logging
        scores = []
        if 'train' in eval_config['metric_name']:
            scores += [results[eval_config['metric_name']]]
        if 'valid' in eval_config['metric_name'] or 'final_valid_acc' in results:
            scores += [results['final_test_acc']]
        if 'success_rate' in eval_config['metric_name']:
            scores += [results['success_rate']]
        if 'avg_distortion' in eval_config['metric_name']:
            scores += [results['avg_distortion']]
        scores += [results[eval_config['metric_name']]]
        all_scores.append(scores)
        print(scores)


    ## logging
    print("total duration:", time.time() - s)
    avg_scores = []
    all_scores = np.array(all_scores) # (seeds, metrics)
    avg_scores = all_scores.mean(axis=0)
    std_scores = all_scores.std(axis=0)
    idx = 0
    if 'train' in eval_config['metric_name']:
        if 'loss' in eval_config['metric_name']:
            log_and_print("Eval: {}: {:.5f} ±{:.5f}".format(eval_config['metric_name'], avg_scores[idx], std_scores[idx]));idx+=1
        else:
            log_and_print("Eval: {}: {:.2f} ±{:.2f}".format(eval_config['metric_name'], avg_scores[idx], std_scores[idx]));idx+=1
    if 'valid' in eval_config['metric_name'] or 'final_valid_acc' in results:
        log_and_print("Eval: {}: {:.2f}% ±{:.2f}".format('final_test_acc', avg_scores[idx]*100, std_scores[idx]*100));idx+=1
    if 'success_rate' in eval_config['metric_name']:
        log_and_print("Eval: {}: {:.2f}% ±{:.2f}".format('success_rate', avg_scores[idx]*100, std_scores[idx]*100));idx+=1
    if 'avg_distortion' in eval_config['metric_name']:
        log_and_print("Eval: {}: {:.2f}% ±{:.2f}".format('avg_distortion', avg_scores[idx]*100, std_scores[idx]*100));idx+=1

    log_and_print("Eval: {}: {:.2f}% ±{:.2f}".format(eval_config['metric_name'], avg_scores[idx]*100, std_scores[idx]*100));idx+=1

    log_and_print("EXP END \n")
    print(all_scores.T)

    ## plotting
    plot_data_save_path = os.path.join(args.ckpt_path, 'all_plot_dict.pth')
    torch.save(all_plot_dict, plot_data_save_path)
    log_and_print(f'raw data path: {plot_data_save_path}')



def get_stock_optimizers(libname, prog_str):
    import nos.opt_library as optlib
    opt_library = getattr(optlib, libname)
    if prog_str == 'all':
        return list(opt_library.values())
    else:
        return opt_library[prog_str]



if __name__ == '__main__':
    args = parse_args()

    ## TODO under dev ## leftover from near, not used in ENOS
    args.input_type = 'atom'
    args.output_type = 'atom'
    args.input_size = 1
    args.output_size = 1


    #### args augment
    if args.prog_str != 'none' and '()' not in args.prog_str: ## evaluating stock optimizers
        ret = get_stock_optimizers(args.libname, args.prog_str)
        if isinstance(ret, list):
            args.prog_str, args.prog_lr = ret
        else:
            args.prog_str = ret

    if args.seed == -1:
        now = datetime.now().strftime("%H%M%S")
        args.seed = int(now)

    script_name = args.save
    if args.program_ckpt != 'none':
        args.save = args.program_ckpt[:args.program_ckpt.rfind('logs')-1]
    else:
        exp_id = '{}'.format(script_name)
        exp_id += '_[cfg={}]'.format(args.config)
        exp_id += '_[excfg={}]'.format(args.extra_configs)
        if args.optimizer != 'learned_opt': exp_id += f'_[opt={args.optimizer}]'
        if args.prog_str  != 'none':
            if isinstance(args.prog_str, list):
                exp_id += f'_[opt=esb-{args.libname}]'
            else:
                exp_id += f'_[opt={args.prog_str}]'
        if args.tag != 'none': exp_id += f'_[tag={args.tag}]'
        exp_id += f'_[seed={args.seed}]'
        if 'debug' in args.tag: exp_id = args.tag
        args.save = os.path.join('/nfs/data/ruocwang/projects/experiments/comp-graph-search/experiments/eval-new', args.group, exp_id)
    ## override path
    args.log_path = os.path.join(args.save, 'logs')
    args.ckpt_path = os.path.join(args.save, 'ckpts')
    if args.program_ckpt == 'none':
        if os.path.exists(args.save):
            if ('debug' in args.tag) or (input(f'Exp {exp_id} exists, override? [y/n]') == 'y'):
                shutil.rmtree(args.save)
            else:
                exit()
        create_exp_dir(args.save, args.entry, run_script='./exp_scripts/{}'.format(script_name + '.sh'))
        os.mkdir(args.log_path)
        os.mkdir(args.ckpt_path)

    ## logging
    log_postfix = '_eval'
    if args.extra_configs != 'none':
        log_postfix += f'_[{args.extra_configs}]'
    if args.optimizer != 'learned_opt':
        log_postfix += f'_[{args.optimizer}]'
    if args.log_tag != 'none':
        log_postfix += f'_[tag-{args.log_tag}]'
    init_logging(args.log_path, postfix=log_postfix, tag=args.tag)
    if args.fast:
        log_and_print("WARNING!!!!\n" * 3)
        log_and_print("fast mode\n" + "-" * 20)
    log_and_print(f"\nStarting evaluation on SERVER={os.uname()[1]}, ENV={os.environ['CONDA_DEFAULT_ENV']}\n")
    log_and_print('\n================== Args ==================\n')
    log_and_print(pformat(vars(args)))



    #### env
    random.seed(args.seed)
    np.random.seed(args.seed)
    gpu = pick_gpu_lowest_memory() if args.gpu == 'auto' else int(args.gpu)
    args.gpu = gpu
    torch.cuda.set_device(gpu)

    if args.deter:
        torch.backends.cudnn.deterministic = True
        cudnn.benchmark = False
        # cudnn.enabled = False
    else:
        cudnn.benchmark = True
        cudnn.enabled = True ## must enable, otherwise cannot reproduce AA's results
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    log_and_print('gpu device = %d' % gpu)


    if torch.cuda.is_available(): device = f'cuda:{args.gpu}'
    else: device = 'cpu'


    # Initialize program graph
    DSL_DICT, CUSTOM_EDGE_COSTS, VOCAB = get_search_space(args.dsl)
    program_graph = ProgramGraph(DSL_DICT, CUSTOM_EDGE_COSTS,
                                 args.input_type, args.output_type, args.input_size, args.output_size,
                                 args.max_num_units, args.min_num_units, args.max_num_children, args.max_depth,
                                 args.penalty, ite_beta=args.ite_beta)


    log_and_print("\n\n======== EVAL PROGRAMS ========\n")
    if args.prog_str != 'none':
        assert args.optimizer in ['learned_opt', 'learned_sign_opt', 'learned_log_opt'] and args.program_ckpt == 'none'
        log_and_print(f'prog_str provided, eval {args.prog_str}')
        best_program = prog_str2program(program_graph, args.prog_str, enumeration_depth=9999)
        best_programs = [best_program]
    elif args.optimizer != 'learned_opt':
        assert args.prog_str == 'none' and args.program_ckpt == 'none'
        log_and_print(f'eval factory opt {args.optimizer}')
        args.prog_str = 'Start(1())' ## dummy
        best_program = prog_str2program(program_graph, args.prog_str, enumeration_depth=9999)
        best_programs = [best_program]
    else:
        assert args.prog_str == 'none' and args.optimizer == 'learned_opt'
        log_and_print('eval topN programs')
        best_programs = load_data(args.program_ckpt)


    #### evaluate discovered programs
    for best_program in reversed(best_programs): ## eval top-1 program first
        evaluate_program(best_program, args)

