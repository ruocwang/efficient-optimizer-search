import argparse
import os
import hjson
import random
from pprint import pformat
import shutil
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import time

from datetime import datetime
from algorithms import MC_SAMPLING_NOS, MC_SAMPLING_MP_NOS, RANDOM_NOS
from algorithms.utils_search import Metric
from dsl.spaces.dsl_conv_adv_v1 import CUSTOM_EDGE_COSTS
from dsl.spaces.space_zoo import get_search_space
from program_graph import ProgramGraph
from utils.logging import init_logging, print_program_dict, print_program, log_and_print
from utils.utils import pick_gpu_lowest_memory, create_exp_dir, save_data, load_data
from configs.utils import update_config

from datetime import datetime

if torch.__version__ > "1.7.1":
    print("For adversarial attack, use torch 1.7.x for reproducibility")


def parse_args():
    parser = argparse.ArgumentParser()
    #### experiment setup
    parser.add_argument('--group', type=str, help='experiment group')
    parser.add_argument('--save', type=str, required=True, help="experiment_name")
    parser.add_argument('--tag', type=str, default='none', help='extra tag for exp id')
    parser.add_argument('--entry', type=str, default='src', help='code entry')
    parser.add_argument('--resume_ckpt', type=str, default='none', help='resume searching')

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

    #### algorithm
    parser.add_argument('--algorithm', type=str, required=True,
                        help="the program learning algorithm to run")
    parser.add_argument('--num_mc_samples', type=int, required=False, default=10,
                        help="number of MC samples before choosing a child")
    parser.add_argument('--total_eval', type=int, required=False, default=100,
                        help="total number of programs to evaluate for genetic algorithm")
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
    parser.add_argument('--extra_algo_configs', type=str, default="none",
                        help='extra configs to specific algorithms')
    parser.add_argument('--prog_str', type=str, default="none",
                        help='e.g. Start(Grad()). If provided, evaluate/execute this single program')
    parser.add_argument('--optimizer', type=str, default="learned_opt",
                        help='use to determine which optimize to use')
    parser.add_argument('--budget', type=int, default=100,
                        help='number of unique programs visited')
    parser.add_argument('--topN', type=int, default=5,
                        help='report topN discovered programs')
    parser.add_argument('--dataset_ckpt', type=str, default="none",
                        help='load dataset for offline predictor tuning, do not override data directory')
    parser.add_argument('--dsl', type=str, default="none",
                        help='which dsl to load')
    parser.add_argument('--es_cnt_budget', type=int, default=0,
                        help='if true, early stopped programs also count toward budget')
    parser.add_argument('--constraint', type=int, default=0,
                        help='constrainted generation')
    parser.add_argument('--gpu_capacity', type=int, default=1,
                        help='how many jobs to allocate on a single gpu, for MP')
    parser.add_argument('--skip_eval', type=int, default=0,
                        help='skip evaluation (both after gs and final) of the optimizers')
    parser.add_argument('--use_best_adv', type=int, default=1,
                        help='return best_adv, as in AutoAttack')
    
    parser.add_argument('--update', type=int, default=0, help='produces full update instead of just gradient update')
    parser.add_argument('--clip_score', type=int, default=1, help='whether to clip scores (for ablation study')
    parser.add_argument('--plot', type=int, default=0, help='whether to generate plot data by aggressive evaluation')
    parser.add_argument('--hist', type=int, default=0, help='plotting for dataset')
    ####

    return parser.parse_args()
    



if __name__ == '__main__':
    args = parse_args()
    
    ## TODO under dev ## leftover from near, not used in ENOS
    args.input_type = 'atom'
    args.output_type = 'atom'
    args.input_size = 1
    args.output_size = 1


    #### args augment
    if args.seed == -1:
        now = datetime.now().strftime("%H%M%S")
        args.seed = int(now)
    if 'debug' in args.tag: args.group = 'debug'
    if 'dev'   in args.tag: args.group = 'dev'

    script_name = args.save
    exp_id = '{}'.format(script_name)
    exp_id += f'_[algo={args.algorithm}]'
    exp_id += f'_[budget={args.budget}]'
    exp_id += f'_[dsl={args.dsl}]'
    if args.optimizer != 'learned_opt': exp_id += f'_[dsl={args.optimizer}]'
    if args.extra_configs != 'none': exp_id += f'_[{args.extra_configs}]'
    if args.extra_algo_configs != 'none': exp_id += f'_[{args.extra_algo_configs}]'
    if args.config != 'attack': exp_id += f'_[{args.config}]'
    if args.constraint: exp_id += '_[ctr]'
    if args.max_depth != 10: exp_id += f'_[{args.max_depth}]'
    if args.tag and args.tag != 'none': exp_id += f'_[tag={args.tag}]'
    exp_id += f'_[seed={args.seed}]'
    if 'debug' in args.tag: exp_id = args.tag
    if args.group == 'none':
        args.save = os.path.join('/nfs/data/ruocwang/projects/experiments/comp-graph-search/experiments/default', exp_id)
    else:
        args.save = os.path.join('/nfs/data/ruocwang/projects/experiments/comp-graph-search/experiments/', f'{args.group}/', exp_id)
    if args.dataset_ckpt != 'none':
        args.save = args.dataset_ckpt.replace('dataset.pth', '')

    ## override path
    args.log_path = os.path.join(args.save, 'logs')
    args.ckpt_path = os.path.join(args.save, 'ckpts')
    if args.dataset_ckpt == 'none' and args.resume_ckpt == 'none':
        if os.path.exists(args.save):
            if 'debug' in args.tag or 'dev' in args.tag or \
            input(f'Exp {exp_id} exists, override? [y/n]') == 'y': shutil.rmtree(args.save)
            else: exit()
        create_exp_dir(args.save, entry=args.entry, run_script='./exp_scripts/{}'.format(script_name + '.sh'))
        os.mkdir(args.log_path)
        os.mkdir(args.ckpt_path)
    
    ## logging
    postfix = args.prog_str if args.prog_str != 'none' else None
    postfix = '_pred_offline' if args.dataset_ckpt != 'none' else postfix
    postfix = '_resume' if args.resume_ckpt != 'none' else postfix
    init_logging(args.log_path, args.tag, postfix=postfix)
    if args.fast:
        log_and_print("WARNING!!!!\n" * 3)
        log_and_print("fast mode\n" + "-" * 20)
    log_and_print("\nStarting experiment {} on SERVER={} ENV={}\n".format(exp_id, os.uname()[1], os.environ['CONDA_DEFAULT_ENV']))
    log_and_print(datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
    log_and_print('\n================== Args ==================\n')
    log_and_print(pformat(vars(args)))


    #### env
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.algorithm[-3:] == '-mp':
        gpus = [int(gid) for gid in args.gpu.split(',')]
        args.gpu = gpus
        torch.multiprocessing.set_start_method('spawn')
    else:
        gpu = pick_gpu_lowest_memory() if args.gpu == 'auto' else int(args.gpu)
        args.gpu = gpu
        torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.enabled = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    log_and_print('gpu device = ' + str(args.gpu))


    #### Initialize program graph
    DSL_DICT, CUSTOM_EDGE_COSTS, VOCAB = get_search_space(args.dsl)
    program_graph = ProgramGraph(DSL_DICT, CUSTOM_EDGE_COSTS,
                                 args.input_type, args.output_type, args.input_size, args.output_size,
                                 args.max_num_units, args.min_num_units, args.max_num_children, args.max_depth,
                                 args.penalty, ite_beta=args.ite_beta,
                                 constraint=args.constraint, update=args.update)


    #### Search algorithms
    if args.algorithm == "mc-sampling":
        algorithm = MC_SAMPLING_NOS(num_mc_samples=args.num_mc_samples)
    elif args.algorithm == "mc-sampling-mp":  # multi-process
        algorithm = MC_SAMPLING_MP_NOS(num_mc_samples=args.num_mc_samples)
    elif args.algorithm == "random":
        algorithm = RANDOM_NOS()

    else:
        raise NotImplementedError


    #### Run program learning algorithm
    assert args.optimizer in ['learned_opt', 'learned_sign_opt', 'learned_log_opt']
    log_and_print("\n\n======== RUN FULL SEARCH ========\n")
    search_config = hjson.load(open(f'{args.entry}/configs/task_configs/{args.config}-search.hjson'))
    search_config = update_config(search_config, args.extra_configs)
    search_config['type'] = 'search'
    log_and_print("-"*20 + f"\nsearch config: {hjson.dumps(search_config, indent=4)}\n" + "-"*20)
    metric = Metric(search_config['metric_name'], search_config['num_epochs'])
    search_config['metric'] = metric
    
    algo_config = hjson.load(open(f'{args.entry}/configs/algo_configs/{args.algorithm}.hjson'))
    algo_config = update_config(algo_config, args.extra_algo_configs)
    log_and_print("-"*20 + f"\nalgo config: {hjson.dumps(algo_config, indent=4)}\n" + "-"*20)
    
    s = time.time()
    best_programs = algorithm.run(program_graph, search_config, algo_config, args=args)
    log_and_print('Duration: {:.2f} (h)'.format((time.time() - s)/60/60))

    # log best program
    log_and_print("\n")
    log_and_print("TopN programs found:")
    for item in best_programs:
        print_program_dict(item)
        log_and_print('-'*60)
    save_data(best_programs, os.path.join(args.log_path, "topn_programs.pth"))
    
    
    #### evaluate discovered programs
    if not args.skip_eval:
        from eval import evaluate_program
        for best_program in reversed(best_programs): ## eval top-1 program first
            evaluate_program(best_program, args)

