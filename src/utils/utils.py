import os
import glob
import time
import random
import numpy as np
import torch
from copy import deepcopy


class RANDOM_STATE():

    def save_state(self):
        self.py_state = deepcopy(random.getstate())
        self.np_state = np.random.get_state()
        self.cpu_rng_state = torch.get_rng_state()
        self.gpu_rng_state_all = torch.cuda.get_rng_state_all() # return rng-state of all devices

    def restore_state(self):
        random.setstate(self.py_state)
        np.random.set_state(self.np_state)
        torch.set_rng_state(self.cpu_rng_state)
        torch.cuda.set_rng_state_all(self.gpu_rng_state_all)


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CONFIGS_ATTR():
    """
        generate a fake args (argparse style) from config file
    """
    def __init__(self, config):
        for name in config:
            setattr(self, name, config[name])


def queue_gpu(safe=True, gpus=None, lock=None, args=None): # multi processing (if lock is not None)
    """ copied from rank-nosh """
    import gpustat

    count = {}
    for k in gpus.keys(): count[k] = 0
    safety = 10 #20 * 1 # s
    intv = 2 # s
    if args.gpu_capacity == 1: ## regular machines
        if args.config == 'cluster-gat':
            memory_max = 4000
            utility_max = 80
        elif args.config == 'dm':
            memory_max = 8000
            utility_max = 25
        elif args.config == 'const':
            memory_max = 10000
            utility_max = 100
            safety = 0
            intv = 0
        else:
            memory_max = 2000
            utility_max = 25
    else: ## kerrigan
        memory_max = 7000 * args.gpu_capacity
        utility_max = 50 * args.gpu_capacity
    duration = 0
    success = False
    
    while True:
        stats = gpustat.GPUStatCollection.new_query()
        
        # ids = list(map(lambda gpu: int(gpu.entry['index']), stats))
        utility = list(map(lambda gpu: int(gpu.entry['utilization.gpu']), stats))
        memory = list(map(lambda gpu: int(gpu.entry['memory.used']), stats))

        available_ids = []
        for i in count.keys():
            if utility[i] <= utility_max and memory[i] <= memory_max:
                count[i] += intv
            else: # reset counting
                count[i] = 0

            if (safe and count[i] >= safety) or (not safe and count[i] > 0):
                available_ids.append(i)

        ## found available gpu, check if it is occupied by other processes
        for available_id in available_ids:
            if lock is None: # single process
                success = True
            else: # multi process
                lock.acquire()
                if gpus[available_id] > 0: # not occupied
                    gpus[available_id] -= 1
                    success = True
                else: # reset counter
                    count[available_id] = 0
                lock.release()
            if success: break
        
        if success: break
        
        time.sleep(intv)
        duration += intv

    # print('found available gpu:', available_id)
    return available_id


def pick_gpu_lowest_memory():
    import gpustat
    stats = gpustat.GPUStatCollection.new_query()
    ids = map(lambda gpu: int(gpu.entry['index']), stats)
    ratios = map(lambda gpu: float(gpu.memory_used)/float(gpu.memory_total), stats)
    bestGPU = min(zip(ids, ratios), key=lambda x: x[1])[0]
    return bestGPU


def create_exp_dir(path, entry, run_script=None):
    import os
    import shutil
    if not os.path.exists(path):
        os.makedirs(path)
    # print('Experiment dir : {}'.format(path))
    
    script_path = os.path.join(path, 'scripts')
    if not os.path.exists(script_path):
        os.makedirs(script_path)

    tracked_items = getItemList('.', omit_list_file='.gitignore')
    for item in tracked_items:
        if 'exp_scripts' in item: item = run_script

        dst_item = os.path.join(script_path, os.path.basename(item))
        if './src' in item and item != f'./{entry}': continue ## only copy its own entry
        print(item, dst_item)
        if os.path.isdir(item):
            shutil.copytree(item, dst_item)
        else:
            shutil.copyfile(item, dst_item)


def getItemList(path, omit_list_file=None, omitted_paths=None):
    """ currently assume omit only contains paths with one level """
    def item_match(item1, item2):
        if item1[-1] == '/':  item1 = item1[:-1]
        if item1[:2] == './': item1 = item1[2:]
        if item2[-1] == '/':  item2 = item2[:-1]
        if item2[:2] == './': item2 = item2[2:]
        return item1 == item2

    # return nothing if path is a file
    if os.path.isfile(path):
        return []

    # get gitignored dirs
    if omitted_paths is None:
        omitted_paths = []
        if omit_list_file is not None:
            with open(os.path.join(path, omit_list_file), 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    omitted_paths.append(line)
    
    tracked_items = []
    for item in glob.glob(os.path.join(path, '*')):
        match = sum([item_match(item, it) for it in omitted_paths])
        if match == 0:
            tracked_items.append(item)
    return tracked_items


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