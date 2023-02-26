# task manager:
#     given N:
#         sample N programs
#         output their results
#         let the caller to hand the rest

#### it should never early exist
import time
import torch
import numpy as np
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

from utils.logging import log_and_print, print_program, print_program_dict
from utils.training_zoo import load_training_task
from utils.utils import queue_gpu


class Reference:
    def __init__(self, val):
        self._value = val # just refers to val, no copy

    def get(self):
        return self._value

    def set(self, val):
        self._value = val


class MANAGER():
    def __init__(self, metric):
        self.metric = metric
        
        self.ret_samples = [] # (parent_path, sample, sample_f_score, lr, status)
        self.trained_samples_dict = {} # store trained results once ready, update ret_samples later

        ## mp
        self.gpu_lock = mp.Lock()
        self.jid = 0


    def eval(self, N, sampler, visit, search_config, verbose=False, args=None):
        """
            N: number of UNIQUE programs to be returned
            sampler: a closure(), produce a valid sample of program
            
            return: a list of all sampled programs:
                1. duplicates count as separated entries
        """
        assert args.es_cnt_budget, "mp version always count es into budget for simplicity"
        self.N = N
        self.job_prog = {} ## record (child, sample) of submitted programs, as return_queue refuse to store these


        #### mp shared env setup
        processes = []
        manager = mp.Manager()
        self.gpus = manager.dict() # avoid gpu collision
        for i in range(8):
            if i in args.gpu:
                self.gpus[i] = args.gpu_capacity
        return_queue = mp.Queue() ## replace return_dict
        
        #### eval all children
        complete = True
        num_submitted = 0
        while True:
            sample, child = sampler()
            if sample is None:
                complete = False
                log_and_print('mc sampling failed, exit while loop'); break

            ## eval generated/sampled program
            cached_outcome = visit.is_evaluated(sample.program)

            if cached_outcome is not None: ## already evaluated, or is being evaluated
                if 'dataset' not in args.algorithm: ## dataset mode, do not record visited, hacky
                    status = 'cached'
                    best_lr = 'placeholder'
                    sample_score = 'placeholder'
                    # log_and_print(f'Visited, {print_program(sample.program, ignore_constants=(not verbose))}')
                    self.record_ret_samples(child, sample, sample_score, best_lr, status)
            else:
                ## do not submit too many jobs
                visit.record_eval_mp(sample.program)
                self.job_prog[print_program(sample.program)] = (child, sample)
                p = self.submit_job(child, sample, search_config, return_queue, verbose=verbose, args=args) ## stuck if no available GPUs
                p.start(); processes.append(p)
                num_submitted += 1
            
            if num_submitted >= self.N: ## early stopped count towards budget
                break

        log_and_print(f'<<<< all {num_submitted} jobs submitted, waiting for jobs to terminate >>>>')
        self.wait_for_all_jobs(processes)


        #### collect from return queue all at once
        ## return_queue contains trained and stopped optimizers
        log_and_print(f'return queue, {return_queue.empty()}')
        while not return_queue.empty():
            sample_prog_str, sample_score, best_lr, status, log_str = return_queue.get()
            child, sample = self.job_prog[sample_prog_str]
            log_and_print(f"Finished status={status}")
            log_and_print(log_str)
            self.update_trained_samples_dict(sample, sample_score, best_lr)
            self.record_ret_samples(child, sample, sample_score, best_lr, status)
        
        #### update sample pool (mpsafe)
        ## 1st pass: update "trained" entries in visit with acquired results
        log_and_print('update sample pool')
        for idx, (child, sample, sample_score, best_lr, status) in enumerate(self.ret_samples):
            prog_str = print_program(sample.program, ignore_constants=True)
            if status == 'trained' or status == 'stopped':
                trained_result = self.trained_samples_dict[prog_str]
                visit.update_record(sample.program, best_lr, sample_score)
                self.ret_samples[idx][2] = trained_result[0]
                self.ret_samples[idx][3] = trained_result[1]
        ## 2nd pass: update "cached" entries in visit with acquired results
        for idx, (child, sample, sample_score, best_lr, status) in enumerate(self.ret_samples):
            if status == 'cached':
                cached_outcome = visit.is_evaluated(sample.program)
                self.ret_samples[idx][2] = cached_outcome['sample_score']
                self.ret_samples[idx][3] = cached_outcome['best_lr']

        return self.ret_samples, complete

        
    
    def submit_job(self, child, sample, search_config, return_queue, verbose=False, args=None):
        ### run
        self.jid += 1
        locks = { 'gpu_lock':self.gpu_lock }
        job_args = (child, sample, search_config, verbose, args, self.gpus, locks, return_queue, self.jid)
        p = mp.Process(target=self.submit_job_helper, args=job_args)
        return p



    def submit_job_helper(self, child, sample, search_config, verbose=False, args=None, gpus=None, locks=None, return_queue=None, jid=-1):
        # print(f'---> job submitted {jid}')
        log_str = ''
        device = 'cpu' if 'synthetic' in search_config['dataset'] else 'gpu'
        ## CUDA device
        if device == 'gpu':
            gpu = queue_gpu(safe=False, gpus=gpus, lock=locks['gpu_lock'], args=args)
            time.sleep(0.5)
            torch.cuda.set_device(gpu)

        prog_str = print_program(sample.program, ignore_constants=(not verbose))
        log_and_print("\nTrain: {}".format(prog_str))
        log_str += "\nTrain: {}\n".format(prog_str)
        _, _, execute_and_train_with_grid_search = load_training_task(args)
        all_scores, best_lr, log_str_exe = execute_and_train_with_grid_search(sample.program, search_config, args=args, verbose=(False, False))
        log_str += log_str_exe
        
        if all_scores is None:
            status = 'stopped'
            sample_score, best_lr = self.metric.get_bound(), -1
        else:
            status = 'trained'
            sample_score = np.mean(all_scores)

        ## release device
        if not args.fast and device == 'gpu':
            torch.cuda.empty_cache()
            time.sleep(10)
            locks['gpu_lock'].acquire()
            gpus[gpu] += 1
            locks['gpu_lock'].release()

        #### return
        return_queue.put([prog_str, sample_score, best_lr, status, log_str])
        
        # return
        exit(0)


    def update_trained_samples_dict(self, sample, sample_score, best_lr):
        prog_str = print_program(sample.program, ignore_constants=True)
        self.trained_samples_dict[prog_str] = [sample_score, best_lr] ## mp
    
    
    def record_ret_samples(self, child, sample, sample_score, best_lr, status):
        self.ret_samples.append([child, sample, sample_score, best_lr, status])


    def wait_for_all_jobs(self, processes):
        for pid, p in enumerate(processes):
            p.join()
            # print('>>>>>>>>>> pid', pid, len(processes))