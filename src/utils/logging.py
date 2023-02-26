import logging
import os
import dsl.functions as dsl_func


def init_logging(save_path, tag=None, postfix=None):
    if postfix is None:
        logfile = os.path.join(save_path, 'log.txt')
    else:
        logfile = os.path.join(save_path, f'log{postfix}.txt')
        
    if os.path.exists(logfile):
        if 'debug' in tag or input(f'WARNING: logfile {logfile} exists, override? [y/n]') == 'y':
            # clear log file
            with open(logfile, 'w'):
                pass
        else:
            exit(0)

    # remove previous handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=logfile, level=logging.INFO)

def log_and_print(line):
    print(line)
    logging.info(line)
    return line

def print_program(program, ignore_constants=True):
    if not isinstance(program, dsl_func.LibraryFunction):
        return program.name
    else:
        collected_names = []
        for submodule, functionclass in program.submodules.items():
            collected_names.append(print_program(functionclass, ignore_constants=ignore_constants))
        if program.has_params:
            parameters = "params: {}".format(program.parameters.values())
            if not ignore_constants:
                collected_names.append(parameters)
        joined_names = ', '.join(collected_names)
        return program.name + "(" + joined_names + ")"

def print_program_dict(prog_dict):
    log_and_print(print_program(prog_dict["program"], ignore_constants=True))
    log_and_print("struct_cost {:.4f} | score {:.4f} | path_cost {:.4f} | time {:.4f}".format(
        prog_dict["struct_cost"], prog_dict["score"], prog_dict["path_cost"], prog_dict["time"]))
