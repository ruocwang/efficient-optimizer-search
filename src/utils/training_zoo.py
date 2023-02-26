from utils.logging import log_and_print

def load_training_task(args):
    if 'mnist' in args.config:
        log_and_print('loading mnistnet tasks...')
        from utils.training_mnistnet import execute_and_train, grid_search, execute_and_train_with_grid_search

    if 'conv' in args.config:
        log_and_print('loading image classification with convnet task...')
        from utils.training_conv import execute_and_train, grid_search, execute_and_train_with_grid_search

    if 'attack' in args.config:
        log_and_print('loading attack task...')
        from utils.training_adv import execute_and_train, grid_search, execute_and_train_with_grid_search

    if 'gat' in args.config:
        log_and_print('loading gat task...')
        from utils.training_gat import execute_and_train, grid_search, execute_and_train_with_grid_search

    if 'bert' in args.config:
        log_and_print('loading bert task...')
        from utils.training_bert import execute_and_train, grid_search, execute_and_train_with_grid_search

    return execute_and_train, grid_search, execute_and_train_with_grid_search
