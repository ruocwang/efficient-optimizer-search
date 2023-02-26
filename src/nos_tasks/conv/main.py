import torch
import numpy as np
from nos.model_zoo import load_model
from nos.train_utils_zoo import load_criterion, load_optimizer, load_scheduler
from nos.dataset_zoo import load_dataset


def main(args, opt_func):
    log_str = ''
    plot_dict = {}
    train_loader, test_loader, search_loader, valid_loader = load_dataset(args.dataset, args.batch_size, normalize_data=True) ## train = search + valid
    
    if args.report_mode == 'eval':
        search_loader = train_loader
        valid_loader = test_loader
    search_loader_iter  = iter(search_loader) ## preload to speedup

    model     = load_model(args.model).cuda()
    criterion = load_criterion(args.criterion)
    optimizer = opt_func(model.parameters())

    status = 'finished'
    train_losses, test_accs = [], []
    num_steps = 0
    for epoch in range(args.num_epochs):
        train_losses_epoch, test_accs_epoch, search_loader_iter, num_steps = train_one_epoch(search_loader, optimizer, model, criterion,
                                                                            args=args, search_loader_iter=search_loader_iter, test_loader=test_loader,
                                                                            num_steps=num_steps)
        train_losses += train_losses_epoch
        test_accs += test_accs_epoch

    best_val_loss, best_val_acc, best_test_loss, best_test_acc = 0, 0, 0, 0
    if args.report_mode == 'search':
        best_val_loss, best_val_acc, _ = inference(valid_loader, model, criterion)
    elif args.report_mode == 'eval':
        best_test_loss, best_test_acc, _ = inference(valid_loader, model, criterion)
    else:
        assert False, args.report_mod

    plot_dict['train_losses'] = train_losses
    plot_dict['test_accs'] = test_accs
    return (best_val_loss, best_val_acc), (best_test_loss, best_test_acc), status, epoch, log_str, plot_dict


def train_one_epoch(search_loader, learned_opt, model, criterion, args=None, search_loader_iter=None, test_loader=None, num_steps=0):
    if search_loader_iter is None:
        search_loader_iter = iter(search_loader)
    
    train_losses, test_accs, n_batch = [], [], 0
    for bid in range(len(search_loader)):
        model.train()
        try:
            inputs, targets = next(search_loader_iter)
        except:
            search_loader_iter = iter(search_loader)
            inputs, targets = next(search_loader_iter)
        inputs = inputs.float().cuda()
        targets = targets.long().cuda()
        
        learned_opt.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        learned_opt.step()
        
        train_losses.append(loss.item())
        n_batch += 1

        ## record
        if args.report_mode == 'eval' and args.plot and (num_steps % 100 == 0 or bid == len(search_loader) - 1):
            _, test_acc, _ = inference(test_loader, model, criterion)
            test_accs.append(test_acc)
        num_steps += 1

    return train_losses, test_accs, search_loader_iter, num_steps


def inference(eval_loader, model, criterion, eval_loader_iter=None):
    if eval_loader_iter is None:
        eval_loader_iter = iter(eval_loader)
    model.eval()
    correct, total = 0, 0
    test_loss, n_batch = 0, 0
    for bid in range(len(eval_loader)):
        with torch.no_grad():
            try:
                inputs, targets = next(eval_loader_iter)
            except:
                eval_loader_iter = iter(eval_loader)
                inputs, targets = next(eval_loader_iter)
                ## TODO under dev ##
                assert False
            
            inputs = inputs.float().cuda()
            targets = targets.long().cuda()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            n_batch += 1
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            test_loss += loss.item()
        
    test_loss = test_loss / n_batch
    test_acc = correct / total
    return test_loss, test_acc, eval_loader_iter