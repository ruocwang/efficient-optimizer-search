import torch
from nos.model_zoo import load_model
from nos.train_utils_zoo import load_criterion, load_optimizer, load_scheduler
from nos.dataset_zoo import load_dataset


def main(args, opt_func, terminator, device, fast=False, verbose=False):
    log_str = ''
    plot_dict = {}
    train_loader, test_loader, search_loader, valid_loader = load_dataset(args.dataset, args.batch_size) ## train = search + valid

    if args.report_mode == 'eval':
        search_loader = train_loader
        valid_loader = test_loader
    search_loader_iter  = iter(search_loader) ## preload to speedup

    model     = load_model(args.model).cuda()
    criterion = load_criterion(args.criterion)
    optimizer = opt_func(model.parameters())

    status = 'finished'
    train_losses = []
    num_steps = 0
    while True:
        # inference_closure = lambda : inference(valid_loader, model, criterion)
        train_losses_epoch, search_loader_iter, num_steps = train_one_epoch(search_loader, optimizer, model, criterion,
                                                            args=args, search_loader_iter=search_loader_iter, num_steps=num_steps)
        train_losses += train_losses_epoch
        if num_steps >= args.max_steps:
            break
    
    best_val_loss, best_val_acc, best_test_loss, best_test_acc = 0, 0, 0, 0
    if args.report_mode == 'search':
        best_val_loss, best_val_acc, _ = inference(valid_loader, model, criterion)
    if args.report_mode == 'eval' or 'test' in args.metric_name:
        best_test_loss, best_test_acc, _ = inference(test_loader, model, criterion)

    plot_dict['train_losses'] = train_losses
    return train_losses, (best_val_loss, best_val_acc), (best_test_loss, best_test_acc), status, num_steps, log_str, plot_dict


def train_one_epoch(search_loader, learned_opt, model, criterion, args=None, search_loader_iter=None, num_steps=0):
    if search_loader_iter is None:
        search_loader_iter = iter(search_loader)

    train_losses, n_batch = [], 0
    model.train()
    for bid in range(len(search_loader)):
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
        
        num_steps += 1
        if num_steps >= args.max_steps:
            break

    # train_loss /= n_batch
    return train_losses, search_loader_iter, num_steps


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
