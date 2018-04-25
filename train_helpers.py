import sys
sys.path.append('..')
from dataset import prepare_data_loaders, TARGET_LABEL_NAMES
from models import prepare_model

import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torchnet import meter

import numpy as np

from collections import OrderedDict
import json
from os.path import basename, join, exists
from os import listdir, makedirs
import shutil


def load_hparams(hparams_path):
    if not exists(hparams_path):
        raise Exception('You must provide path to existing hparams.json file')
    
    with open(hparams_path, 'r') as f:
        hparams = json.load(f, object_pairs_hook=OrderedDict)
    
    return hparams


def prepare_training(hparams):
    """Converts given hyperparameters to instances of classes.
       Arguments:
           hparams: dict of hyperparameters
       Returns:
           model, criterion, optimizer, scheduler: classes for training
    """
    if 'model_params' not in hparams:
        raise Exception('You must add model params to hparams')
    
    model = prepare_model(hparams['model_params'])
    
    if 'criterion_params' not in hparams or \
        'criterion' not in hparams['criterion_params']:
        raise Exception('You must add criterion params to hparams')
    
    criterion_params = hparams['criterion_params']
    criterion_name = criterion_params.pop('criterion')
    criterion = torch.nn.__dict__[criterion_name](**criterion_params)
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    
    if 'optimizer_params' not in hparams or \
        'optimizer' not in hparams['optimizer_params']:
        raise Exception('You must add optimizer params to hparams')
    
    optimizer_params = hparams['optimizer_params']
    optimizer_name = optimizer_params.pop('optimizer')
    optimizer = torch.optim.__dict__[optimizer_name](
        filter(lambda p: p.requires_grad, model.parameters()),
        **optimizer_params
    )
    
    if 'scheduler_params' in hparams:
        scheduler_params = hparams['scheduler_params']
        if 'scheduler' not in scheduler_params:
            raise Exception('If you provided scheduler params you also must add scheduler name')
        scheduler_name = scheduler_params.pop('scheduler')
        
        scheduler = torch.optim.lr_scheduler.__dict__[scheduler_name](
            optimizer, **scheduler_params
        )
    else:
        scheduler = None
    
    return model, criterion, optimizer, scheduler


def get_val_from_metric(metric_value):
    """Converts the metric value to a single number."""
    if isinstance(metric_value, (int, float)):
        return metric_value
    else:
        metric_value = metric_value.value()
        if isinstance(metric_value, tuple):
            metric_value = metric_value[0]
        return metric_value


def run_train_val_loader(epoch, loader, mode, model, criterion, optimizer):
    """Runs one epoch of the training loop.
    Arguments:
        epoch: index of the epoch
        loader: data loader
        mode: 'train' or 'valid'
        model: model for training \ validation
        criterion: loss function to minimize
        optimizer: optimisation function
    """
    if mode == 'train':
        model.train()
    else:
        model.eval()
    
    epoch_metrics = {
        "loss": meter.AverageValueMeter(),
        "confusion_matrix": meter.ConfusionMeter(k=len(TARGET_LABEL_NAMES))
    }
    
    for i, batch in enumerate(loader):
        target = batch.pop('target')
        batch_size = len(target)
        
        if torch.cuda.is_available():
            input_var = {
                key: torch.autograd.Variable(value.cuda(async=True), requires_grad=False)
                for key, value in batch.items()
            }
        else:
            input_var = {
                key: torch.autograd.Variable(value, requires_grad=False)
                for key, value in batch.items()
            }
        
        if torch.cuda.is_available():
            target = target.cuda(async=True).type(torch.cuda.LongTensor)
        else:
            target = target.type(torch.LongTensor)
        target_var = torch.autograd.Variable(target, requires_grad=False)
        
        output = model.forward(input_var)
        loss = criterion(output, target_var)
        
        epoch_metrics['loss'].add(float(loss.data.cpu().numpy()))
        epoch_metrics['confusion_matrix'].add(output.data, target_var.data)
        
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        epoch_metrics['batch_size'] = batch_size
    
    out_metrics = {key: get_val_from_metric(value) for key, value in epoch_metrics.items()}
#    epoch_metrics_str = "\t".join([
#        "{key} {value:.4f}".format(key=key, value=value)
#        for key, value in sorted(out_metrics.items())])
    epoch_metrics_str = "loss\t{}".format(get_val_from_metric(epoch_metrics['loss']))
    print("{epoch} * Epoch ({mode}): ".format(epoch=epoch, mode=mode) + epoch_metrics_str)
    
    return out_metrics


def save_checkpoint(state, is_best, logdir):
    if not exists(logdir):
        makedirs(logdir)
    
    filename = "{}/checkpoint.pth.tar".format(logdir)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}/checkpoint.best.pth.tar'.format(logdir))


def run_train(hparams, args):
    model, criterion, optimizer, scheduler = prepare_training(hparams)
    
    best_loss = int(1e10)
    best_metrics = None
    start_epoch = 0
    train_metric_history, val_metric_history = [], []
    
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            best_metrics = checkpoint['best_metrics']
            train_metric_history = checkpoint['train_metric_history']
            val_metric_history = checkpoint['val_metric_history']

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise Exception("no checkpoint found at '{}'".format(args.resume))
    
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        # speed up
        cudnn.benchmark = True

    train_loader, valid_loader = prepare_data_loaders(hparams)
    
    if 'training_params' not in hparams:
        raise Exception('You must provide training_params in hparams')
    
    training_params = hparams['training_params']
    if 'epochs' not in training_params or 'batch_size' not in training_params:
        raise Exception('You must add epochs and batch_size parameters into hparams')
    
    print('Training started')
    
    for epoch in range(start_epoch, training_params['epochs']):
        epoch_train_metrics = run_train_val_loader(epoch, train_loader, 'train', model, criterion, optimizer)
        epoch_val_metrics = run_train_val_loader(epoch, valid_loader, 'valid', model, criterion, optimizer)
        
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_val_metrics["loss"])
        else:
            scheduler.step()
        
        train_metric_history.append(epoch_train_metrics)
        val_metric_history.append(epoch_val_metrics)
        
        # remember best loss and save checkpoint
        is_best = epoch_val_metrics["loss"] < best_loss
        best_loss = min(epoch_val_metrics["loss"], best_loss)
        best_metrics = epoch_val_metrics if is_best else best_metrics
        best_metrics = {
            key: value for key, value in best_metrics.items()
            if isinstance(value, float)}
        
        save_checkpoint({
            "epoch": epoch + 1,
            "best_loss": best_loss,
            "best_metrics": best_metrics,
            "val_metrics_history": val_metric_history,
            "train_metrics_history": train_metric_history,
            "model": model.module,
            "model_state_dict": model.module.state_dict(),
            "optimizer": optimizer,
            "optimizer_state_dict": optimizer.state_dict(),
        }, is_best, logdir=args.logdir)
