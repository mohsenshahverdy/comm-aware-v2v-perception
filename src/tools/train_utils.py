# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib


import glob
import importlib
import yaml
import sys
import os
import re
from datetime import datetime

import torch
import torch.optim as optim
import timm


def find_last_checkpoint(save_dir):
    """Return the next epoch index implied by the latest checkpoint."""
    latest_path = os.path.join(save_dir, 'latest.pth')
    if os.path.exists(latest_path):
        checkpoint = torch.load(latest_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
            return int(checkpoint['epoch'])
        return 10000

    file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*epoch(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        return max(epochs_exist)

    return 0

def load_saved_model(saved_path, model):
    """
    Load saved model if exiseted

    Parameters
    __________
    saved_path : str
       model saved path
    model : src object
        The model instance.

    Returns
    -------
    model : src object
        The model instance loaded pretrained params.
    """
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)

    initial_epoch = find_last_checkpoint(saved_path)
    if initial_epoch > 0:
        latest_path = os.path.join(saved_path, 'latest.pth')
        model_file = latest_path if os.path.exists(latest_path) else \
            os.path.join(saved_path, 'net_epoch%d.pth' % initial_epoch)
        print('resuming by loading epoch %d' % initial_epoch)
        checkpoint = torch.load(
            model_file,
            map_location='cpu')
        model_state = checkpoint['model_state_dict'] \
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint \
            else checkpoint
        model.load_state_dict(model_state, strict=False)

        del checkpoint

    return initial_epoch, model


def load_training_state(saved_path, model, optimizer=None, scheduler=None,
                        scaler=None):
    """
    Load a resumable training checkpoint if available.

    Returns
    -------
    start_epoch : int
        Epoch index to continue from.
    """
    latest_path = os.path.join(saved_path, 'latest.pth')
    if not os.path.exists(latest_path):
        start_epoch, model = load_saved_model(saved_path, model)
        return start_epoch, model, optimizer, scheduler, scaler

    checkpoint = torch.load(latest_path, map_location='cpu')
    model_state = checkpoint['model_state_dict'] \
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint \
        else checkpoint
    model.load_state_dict(model_state, strict=False)

    if optimizer is not None and isinstance(checkpoint, dict) and \
            'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and isinstance(checkpoint, dict) and \
            'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if scaler is not None and isinstance(checkpoint, dict) and \
            checkpoint.get('scaler_state_dict') is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    start_epoch = int(checkpoint.get('epoch', 0)) \
        if isinstance(checkpoint, dict) else 0
    print('resuming full training state from epoch %d' % start_epoch)

    return start_epoch, model, optimizer, scheduler, scaler


def save_training_state(saved_path, epoch, model, optimizer, scheduler=None,
                        scaler=None):
    """
    Save a resumable training checkpoint to latest.pth.
    """
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    if scheduler is not None and hasattr(scheduler, 'state_dict'):
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    checkpoint['scaler_state_dict'] = \
        scaler.state_dict() if scaler is not None else None

    torch.save(checkpoint, os.path.join(saved_path, 'latest.pth'))


def setup_train(hypes):
    """
    Create folder for saved model based on current timestep and model name

    Parameters
    ----------
    hypes: dict
        Config yaml dictionary for training:
    """
    model_name = hypes['name']
    current_time = datetime.now()

    folder_name = current_time.strftime("_%Y_%m_%d_%H_%M_%S")
    folder_name = model_name + folder_name

    current_path = os.path.dirname(__file__)
    current_path = os.path.join(current_path, '../logs')

    full_path = os.path.join(current_path, folder_name)

    if not os.path.exists(full_path):
        if not os.path.exists(full_path):
            try:
                os.makedirs(full_path)
            except FileExistsError:
                pass
        # save the yaml file
        save_name = os.path.join(full_path, 'config.yaml')
        with open(save_name, 'w') as outfile:
            yaml.dump(hypes, outfile)

    return full_path



def create_model(hypes):
    """
    Import the module "models/[model_name].py

    Parameters
    __________
    hypes : dict
        Dictionary containing parameters.

    Returns
    -------
    model : src,object
        Model object.
    """
    backbone_name = hypes['model']['core_method']
    backbone_config = hypes['model']['args']

    model_filename = "src.models." + backbone_name
    model_lib = importlib.import_module(model_filename)
    model = None
    target_model_name = backbone_name.replace('_', '')

    for name, cls in model_lib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print('backbone not found in models folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (model_filename,
                                                       target_model_name))
        exit(0)
    instance = model(backbone_config)
    
    return instance


def create_loss(hypes):
    """
    Create the loss function based on the given loss name.

    Parameters
    ----------
    hypes : dict
        Configuration params for training.
    Returns
    -------
    criterion : src.object
        The loss function.
    """
    loss_func_name = hypes['loss']['core_method'] # point_pillar_loss
    loss_func_config = hypes['loss']['args']

    loss_filename = "src.loss." + loss_func_name # src.loss.point_pillar_loss
    loss_lib = importlib.import_module(loss_filename)
    loss_func = None
    target_loss_name = loss_func_name.replace('_', '')

    for name, lfunc in loss_lib.__dict__.items():
        if name.lower() == target_loss_name.lower():
            loss_func = lfunc

    if loss_func is None:
        print('loss function not found in loss folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (loss_filename,
                                                       target_loss_name))
        exit(0)

    print(f"Loss function: [{loss_func_name}] was created for this model.")
    criterion = loss_func(loss_func_config)
    return criterion


def setup_optimizer(hypes, model):
    """
    Create optimizer corresponding to the yaml file

    Parameters
    ----------
    hypes : dict
        The training configurations.
    model : src model
        The pytorch model
    """
    method_dict = hypes['optimizer']
    optimizer_method = getattr(optim, method_dict['core_method'], None)
    print('optimizer method is: %s' % optimizer_method)

    if not optimizer_method:
        raise ValueError('{} is not supported'.format(method_dict['name']))
    if 'args' in method_dict:
        return optimizer_method(filter(lambda p: p.requires_grad,
                                       model.parameters()),
                                lr=method_dict['lr'],
                                **method_dict['args'])
    else:
        return optimizer_method(filter(lambda p: p.requires_grad,
                                       model.parameters()),
                                lr=method_dict['lr'])


def setup_lr_schedular(hypes, optimizer, n_iter_per_epoch):
    """
    Set up the learning rate schedular.

    Parameters
    ----------
    hypes : dict
        The training configurations.

    optimizer : torch.optimizer
    """
    lr_schedule_config = hypes['lr_scheduler']

    if lr_schedule_config['core_method'] == 'step':
        from torch.optim.lr_scheduler import StepLR
        step_size = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif lr_schedule_config['core_method'] == 'multistep':
        from torch.optim.lr_scheduler import MultiStepLR
        milestones = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = MultiStepLR(optimizer,
                                milestones=milestones,
                                gamma=gamma)

    elif lr_schedule_config['core_method'] == 'exponential':
        print('ExponentialLR is chosen for lr scheduler')
        from torch.optim.lr_scheduler import ExponentialLR
        gamma = lr_schedule_config['gamma']
        scheduler = ExponentialLR(optimizer, gamma)

    elif lr_schedule_config['core_method'] == 'cosineannealwarm':
        print('cosine annealing is chosen for lr scheduler')
        from timm.scheduler.cosine_lr import CosineLRScheduler

        num_steps = lr_schedule_config['epoches'] * n_iter_per_epoch
        warmup_lr = lr_schedule_config['warmup_lr']
        warmup_steps = lr_schedule_config['warmup_epoches'] * n_iter_per_epoch
        lr_min = lr_schedule_config['lr_min']

        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=lr_min,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    else:
        sys.exit('not supported lr schedular')

    return scheduler


def to_device(inputs, device):
    if isinstance(inputs, list):
        return [to_device(x, device) for x in inputs]
    elif isinstance(inputs, dict):
        return {k: to_device(v, device) for k, v in inputs.items()}
    else:
        if isinstance(inputs, int) or isinstance(inputs, float) \
                or isinstance(inputs, str):
            return inputs
        return inputs.to(device)
