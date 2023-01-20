import torch
from torch import nn
import numpy as np
import random
import torch.optim as optim

from losses.loss_schemes import MultiTaskLoss
#from losses.loss_functions import 

#maybe move these inside utils folder

def get_optimizer(args,nn_type,nn_params):
    if nn_type == 'heads':
        learn_rate = args['lr_heads']
    elif nn_type == 'generator':
        learn_rate = args['lr_generator']
    if args['optimizer'] == 'Adam':
        optimizer = optim.Adam(nn_params, lr=learn_rate, weight_decay=args['weight_decay'], betas = (args['beta_1'],args['beta_2']))
    elif args['optimizer']  == 'SGD':
        optimizer = optim.SGD(nn_params, lr=learn_rate, momentum=args['sgd_momentum'], weight_decay=args['weight_decay'])
    else:
        raise NotImplementedError('Optimizer {} not recognised.'.format(args['optimizer']))    
    return optimizer


def wasserstein_dec_loss(true_data,fake_data):
    return -true_data.mean() + fake_data.mean()

def wasserstein_gen_loss(true_data,fake_data):
        return -fake_data.mean()


def get_dec_loss_ft(args,task):

    if task == 'gan':
        if args['gan_loss'] == 'classic':
            return nn.BCELoss()
        elif args['gan_loss'] == 'wasserstein':
            return wasserstein_dec_loss
    else:
        # make smarter
        return nn.CrossEntropyLoss()

def get_gen_loss_ft(args):
    if args['gan_loss'] == 'classic':
            return nn.BCELoss()
    elif args['gan_loss'] == 'wasserstein':
            return wasserstein_gen_loss




def get_loss(args,network_type):

    if network_type == 'generator':
        loss_ft = get_gen_loss_ft(args)
        return loss_ft

    elif network_type == 'decoders':
        tasks = args['tasks']
        loss_fts = torch.nn.ModuleDict({task: get_dec_loss_ft(args, task) for task in tasks})
        loss_weights = args['loss_weights']
        return MultiTaskLoss(tasks, loss_fts, loss_weights)



def get_scheduler(args):

    pass

def get_latent_noise(args,device):

    dim = int(args['z_dim'])

    if args['latent_noise'] =='gaussian':
        loc = torch.zeros(dim).to(device)
        scale = torch.ones(dim).to(device)
        normal = torch.distributions.normal.Normal(loc, scale)
        return torch.distributions.independent.Independent(normal,1)
    elif args['latent_noise'] =='uniform':
        return torch.distributions.Uniform(torch.zeros(dim).to(device),torch.ones(dim).to(device))
    else:
        raise NotImplementedError('Latent Distribution {} not recognised.'.format(args['latent_noise'] ))  

def get_class_dist(args,device):
    num_classes = args['num_classes']
    labels = 1.*np.array(range(num_classes))/num_classes
    labels= torch.tensor(list(labels)).to(device)
    return torch.distributions.categorical.Categorical(labels)

    pass

def set_random_seeds(seed):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

def assign_device(device):
    return 'cpu'

def get_coarse_label(data):
    # from https://github.com/ryanchankh/cifar100coarse/blob/master/sparse2coarse.py
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.
    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[data]

