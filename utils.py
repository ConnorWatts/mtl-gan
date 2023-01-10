import torch
import numpy as np
import random
import torch.optim as optim

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

def get_loss(args):

    pass

def get_scheduler(args):

    pass

def get_latent_noise(args,dim,device):
    dim = int(dim)
    if args['latent_noise'] =='gaussian':
        loc = torch.zeros(dim).to(device)
        scale = torch.ones(dim).to(device)
        normal = torch.distributions.normal.Normal(loc, scale)
        return torch.distributions.independent.Independent(normal,1)
    elif args['latent_noise'] =='uniform':
        return torch.distributions.Uniform(torch.zeros(dim).to(device),torch.ones(dim).to(device))
    else:
        raise NotImplementedError('Latent Distribution {} not recognised.'.format(args['latent_noise'] ))  

def set_random_seeds(seed):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

def assign_device(device):
    pass
