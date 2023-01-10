import torch
import numpy as np
import random

#maybe move these inside utils folder

def get_optimizer(args):
    pass

def get_loss(args):

    pass

def set_random_seeds(seed):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

def assign_device(device):
    pass
