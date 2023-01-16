import torch

from .heads import Head
from .discriminator import DCGANSNDiscriminator
from .generator import DCGANSNGenerator
from .decoder import MultiTaskDecoder


def get_generator(args):

    z_dim = args['z_dim']
    num_classes = args['num_classes']
    if args['nn_type'] == 'DCGAN-SN':
        return DCGANSNGenerator(z_dim,num_classes)

def get_shared_decoder(args):
    if args['nn_type'] == 'DCGAN-SN':
        # arb channel number at the moment 
        return DCGANSNDiscriminator(), 7


def get_head(args,channels,task):

    # find a way to assign number of classes to each task
    # maybe use channels as well

    if task == 'gan':
        return Head(2)
    elif task == 'fine':
        return Head(100)
    elif task == 'coarse':
        return Head(10)
    else:
        raise NotImplementedError('Task {} not recognised.'.format(task))  



def get_decoders(args):

    tasks = args['tasks']
    shared_decoder, shared_decoder_channels = get_shared_decoder(args)
    heads = torch.nn.ModuleDict({task: get_head(args, shared_decoder_channels, task) for task in tasks})
    return MultiTaskDecoder(shared_decoder,heads,tasks)