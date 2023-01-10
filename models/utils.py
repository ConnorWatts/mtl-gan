from .generator import DCGANSNGenerator

def get_generator(args):

    z_dim = args['z_dim']
    if args['nn_type'] == 'DCGAN-SN':
        return DCGANSNGenerator(z_dim)
    pass

def get_heads(args):
    
    pass