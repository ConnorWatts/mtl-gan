from .generator import DCGANSNGenerator,TestGenerator

def get_generator(args):

    z_dim = args['z_dim']
    num_classes = args['num_classes']
    if args['nn_type'] == 'DCGAN-SN':
        return DCGANSNGenerator(z_dim)
    elif args['nn_type'] == 'Test':
        return TestGenerator(z_dim,num_classes)
    pass

def get_heads(args):
    
    pass