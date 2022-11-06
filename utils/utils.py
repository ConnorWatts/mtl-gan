from models.model import Model
from models.generator import DCGANSNGenerator

def get_generator(params):
    z_dim = params.z_dim
    if params.nn_type == 'DCGAN-SN':
        return DCGANSNGenerator(z_dim)
    return True

def get_heads(params):
    return True

def get_model(params):
    """Create and return model"""
    generator, generator_channels = get_generator(params)
    heads = get_heads(generator_channels)
    model = Model(generator,heads,params)
    return model