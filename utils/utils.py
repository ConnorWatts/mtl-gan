from models.model import Model

def get_generator(params):
    return True

def get_heads(params):
    return True

def get_model(params):
    """Create and return model"""
    generator, generator_channels = get_generator(params)
    heads = get_heads(generator_channels)
    model = Model(generator,heads,params)
