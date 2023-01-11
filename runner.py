import utils
from models.utils import get_generator, get_heads
from data.utils import get_data_loader
from torch.utils.tensorboard import SummaryWriter


class ModelRunner:

    def __init__(self,args):

        self.args = args

        self.writer = SummaryWriter()
        utils.set_random_seeds(args['seed'])
        self.device = utils.assign_device(args['device'])

        train_loader, test_loader, valid_loader = get_data_loader(args)

        self.generator = get_generator(args)
        self.heads = get_heads(args)
        self.noise_gen = utils.get_latent_noise(args,self.device)

        self.optimizer_g = utils.get_optimizer(args,'generator', self.generator.parameters())
        self.optimizer_heads = utils.get_optimizer(args, 'heads', self.heads.parameters())

        self.loss = utils.get_loss(args)

        

    def train(self):


        pass

    def train_epoch(self):
        pass

    def eval(self):
        pass
