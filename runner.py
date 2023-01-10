import utils
from models.utils import get_generator, get_heads
from data.utils import get_data_loader
from torch.utils.tensorboard import SummaryWriter


class ModelRunner:

    def __init__(self,args):

        self.writer = SummaryWriter()

        train_loader, test_loader, valid_loader = get_data_loader(args)

        self.generator = get_generator(args)
        self.heads = get_heads(args)

        self.optimizer = utils.get_optimizer(args)

        self.loss = utils.get_loss(args)

        utils.set_random_seeds(args['seed'])
        self.device = utils.assign_device(args['device'])

    def train(self):
        pass

    def train_epoch(self):
        pass

    def eval(self):
        pass
