import utils
from models.utils import get_generator, get_heads
from data.utils import get_data_loader
from torch.utils.tensorboard import SummaryWriter


class ModelRunner:

    def __init__(self,args):

        self.args = args
        self.max_train_epoch = args['max_train_epochs']
        self.batch_size_train = args['batch_size_train']

        self.writer = SummaryWriter()
        utils.set_random_seeds(args['seed'])
        self.device = utils.assign_device(args['device'])

        self.train_loader, self.test_loader, self.valid_loader = get_data_loader(args)

        self.generator = get_generator(args)
        self.heads = get_heads(args)

        self.latent_noise_gen = utils.get_latent_noise(args,self.device)
        self.class_dist = utils.get_class_dist(args,self.device)

        self.optimizer_g = utils.get_optimizer(args,'generator', self.generator.parameters())
        #self.optimizer_heads = utils.get_optimizer(args, 'heads', self.heads.parameters())

        self.loss = utils.get_loss(args)

        #maybe think about setting 

        

    def train(self):

        # maybe make ConditionalNoiseGen class(see GEBM)

        z = self.latent_noise_gen.sample([self.batch_size_train])
        c = self.class_dist.sample([self.batch_size_train])
        z_ = z.view(-1, 100, 1, 1)
        output = self.generator(z,c)
        print('Hello')

        #for epoch in range(self.max_train_epoch):

            #self.train_epoch()


    def train_epoch(self):

        for batch_idx, re in enumerate(self.train_loader):
            check = re


    def eval(self):
        pass
