import torch

import utils
from models.utils import get_generator, get_decoders
from data.utils import get_data_loader
from torch.utils.tensorboard import SummaryWriter


class ModelRunner:

    def __init__(self,args):

        self.args = args
        self.max_train_epoch = args['max_train_epochs']
        self.batch_size_train = args['batch_size_train']

        self.writer = SummaryWriter()
        utils.set_random_seeds(args['seed'])

        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        #self.device = utils.assign_device(args['device'])

        self.train_loader, self.test_loader, self.valid_loader = get_data_loader(args)

        self.generator = get_generator(args)
        self.decoders = get_decoders(args)

        self.latent_noise_gen = utils.get_latent_noise(args,self.device)
        self.class_dist = utils.get_class_dist(args,self.device)

        self.optimizer_g = utils.get_optimizer(args,'generator', self.generator.parameters())
        self.optimizer_d = utils.get_optimizer(args, 'heads', self.decoders.parameters())

        self.loss = utils.get_loss(args)

        #maybe think about setting 

    def load_generator(self):
        pass

    def load_decoders(self):
        pass

    def train(self):

        # maybe make ConditionalNoiseGen class(see GEBM)

        z = self.latent_noise_gen.sample([self.batch_size_train])
        c = self.class_dist.sample([self.batch_size_train])
        output = self.generator(z,c)
        output = self.decoders(output)
        print('yes')
        for epoch in range(self.max_train_epoch):
            self.train_epoch()

    def train_epoch(self):

        for batch_idx, data in enumerate(self.train_loader):
            check = data

    

    def eval(self):
        pass

    def batch_step(self,data,net_type,train_mode):

        # modified from https://github.com/MichaelArbel/GeneralizedEBM/blob/b2fb244bacef23a7347aecc0e8ff4863153f94f0/trainer.py#L165

        optimizer = self.prepare_optimizer(net_type)

        z = self.latent_noise_gen.sample([self.batch_size_train])
        c = self.class_dist.sample([self.batch_size_train])

        with_gen_grad = train_mode and (net_type=='generator')
        with torch.set_grad_enabled(with_gen_grad):
            fake_data = self.generator(z,c)

        with torch.set_grad_enabled(train_mode):
            true_results = self.decoders(data)
            fake_results = self.decoders(fake_data)

        loss = self.loss(true_results, fake_results, net_type)

        if train_mode:
            total_loss = self.add_penalty(loss, net_type, data, fake_data)
            total_loss.backward()
            optimizer.step()
        return loss


    def prepare_optimizer(self,net_type):

        # from https://github.com/MichaelArbel/GeneralizedEBM/blob/b2fb244bacef23a7347aecc0e8ff4863153f94f0/trainer.py#L165

        if net_type=='decoders':           
            optimizer = self.optimizer_d
            self.decoders.train()
            self.generator.eval()
        elif net_type=='generator':
            optimizer = self.optimizer_g
            self.generator.train()
            self.decoders.eval()  
        optimizer.zero_grad()
        return optimizer