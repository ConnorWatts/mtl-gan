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

        
        self.enable_tb = args['enable_tensorboard']
        if self.enable_tb:
            self.writer = SummaryWriter()

        utils.set_random_seeds(args['seed'])

        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        #self.device = utils.assign_device(args['device'])

        self.train_loader, self.test_loader, self.valid_loader = get_data_loader(args)
        self.train_loader_size = len(self.train_loader)

        self.generator = get_generator(args)
        self.decoders = get_decoders(args)

        self.latent_noise_gen = utils.get_latent_noise(args,self.device)
        self.class_dist = utils.get_class_dist(args,self.device)

        self.optimizer_g = utils.get_optimizer(args,'generator', self.generator.parameters())
        self.optimizer_d = utils.get_optimizer(args, 'heads', self.decoders.parameters())

        self.loss = utils.get_loss(args)

        self.gen_loss = []
        self.dec_loss = []

        self.epoch = 0


        #maybe think about setting 

    def load_generator(self):
        pass

    def load_decoders(self):
        pass

    def train(self):

        # maybe make ConditionalNoiseGen class(see GEBM)
        while self.epoch < self.max_train_epoch:
            self.train_epoch()

    def train_epoch(self):

        for batch_idx, data in enumerate(self.train_loader):

            decoders_loss = self.batch_step(data,'decoders',train_mode=True)
            # might have to specifity which loss
            self.dec_loss.append(decoders_loss)

            generator_loss = self.batch_step(data,'generator',train_mode=True)
            self.gen_loss.append(generator_loss)

            if self.enable_tb:
                self.tb_log(decoders_loss,generator_loss,batch_idx,self.train_loader_size)

        self.epoch += 1

    

    def eval(self):
        pass

    def batch_step(self,data,net_type,train_mode):

        # modified from https://github.com/MichaelArbel/GeneralizedEBM/blob/b2fb244bacef23a7347aecc0e8ff4863153f94f0/trainer.py#L165

        optimizer = self.prepare_optimizer(net_type)

        images = data['images']#.cuda(non_blocking=True)

        z = self.latent_noise_gen.sample([self.batch_size_train])
        c = self.class_dist.sample([self.batch_size_train])

        real_targets = self.get_real_targets(data)
        fake_targets = self.get_fake_targets(c)

        with_gen_grad = train_mode and (net_type=='generator')
        with torch.set_grad_enabled(with_gen_grad):
            fake_data = self.generator(z,c)

        with torch.set_grad_enabled(train_mode):
            real_results = self.decoders(images)
            fake_results = self.decoders(fake_data)

        if net_type == 'generator':
            loss = self.loss(real_results, fake_results,real_targets,fake_targets,'generator')
        elif net_type == 'decoders':
            loss = self.loss(real_results, fake_results,real_targets,fake_targets,'decoders')

        if train_mode:
            #total_loss = self.add_penalty(loss, net_type, data, fake_data)
            loss['total'].backward()
            optimizer.step()

        return loss

    def get_real_targets(self,data):

        # v1 - write better

        out = {'gan': torch.ones(len(data['images']))}

        for task in self.args['tasks']:
            if task == 'gan':
                continue
            else:
                out[task] = data[task]
        return out

    def get_fake_targets(self,data):

        # v1 - write better

        out = {'gan': torch.zeros(len(data))}

        for task in self.args['tasks']:
            if task == 'gan':
                continue
            else:
                if task == 'fine':
                    out[task] = data
                else:
                    out[task] = torch.Tensor(utils.get_coarse_label(data))
        return out



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

    def tb_log(self, decoders_loss: dict, generator_loss: dict, batch_idx: int, train_loader_size: int):
        self.writer.add_scalars('gan loss', {'g': generator_loss['gan'].item(), 'd': decoders_loss['gan'].item()}, train_loader_size* self.epoch + batch_idx + 1)
