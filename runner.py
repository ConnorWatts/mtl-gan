import torch
import utils
from models.utils import get_generator, get_decoders
from data.utils import get_data_loader
from torch.utils.tensorboard import SummaryWriter


class ModelRunner:

    def __init__(self, config: dict) -> None:
        super().__init__()

        """
        Constructs an Agent to run the MTL GAN Experiment
        
        :param config: Dict denoting the command line inputs to be 
            used as the configuration of the experiment 

        """

        self.config = config
        self.max_train_epoch = config['max_train_epochs']
        self.batch_size_train = config['batch_size_train']

        self.enable_tb = config['enable_tensorboard']
        if self.enable_tb:
            self.writer = SummaryWriter()

        utils.set_random_seeds(config['seed'])

        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        #self.device = utils.assign_device(args['device'])

        self.train_loader, self.test_loader, self.valid_loader = get_data_loader(config)
        self.train_loader_size = len(self.train_loader)

        self.generator = get_generator(config)
        self.decoders = get_decoders(config)

        self.latent_noise_gen = utils.get_latent_noise(config,self.device)
        self.class_dist = utils.get_class_dist(config,self.device)

        self.optimizer_g = utils.get_optimizer(config,'generator', self.generator.parameters())
        self.optimizer_d = utils.get_optimizer(config, 'heads', self.decoders.parameters())

        self.loss = utils.get_loss(config)

        self.gen_loss = []
        self.dec_loss = []

        self.epoch = 0


    def load_generator(self, config: dict) -> None:
        """
        Loads a generator state into self.generator .
        :param : .

        """
        ...

    def load_decoders(self, config: dict) -> None:
        """
        Loads a generator state into self.generator .
        :param : .

        """
        ...


    def train(self) -> None:
        """
        Trains the Agent for a set number of epochs

        :return: None
        """

        while self.epoch < self.max_train_epoch:
            self.train_epoch()


    def train_epoch(self) -> None:
        """
        Trains the Agent for an individual epoch

        For each batch of the dataset first the decoder is trained
        and then the generator. 

        :return: None
        """

        for batch_idx, data in enumerate(self.train_loader):

            decoders_loss = self.batch_step(data,'decoders',train_mode=True)
            self.dec_loss.append(decoders_loss)
            generator_loss = self.batch_step(data,'generator',train_mode=True)
            self.gen_loss.append(generator_loss)

            if self.enable_tb:
                self.tb_log(decoders_loss,generator_loss,batch_idx,self.train_loader_size)

        self.epoch += 1

    

    def batch_step(self,data,net_type,train_mode) -> None:
        """
        Trains the Agent for a single batch

        :return: None
        """

        # modified from https://github.com/MichaelArbel/GeneralizedEBM/
        # blob/b2fb244bacef23a7347aecc0e8ff4863153f94f0/trainer.py#L165

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


    def eval(self) -> None:
        """
        Evaluate the performance of the Generator

        :return: None
        """
        ...

    def get_real_targets(self,data) -> dict:
        """
        Generate the targets for the decoders given the data is real

        :param data: torch.Tensor denoting the data from the batch. This
            contains the images (for the gan task) and the targets for the 
            individual tasks

        :return out: dict of targets for (example {"gan":[1,1,1..],"coarse": \
            [12,4,11,8,2...],"fine": [34,89,45,23..]})
        """

        # TO DO: improve (maybe merch with real)
        out = {'gan': torch.ones(len(data['images']))}

        for task in self.args['tasks']:
            if task == 'gan':
                continue
            else:
                out[task] = data[task]
        return out

    def get_fake_targets(self,data) -> dict:
        """
        Generate the targets for the decoders given the data is fake

        :param data: torch.Tensor denoting the data from the batch. This
            contains the images (for the gan task) and the targets for the 
            individual tasks

        :return out: dict of targets for (example {"gan":[0,0,0..],"coarse": \
            [12,4,11,8,2...],"fine": [34,89,45,23..]})
        """

        # TO DO: improve (maybe merch with real)
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
        """
        Prepare the optimizers for training

        :param net_type: str denoting which of the networks is being trained
            from this we then set it to .train() and the other to .eval()

        :return None:
        """
        # from https://github.com/MichaelArbel/GeneralizedEBM/blob/b2fb244bacef
        # 23a7347aecc0e8ff4863153f94f0/trainer.py#L165

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

    def tb_log(self, decoders_loss: dict, generator_loss: dict, \
        batch_idx: int, train_loader_size: int) -> None:
        """
        Write to Tensorbaord

        :param decoders_loss: dict denoting the losses of each task for the decoders
        :param generator_loss: dict denoting the losses of each task for the generator
        :param batch_idx: int denoting the current batch index for indexing to viz
        :param train_loader_size: int denoting the size of the training set

        :return None:
        """

        self.writer.add_scalars('gan loss', {'g': generator_loss['gan'].item(),\
             'd': decoders_loss['gan'].item()}, train_loader_size* self.epoch + batch_idx + 1)
