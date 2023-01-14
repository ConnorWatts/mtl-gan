from torch import nn
import torch

class TestGenerator(nn.Module):
        def __init__(self,z_dim,num_classes) -> None:
             super().__init__()

             # maybe be nicer this way
             # self.net = nn.Sequential(
             # *vanilla_block(num_neurons_per_layer[0], num_neurons_per_layer[1]),
             # vanilla_block(num_neurons_per_layer[1], num_neurons_per_layer[2]),

             self.z_dim = z_dim
             self.num_classes = num_classes
             self.label_embedding = nn.Embedding(self.num_classes, self.num_classes)

             latent_conv = nn.ConvTranspose2d(self.z_dim, 512, 4, stride=1, padding=0)
             label_conv = nn.ConvTranspose2d(self.num_classes, 512, 4, stride=1, padding=0)
             #should this be 521 > 256
             joint_conv1 = nn.ConvTranspose2d(1024, 512, 4,  stride=2, padding=(1,1))
             joint_conv2 = nn.ConvTranspose2d(512, 256, 4,  stride=2, padding=(1,1))
             joint_conv3 = nn.ConvTranspose2d(256, 3, 4,  stride=2, padding=(1,1))
             #joint_conv4 = nn.ConvTranspose2d(128, 64, 4,  stride=2, padding=(1,1))
             #joint_conv5 = nn.ConvTranspose2d(64, 3, 4,  stride=2, padding=(1,1))
        

             self.latent_layer = nn.Sequential(
                latent_conv,
                nn.BatchNorm2d(512),
                nn.LeakyReLU())

             self.label_layer = nn.Sequential(
                label_conv,
                nn.BatchNorm2d(512),
                nn.LeakyReLU())

             self.joint_layers = nn.Sequential(
                joint_conv1,
                nn.BatchNorm2d(512),
                nn.LeakyReLU(),
                joint_conv2,
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
                joint_conv3,
                #nn.BatchNorm2d(128),
                #nn.LeakyReLU(),
                #joint_conv4,
                #nn.BatchNorm2d(64),
                #.LeakyReLU(),
                #joint_conv5,
                nn.Tanh())

        def forward(self,z,c):
                h1 = self.latent_layer(z.view(-1, self.z_dim, 1, 1))
                h2 = self.label_layer(self.label_embedding(c).view(-1, self.num_classes,1,1))
                j1 = self.joint_layers(torch.cat([h1,h2],1))
                return j1
  





class DCGANSNGenerator(nn.Module):

    def __init__(self, z_dim):
        super(DCGANSNGenerator, self).__init__()
        channels = 3
        input_dim = 4
        label_dim = 4
        conv_1 = nn.ConvTranspose2d(label_dim, 512, 4, stride=1, padding=0)
        conv_2 =  nn.ConvTranspose2d(input_dim, 512, 4, stride=1, padding=0)
        conv_3 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=(1,1))
        conv_4 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=(1,1))
        conv_5 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=(1,1))
        conv_6 = nn.ConvTranspose2d(64, channels, 3, stride=1, padding=(1,1))

        nn.init.xavier_uniform_(conv_1.weight.data, 1.)
        nn.init.xavier_uniform_(conv_2.weight.data, 1.)
        nn.init.xavier_uniform_(conv_3.weight.data, 1.)
        nn.init.xavier_uniform_(conv_4.weight.data, 1.)
        nn.init.xavier_uniform_(conv_5.weight.data, 1.)
        nn.init.xavier_uniform_(conv_6.weight.data, 1.)

        self.input_layers = nn.Sequential(
                conv_1,
                nn.BatchNorm2d(512),
                nn.ReLU())

        self.label_layers = nn.Sequential(
                conv_2,
                nn.BatchNorm2d(512),
                nn.ReLU())

        self.joint_layers = nn.Sequential(
                conv_3,
                nn.BatchNorm2d(256),
                nn.ReLU(),
                conv_4,
                nn.BatchNorm2d(128),
                nn.ReLU(),
                conv_5,
                nn.BatchNorm2d(64),
                nn.ReLU(),
                conv_6,
                nn.Tanh())


    def forward(self, z,c):
        h1 = self.inputs_layers(z)
        h2 = self.label_layers(c)
        x = torch.cat([h1, h2], 1)
        out = self.joint_layers(x)
        return out