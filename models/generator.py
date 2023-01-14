from torch import nn
import torch

class TestGenerator(nn.Module):
        def __init__(self,z_dim) -> None:
             super().__init__()

             #consider using embedding??

             self.z_dim = z_dim

             conv_1 = nn.ConvTranspose2d(self.z_dim, 512, 4, stride=1, padding=0)

             self.input_layers = nn.Sequential(
                conv_1,
                nn.BatchNorm2d(512),
                nn.ReLU())

        def forward(self,z,c):
                return self.input_layers(z.view(-1, self.z_dim, 1, 1))
  





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