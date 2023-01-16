from torch import nn
import torch

class DCGANSNDiscriminator(nn.Module):
        def __init__(self) -> None:
            super().__init__()

            conv1 = nn.Conv2d(3,64,4,2,1)
            conv2 = nn.Conv2d(64,32,4,2,1)
            conv3 = nn.Conv2d(32,16,4,2,1)

            self.layers = nn.Sequential(
                conv1,
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                conv2,
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                conv3,
                nn.BatchNorm2d(16),
                nn.LeakyReLU(),

            )

        def forward(self,x):
            out = self.layers(x)
            return out