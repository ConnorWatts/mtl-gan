from torch import nn
import torch

class DCGANSNDiscriminator(nn.Module):
        def __init__(self) -> None:
            super().__init__()

            conv1 = nn.Conv2d(3,64,4,2,1)

            self.layers = nn.Sequential(
                conv1,
                nn.BatchNorm2d(512),
                nn.LeakyReLU()
            )

        def forward(self,x):
            out = self.layers(x)
            return out