from torch import nn
import torch

class Head(nn.Module):
        def __init__(self,num_classes) -> None:
             super().__init__()
             self.conv1 = nn.Conv2d(16,num_classes,4,2,1)

        def forward(self,x):
            return torch.sigmoid(self.conv1(x))
