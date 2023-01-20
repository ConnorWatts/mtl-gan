from torch import nn
import torch

class Head(nn.Module):
     def __init__(self,num_classes) -> None:
          super().__init__()
          self.conv1 = nn.Conv2d(16,8,4,2,1)
          self.conv2 = nn.Conv2d(8,num_classes,4,2,1)

     def forward(self,x):

          return torch.sigmoid(self.conv2(self.conv1(x)))
