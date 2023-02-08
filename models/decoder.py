import torch
import torch.nn as nn

class MultiTaskDecoder(nn.Module):
    """ Multi-task decoder with shared network + task-specific heads """
    def __init__(self, decoder: nn.Module, heads: nn.ModuleDict, tasks: list):
        super(MultiTaskDecoder, self).__init__()
        assert(set(heads.keys()) == set(tasks))
        self.decoder = decoder
        self.heads = heads
        self.tasks = tasks

    def forward(self, x):
        shared_representation = self.decoder(x)
        return {task: self.heads[task](shared_representation) for task in self.tasks}