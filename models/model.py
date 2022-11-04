
import torch.nn as nn

class Model(nn.Module):
    """ Multi-task baseline model with shared encoder + task-specific decoders """
    def __init__(self, generator: nn.Module, heads: nn.ModuleDict, tasks: list):
        super(Model, self).__init__()
        assert(set(heads.keys()) == set(tasks))
        self.generator = generator
        self.heads = heads
        self.tasks = tasks
