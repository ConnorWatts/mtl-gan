from torch import nn
import torch

class MultiTaskLoss(nn.Module):
    def __init__(self, tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict)-> None:
        super(MultiTaskLoss, self).__init__()
        assert(set(tasks) == set(loss_ft.keys()))
        assert(set(tasks) == set(loss_weights.keys()))
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights

    
    def forward(self, true_data, false_data, true_target, false_target,network_type):
        out = {task: self.loss_ft[task](true_data[task], false_data[task],true_target[task],false_target[task],network_type) for task in self.tasks}
        out['total'] = torch.sum(torch.stack([self.loss_weights[task] * out[task] for task in self.tasks]))
        return out