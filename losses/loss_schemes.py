from torch import nn
import torch

# from https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch/blob/ed3d9ac1c35493d8d22c4e33574919c326d77b6d/losses/loss_schemes.py#L23
# Authors: Simon Vandenhende

class MultiTaskLoss(nn.Module):
    def __init__(self, tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict)-> None:
        super(MultiTaskLoss, self).__init__()
        assert(set(tasks) == set(loss_ft.keys()))
        assert(set(tasks) == set(loss_weights.keys()))
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights

    
    def forward(self, pred, gt):
        out = {task: self.loss_ft[task](pred[task], gt[task]) for task in self.tasks}
        out['total'] = torch.sum(torch.stack([self.loss_weights[t] * out[t] for t in self.tasks]))
        return out