from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
import torch


def get_data_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform


def get_dataset(args,transform):

    if args['dataset'] == 'Cifar100':

        trainset = CIFAR100(root='./data', train=True, download=True, transform=transform)
        testset = CIFAR100(root='./data', train=False, download=True, transform=transform)

    return trainset, testset

from keras.utils import to_categorical
import numpy as np

def get_coarse_label(data):
    # from https://github.com/ryanchankh/cifar100coarse/blob/master/sparse2coarse.py
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.
    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[data]

def get_data_loader(args):

    transform = get_data_transform()
    trainset, testset = get_dataset(args,transform)

    b_size = args['batch_size_train']
    num_workers = args['num_workers']

    def collate_batch(batch):
        
        images = torch.stack([sample[0] for sample in batch])
        batch_output = {'images' : images}
        if 'fine' in args['tasks']:
            #batch_output['fine'] = [torch.Tensor(to_categorical(sample[1],num_classes=100)) for sample in batch]
            batch_output['fine'] = torch.Tensor([sample[1] for sample in batch])
        if 'coarse' in args['tasks']:
            #batch_output['coarse'] = [torch.Tensor(to_categorical(get_coarse_label(sample[1]),num_classes=20)) for sample in batch]
            batch_output['coarse'] = torch.Tensor([get_coarse_label(sample[1]) for sample in batch])
        return batch_output

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=b_size, shuffle=True, num_workers=num_workers,collate_fn=collate_batch)
    testloader = torch.utils.data.DataLoader(testset, batch_size=b_size, shuffle=False, num_workers=num_workers, collate_fn=collate_batch)

    return trainloader, testloader, None
