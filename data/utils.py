from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
import torch

def get_data_loader(params,num_workers):

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    b_size = params.batch_size

    if params.dataset == 'Cifar100':

        trainset = CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = CIFAR100(root='./data', train=False, download=True, transform=transform_train)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=b_size, shuffle=True, num_workers=num_workers)
        testloader = torch.utils.data.DataLoader(testset, batch_size=b_size, shuffle=False, num_workers=num_workers)

        return trainloader, testloader, None
