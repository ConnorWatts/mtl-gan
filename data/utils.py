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


def get_data_loader(args):

    transform = get_data_transform()
    trainset, testset = get_dataset(args,transform)

    b_size = args['batch_size_train']
    num_workers = args['num_workers']

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=b_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=b_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader, None
