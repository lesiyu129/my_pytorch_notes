from tools import calculate_mean_standardDeviation
import torch
import torchvision
import torchvision.transforms as transforms


# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])


transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
mean, std = calculate_mean_standardDeviation.GetMeanStd(
    trainset).get_mean_std()
print(mean, std)
