import torch.utils
import torch.utils.data
from tools import calculate_mean_standardDeviation
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tools.print_images_and_lables_4 import PrintImagesAndLables
from models.LeNet import LeNet
from tools.get_device_type import device


class Demo1:
    def __init__(self, num_epochs=5):
        self.num_epochs = num_epochs
        pass

    def run(self):
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        mean, std = calculate_mean_standardDeviation.GetMeanStd(
            trainset).get_mean_std()
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean, std)
                                        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                transform=transform)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=64, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               transform=transform)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=64, shuffle=False, num_workers=2)
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        # PrintImagesAndLables(trainloader, classes).show_images()
        model = LeNet().to(device=device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        model.train()
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0
        print('Finished Training')

        correct = 0
        total = 0
        with torch.no_grad():
            model.eval()
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
        torch.save(model.state_dict(), "./saveModels/leNetModel.pth")
