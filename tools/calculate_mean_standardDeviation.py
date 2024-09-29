import torch.utils.data
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset


class GetMeanStd:
    def __init__(self, trainset):
        self.trainset = trainset
        self.x = torch.stack([sample[0]
                             for sample in ConcatDataset([trainset])])
        self.mean = torch.mean(self.x, dim=(0, 2, 3))
        self.std = torch.std(self.x, dim=(0, 2, 3))

    def get_mean_std(self):

        mean = torch.mean(self.x, dim=(0, 2, 3))
        std = torch.std(self.x, dim=(0, 2, 3))
        return self.mean, self.std
