import matplotlib.pyplot as plt
import numpy as np
import torchvision


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


class PrintImagesAndLables:
    def __init__(self, trainloader, classes):
        self.trainloader = trainloader
        self.classes = classes

    def show_images(self):
        dataiter = iter(self.trainloader)
        images, labels = next(dataiter)

        # show images
        imshow(torchvision.utils.make_grid(images))
        # print labels
        print(' '.join('%5s' % self.classes[labels[j]] for j in range(4)))

        plt.show()
