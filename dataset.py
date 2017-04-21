import torch.utils.data as data
from torchvision import datasets
import torch
from PIL import Image


class SiameseDataset(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, download=False):
        super(SiameseDataset, self).__init__(root, train=train, transform=transform, target_transform=None, download=download)
        self.train = train

    def __getitem__(self, index):
        random_index = torch.LongTensor(2).random_(0, len(self) - 1)
        if self.train:
            img0, target0 = self.train_data[random_index[0]], self.train_labels[random_index[0]]
            img1, target1 = self.train_data[random_index[1]], self.train_labels[random_index[1]]
        else:
            img0, target0 = self.test_data[random_index[0]], self.test_labels[random_index[0]]
            img1, target1 = self.test_data[random_index[1]], self.test_labels[random_index[1]]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img0 = Image.fromarray(img0.numpy(), mode='L')
        img1 = Image.fromarray(img1.numpy(), mode='L')

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        target = torch.FloatTensor([1.0]) if target0 == target1 else torch.FloatTensor([-1.0])

        return img0, img1, target

    def __len__(self):
        if self.train:
            return 60000
        else:
            return 10000

