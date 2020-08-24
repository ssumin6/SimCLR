import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from gaussian_blur import GaussianBlur
from torchvision import datasets

class DataSetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, input_shape, strength):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.input_shape = input_shape
        self.strength = strength

    def get_data_loaders(self):
        data_augment = self._simclr_transform()

        train_dataset = datasets.STL10('./data', split='train+unlabeled', download=True,
                                       transform=SimCLRDataTransform(data_augment))

        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def _simclr_transform(self):
        # I strongly recommand you to use torchvision.transforms to implement data augmentation
        # You can use provided gaussian_blur if you want

        data_transforms: transforms.Compose

        s = self.strength # color distortion strength
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        gaussian_blur = GaussianBlur(kernel_size = int(0.1* self.input_shape[0]))

        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=self.input_shape[0]), # Random cropping & resize to original image size
            transforms.RandomHorizontalFlip(), 
            transforms.RandomApply([color_jitter], p=0.8), # Random color distortions
            transforms.RandomGrayscale(p=0.2),
            gaussian_blur,
            transforms.ToTensor()
        ])

        return data_transforms

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)
        return train_loader, valid_loader


class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj
