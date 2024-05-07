import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np


train_transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR10('data/cifar', download=True, transform=train_transform)


targets = np.array(train_dataset.targets)

all_train_indices = []
all_test_indices = []

for i in range(101):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=i)
    for train_index, test_index in sss.split(targets, targets):
        all_train_indices.append(train_index)
        all_test_indices.append(test_index)

for i in range(101):
    np.save(f'./indices/train_idx_{i}.npy', all_train_indices[i])
    np.save(f'./indices/test_idx_{i}.npy', all_test_indices[i])


