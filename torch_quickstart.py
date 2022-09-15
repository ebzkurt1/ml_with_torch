import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

'''
The torchvision.datasets module contains Dataset objects for 
many real-world vision data like CIFAR, COCO
'''

# Download training data from open datasets
training_data = datasets.FashionMNIST(
        root='data',
        train=True,
        download=True,
        transform=ToTensor(),
)

# Download test data from open datasets
test_data = datasets.FashionMNIST(
        root='data',
        train=False,
        download=True,
        transform=ToTensor(),
)

'''
We pass the Dataset as an argument to DataLoader. This wraps
an iterable over our dataset, and supports automatic batching,
sampling, shuffling and multiprocess data loading.
'''

batch_size = 64

# Create data loaders
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:  # Iterate through the dataloader
    print(f'Shape of X [N, C, H, W]: {X.shape}')
    print(f'Shape of y: {y.shape} {y.dtype}')
    break  # Break from the loop since we only showcase this functionality


