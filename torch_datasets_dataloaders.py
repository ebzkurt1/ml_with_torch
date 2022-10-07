import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
        root='data',  # where the train/test data is stored
        train=True,  # specifies training or test dataset
        download=True,  # download the data from the internet if not available in the root
        transform=ToTensor()  # specify the feature and label transformations
)

test_data = datasets.FashionMNIST(
        root='data',
        train=False,
        download=True,
        transform=ToTensor()
)


# Iterating and Visualizing the Dataset

labels_map = {
        0: 'T-Shirt',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle Boot',
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis('off')
    plt.imshow(img.squeeze(), cmap='gray')
plt.show()


'''
TRANSFORMS
All TorchVision datasets have two parameters -transform to modify the features and target_transform to modify
the labels - that accept callables containing the transformation logic. The torchvision.transforms module
offers several commonly-used transforms out of the box.
'''

# Loading dataset with custom transform
ds = datasets.FashionMNIST(
        root='data',
        train=True,
        dowload=True,
        transform=ToTensor(),
        target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch,float).scatter_(0, torch.tensor(y), value=1))
)


