import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Get device for training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

# Define the NN class
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

X = torch.rand(1, 28, 28, device=device)  # Generate random data
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f'Predicted class: {y_pred}')


# Mode Layers

input_image = torch.rand(3, 28, 28)  # Random data to represent image
print('Random image tensor size: ', input_image.size())

flatten = nn.Flatten()
flat_image = flatten(input_image)
print('Image after flatten layer: ',flat_image.size())

layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)  # Feed the flattened image into the linear layer
print('Size of the first hidden layer: ', hidden1.size())

print(f'Before ReLU: {hidden1}\n\n')
hidden1 = nn.ReLU()(hidden1)
print(f'After ReLU: {hidden1}')


