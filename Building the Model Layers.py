'''
Weights, Bias, Activation function 

Building a neural network to classify images in the FashionMNIST dataset. 

'''
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

training_data = datasets.FashionMNIST(
    root = "data", 
    train= True,
    download = True, 
    transform = ToTensor()
)

test_data = datasets.FashionMNIST(
    root= "data", 
    train = False, 
    download=True, 
    transform= ToTensor()
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)

train_loader = DataLoader(training_data, batch_size = 64, shuffle = True, generator= torch.Generator(device))

'''
Define the class




'''