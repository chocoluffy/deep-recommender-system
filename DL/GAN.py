import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn import Module, Sequential, Linear, ReLU, Tanh, Dropout, Sigmoid
from torch import Tensor, randn
import time
import torch.autograd as autograd
from tqdm import tqdm
import torchvision
from torchvision import transforms

# apply transformation on dataset. here turn PIL image format into pytorch tensor. 
transform = transforms.Compose([
    transforms.ToTensor()
])

batch_size = 128
fmnist = torchvision.datasets.FashionMNIST(root="./", train=True, transform=transform, download=True)
data_loader = torch.utils.data.DataLoader(dataset=fmnist, batch_size=batch_size, shuffle=True)

"""
Generator:
- last output layer uses tanh, as its value range is (-inf, inf), cannot use sigmoid or so.
- input are randomly generated noise.
"""
class Generator(Module):
    def __init__(self):
        super(Generator, self).__init__()
        d_input = 100
        d_output = 28*28
        
        self.input = Sequential(
            Linear(d_input, 256),
            ReLU()
        )
        self.hidden1 = Sequential(            
            Linear(256, 512),
            ReLU()
        )
        self.hidden2 = Sequential(
            Linear(512, 1024),
            ReLU()
        )
        self.output = Sequential(
            Linear(1024, d_output),
            Tanh()
        )
        
    def forward(self, x):
        x = self.input(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.output(x)
        return x

"""
Discriminator:
- input dimensions: feature dimensions. 
- network input: a batch of input data. dim: batch_size * featuer_size. network output result: batch_size * output_size
- pick activation functions depending on the task. for example, binary classification uses Sigmoid(), while multiple classess classification uses Softmax() and so on.
"""
class Discriminator(Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        d_input = 28*28
        d_output = 1
        
        self.input = Sequential( 
            Linear(d_input, 256),
            ReLU()
        )
        self.hidden1 = Sequential(
            Linear(256, 128),
            ReLU()
        )
        self.hidden2 = Sequential(
            Linear(128, 64),
            ReLU()
        )
        self.output = Sequential(
            Linear(64, d_output),
            Sigmoid()
        )
    def forward(self, x):
        x = self.input(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.output(x)
        return x
    
lr = 1e-4
epochs = 20
errors_G = []
errors_D = []

generator = Generator()
discriminator = Discriminator()
g_optim = optim.Adam(generator.parameters(), lr=lr)
d_optim = optim.Adam(discriminator.parameters(), lr=lr)

loss = nn.BCELoss() # use binary cross entropy loss.

for epoch in range(1, epochs):
    for n_batch, batch in enumerate(data_loader):
        # Train Discriminator
        batch = batch[0] # batch = [batch_img, label] batch_img: batch_size * 1 * 28 * 28
        trueData = autograd.Variable(batch)
        falseData = generator(autograd.Variable(randn(batch.size(0), 100))).detach() # detach to not calculate gradients
        
        trueN = trueData.size(0)
        falseN = falseData.size(0)
        d_optim.zero_grad()

        # Train on true data
        trueData = trueData.view(trueData.shape[0], 28*28) # reshape tensor to accord with network input.
        truePred = discriminator(trueData)
        trueError = loss(truePred, autograd.Variable(torch.ones(trueN, 1)))
        trueError.backward()

        # Train on generated data
        falsePred = discriminator(falseData)
        falseError = loss(falsePred, autograd.Variable(torch.zeros(falseN, 1)))
        falseError.backward()

        d_optim.step()
        error_d = trueError + falseError

        # Train Generator
        falseData = generator(autograd.Variable(randn(batch.size(0), 100)))
        falseN = falseData.size(0)
        g_optim.zero_grad()

        # Get response from Discriminator
        pred = discriminator(falseData) 

        # Pretend that the false data is true data
        error = loss(pred, autograd.Variable(torch.ones(falseN, 1)))
        error.backward()

        g_optim.step()
        error_g = error
    
    print('Epoch', epoch)
    print('Generator Error:', error_g.item())
    print('Discriminator Error:', error_d.item())
    print()
    
    errors_G.append(error_g.item())
    errors_D.append(error_d.item())
    
    if epoch % 5 == 0:
        # save model every 5 epochs.
        torch.save(generator.state_dict(), './models/epoch-{}-generator.pt'.format(epoch))
        torch.save(discriminator.state_dict(), './models/epoch-{}-discriminator.pt'.format(epoch))


# Plot errors and save figs
figG, axG = plt.subplots()
axG.plot(range(1, epochs), errors_G)
axG.set_xlabel('epoch')
axG.set_ylabel('error')
axG.set_title('Generator')
figG.savefig('errorG.png')


figD, axD = plt.subplots()
axD.plot(range(1, epochs), errors_D)
axD.set_xlabel('epoch')
axD.set_ylabel('error')
axD.set_title('Discriminator')
figD.savefig('errorD.png')