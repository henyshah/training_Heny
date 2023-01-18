from __future__ import print_function
import argparse

import numpy as np
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from model import *
import torch.nn.functional as F

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
if __name__== '__main__':

    seed = 12345
    use_cuda = True
    batch_size = 64
    epochs = 14

    if use_cuda:
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size)

    first_batch, _ = next(iter(train_loader))
    for i in range(len(first_batch)):
        img = first_batch[i][0]
        np.save(f'./mnist/image_{i}', img)

    model = Net().to(device)
    checkpoint = torch.load ('mnist_cnn.pt')
    model.load_state_dict(checkpoint)
    test(model, device, test_loader)