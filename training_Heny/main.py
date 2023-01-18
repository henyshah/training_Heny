from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from model import *
import torch.nn.functional as F
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def main():
    seed = 12345
    use_cuda = True
    batch_size = 64
    epochs = 14

    torch.manual_seed(seed)
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
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=0.001)
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
    torch.save(model.state_dict(), "mnist_cnn.pt")
if __name__ == '__main__':
    main()

