import numpy as np
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_mnist():
    trans = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=0.5, std=0.5)
    ])
    train = datasets.MNIST(        
        root="data", 
        train=True, 
        download=True, 
        transform=trans,
    )
    val = datasets.MNIST(        
        root="data",
        train=False, 
        download=True, 
        transform=trans,
    )
    return train, val


class LeNet(nn.Module):
  def __init__(self):
    super().__init__() 
    self.conv1 = nn.Conv2d(1, 6, 5) 
    self.pool1 = nn.MaxPool2d(2, 2) 
    self.conv2 = nn.Conv2d(6, 16, 5) 
    self.pool2 = nn.MaxPool2d(2, 2) 
    self.fc1 = nn.Linear(16 * 4 * 4,  120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):  # (1, 28, 28)
    x = self.pool1(F.relu(self.conv1(x)))  # (6, 12, 12)
    x = self.pool2(F.relu(self.conv2(x)))  # (16, 4, 4)
    x = torch.flatten(x, 1)  # what shape is x before flattening?
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x)) 
    logits = self.fc3(x) 
    return logits


def train(loader, model, loss_fn, opt):
  size = len(loader.dataset)
  for batch, (x, y) in enumerate(loader):
    x = x.double()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if batch % 100 == 0:
      loss, current = loss.item(), batch * len(x) 
      print(f"loss is {loss}  [{current} / {size}]")


def test(loader, model, loss_fn):
  size = len(loader.dataset)
  num_batches = len(loader)
  correct, test_loss = 0, 0
  with torch.no_grad():
    for x, y in loader:
      x = x.double()
      y_pred = model(x)
      test_loss += loss_fn(y_pred, y).item()
      correct += (y_pred.argmax(1) == y).type(torch.float).sum().item()
  test_loss /= num_batches
  correct /= size
  print(f"accuracy is {correct * 100} test_loss is {test_loss}")


def save(state_dict, path):
    torch.save(state_dict, "weights.pt")
    os.makedirs(path, exist_ok=True)
    for key, val in state_dict.items():
        np.save(os.path.join(path, f"{key}.npy"), val.numpy())


def main():
    train_data, val_data = load_mnist()
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

    model = LeNet().double()
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 10
    for epochs in range(EPOCHS):
      print(f"epoch: {epochs + 1} ---------------------------")
      train(train_loader, model, loss_fn, opt)
      test(val_loader, model, loss_fn)

    save(model.state_dict(), "weights")
    print("DONE")


if __name__ == "__main__":
    main()
