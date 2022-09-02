"""
Train a LeNet Model (two convolutional layers and three fully-connected layers) on Fashion-MNIST to obtain
the best accuracy we can.
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from layers.models import SparseConv2d

device = "cuda" if torch.cuda.is_available() else "cpu"


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

testing_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

# Define some hyper-parameters
batch_size = 64
epochs = 20

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(testing_data, batch_size=batch_size)


class SparseLeNet(nn.Module):
    def __init__(self):
        super(SparseLeNet, self).__init__()
        self.conv1 = SparseConv2d(28, 28, 1, 32, 5, connect_type="normal")
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = SparseConv2d(12, 12, 32, 16, 5, connect_type="normal")
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16*4*4, 32)
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.act4 = nn.ReLU()
        self.fc3 = nn.Linear(16, 10)

    def forward(self, x):
        out = self.pool1(self.act1(self.conv1(x)))
        out = self.pool2(self.act2(self.conv2(out)))
        out = out.view(-1, 16*4*4)
        out = self.act3(self.fc1(out))
        out = self.act4(self.fc2(out))
        out = self.fc3(out)
        return out


# model = SparseLeNet().to(device)
# loss_fn = nn.CrossEntropyLoss()


def train_loop(dataloader, network, loss_function, optim):
    size = len(dataloader.dataset)

    for batch, (img_batch, result_batch) in enumerate(dataloader):
        img_batch = img_batch.to(device)
        result_batch = result_batch.to(device)

        pred = network(img_batch)
        loss = loss_function(pred, result_batch)

        optim.zero_grad()
        loss.backward()
        optim.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(pred)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, network, loss_function):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for img_batch, result_batch in dataloader:
            img_batch = img_batch.to(device)
            result_batch = result_batch.to(device)
            pred = network(img_batch)
            test_loss += loss_function(pred, result_batch).item()

            correct += (pred.argmax(1) == result_batch).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    # print(datetime.datetime.now())
    # print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

"""
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")

    if t <= 25:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    elif t <= 40:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)

PATH = "sparseCNN.pth"
# torch.save(model.state_dict(), PATH)
"""
