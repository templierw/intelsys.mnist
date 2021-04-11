import torch.nn as nn
import torch.nn.functional as F

# The models return the same value twice to match the return type of the CNN models

class MLPZero(nn.Module):
    def __init__(self):
        super(MLPZero, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc(x)
        return x, F.softmax(x, dim=1)

class MLPZeroReLu(nn.Module):
    def __init__(self):
        super(MLPZeroReLu, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc(x))
        return x, F.softmax(x, dim=1)

class MLPOne(nn.Module):
    def __init__(self):
        super(MLPOne, self).__init__()
        self.fc1 = nn.Linear(784, 392)
        self.fc2 = nn.Linear(392, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x, F.softmax(x, dim=1)

class MLPTwo(nn.Module):
    def __init__(self):
        super(MLPTwo, self).__init__()
        self.fc1 = nn.Linear(784, 392)
        self.fc2 = nn.Linear(392, 196)
        self.fc3 = nn.Linear(196, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x, F.softmax(x, dim=1)
