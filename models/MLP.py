import torch.nn as nn
import torch.nn.functional as F

class MLPZero(nn.Module):
    def __init__(self):
        super(MLPZero, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        return F.relu(self.fc(x)), None

class MLPOne(nn.Module):
    def __init__(self):
        super(MLPOne, self).__init__()
        self.fc1 = nn.Linear(784, 392)
        self.fc2 = nn.Linear(392, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x, None

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
        return x, None
