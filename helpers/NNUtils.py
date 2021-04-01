import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import datasets as dt, transforms

from helpers.analysis.AnalysisUtils import getNumCorrect

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def loadMNISTDatasets(path='./dataset'):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))] # value needs to be recomputed, taken from internet
    )
    dataset = dt.MNIST(root=path, train=True, download=True, transform=transform)
    
    test_split = int(len(dataset) * 0.2)
    train_split = len(dataset) - test_split
    train_dataset, test_dataset = random_split(dataset, [train_split, test_split])

    val_dataset = dt.MNIST(root=path, train=False, download=True, transform=transform)

    return train_dataset, test_dataset, val_dataset


def getMNISTLoaders(datasets,batch_size=4, num_workers=2):

    train_dataset, test_dataset, val_dataset = datasets

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, val_loader


def getSGDOptim(model, lr, momentum=0):
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)


def create_train_step_function(model, loss_fn, optimizer):
    def train_step(x, y):
        model.train()
        yhat = model(x)

        loss = loss_fn(yhat, y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return train_step


def create_validation_step_function(model, loss_fn):
    def validation_step(x, y):
        model.eval()
        yhat = model(x)
        val_loss = loss_fn(yhat, y)
        return val_loss.item()
    return validation_step


# Based on article: https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e
def fit(epochs, model, loss_fn, optimizer, train_loader, test_loader):
    losses = []
    val_losses = []

    train_step = create_train_step_function(model, loss_fn, optimizer)
    validation_step = create_validation_step_function(model, loss_fn)

    for epoch in range(epochs):
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            loss = train_step(x_batch, y_batch)
            losses.append(loss)

        with torch.no_grad():
            for x_val, y_val in test_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                val_loss = validation_step(x_val, y_val)
                val_losses.append(val_loss)

        print(f'Train Epoch: {epoch} \tLoss: {losses[-1]:.6f} \tTest Loss: {val_losses[-1]:.6f}')


def test(model, loss_fn, val_loader):
    sum_loss = 0
    correct = 0

    with torch.no_grad():
        model.eval()
        for x_test, y_test in val_loader:
            x_test = x_test.to(device)
            y_test = y_test.to(device)

            yhat = model(x_test)
            sum_loss += loss_fn(yhat, y_test).item()

            correct += getNumCorrect(yhat, y_test)

    avg_loss = sum_loss / len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)
    accuracy_percent = accuracy * 100

    print(f'\nTest set: Avg. loss: {avg_loss:.4f}, Accuracy: {accuracy} ({accuracy_percent:.1f}%)\n')


def saveModel(model, name ,path='./models'):
    torch.save(model.state_dict(), f'{path}/{name}')

def loadModel(modelClass, name, path='./models'):
    model = modelClass
    model.load_state_dict(torch.load(f'{path}/{name}'))
    model.eval()

    return model