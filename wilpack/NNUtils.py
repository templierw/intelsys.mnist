import torch
from torch.utils.data import DataLoader
from torchvision import datasets as dt, transforms

def getDataLoader(batch_size=4, num_workers=2, path="./dataset/", train=True, shuffle=True):

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))] # value needs to be recomputed, taken from internet
    )

    set = dt.MNIST(root=path, train=train,
                   download=True, transform=transform)

    return DataLoader(set, batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers)

def loadMNIST(batch_size=4, num_workers=2, path="./dataset/"):

    trainloader = getDataLoader(
        batch_size=batch_size, 
        num_workers=num_workers, 
        path=path
    )

    testloader = getDataLoader(
        batch_size=batch_size, 
        num_workers=num_workers, 
        path=path,
        train=False,
        shuffle=False
    )
    
    return trainloader, testloader


def getSGDOptim(model, lr, momentum=0):
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)


def loss_batch(model, loss_func, xb, yb, opt=None):
    output = model(xb)
    loss = loss_func(output, yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return output, loss.item()


def fit(epochs, model, loss_func, opt, train_dl):

    train_losses = []
    train_counter = []

    for epoch in range(epochs):
        model.train()
        for b_idx, (xb, yb) in enumerate(train_dl):
            _, loss = loss_batch(model, loss_func, xb, yb, opt)

            if b_idx % 1000 == 0:
                print(
                    f'Train Epoch: {epoch} [{100. * b_idx / len(train_dl):.0f}%] \tLoss: {loss:.6f}'
                )

            train_losses.append(loss)
            train_counter.append(
                (b_idx*64) + ((epoch-1)*len(train_dl.dataset)))

            #torch.save(model.state_dict(), '/results/model.pth')
            #torch.save(opt.state_dict(), '/results/optimizer.pth')


def test(model, loss_func, test_dl):
    test_losses = []
    test_loss = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for data, target in test_dl:
            output, loss = loss_batch(model, loss_func, data, target)
            test_loss += loss
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(test_dl.dataset)
    test_losses.append(test_loss)
    print(f'\nTest set: Avg. loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_dl.dataset)}\
         ({100. * correct / len(test_dl.dataset):.0f}%)\n')
