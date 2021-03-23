import torch
import numpy as np

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


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):

    train_losses = []
    train_counter = []

    for epoch in range(epochs):
        model.train()
        for b_idx, (xb, yb) in enumerate(train_dl):
            _, loss = loss_batch(model, loss_func, xb, yb, opt)

            if b_idx%60 == 0:
                print(
                    f'Train Epoch: {epoch} [{b_idx*len(xb)}/{len(train_dl.dataset)} ({100. * b_idx / len(train_dl):.0f}%)]\tLoss: {loss:.6f}'
                )

            train_losses.append(loss)
            train_counter.append((b_idx*64) + ((epoch-1)*len(train_dl.dataset)))
            
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