import os
import sklearn.metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.1307    # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def plotImages(samples, target, nb):
    if len(samples) != len(target) or nb > len(target):
        print("Cannot display the image(s) - please verify your parameters...")
        return

    fig = plt.figure()
    for i in range(nb):
        plt.subplot(nb+1%4,4,i+1)
        plt.tight_layout()
        img = np.reshape(samples[i][0], (-1, 28, 28, 1))
        plt.imshow(img, cmap='gray', interpolation='none')
        plt.title(f"Ground Truth: {target[i]}")
        plt.xticks([])
        plt.yticks([])

    #plt.show()
    return fig


def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    net.eval()
    output, _ = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(10, 14))
    for idx in np.arange(16):
        ax = fig.add_subplot(4, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title(f"Pred {classes[preds[idx]]}, {(probs[idx] * 100.0):.1f}%\n(label: {classes[labels[idx]]})",
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig


def ComputeConfusionMatrices(model, holdback_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    global_cm = np.zeros((10,2,2), dtype=np.uint64)
    with torch.no_grad():
        model.eval()
        for x, y in holdback_loader:
            x = x.to(device)
            yhat, _ = model(x)
            yhat = torch.argmax(yhat, 1).to('cpu')
            cm = sklearn.metrics.multilabel_confusion_matrix(y, yhat, labels=list(range(10)))
            global_cm = np.add(global_cm, cm.astype('uint64'))
    return global_cm


def loadModel(modelClass, name, path='./models/saved'):
    if not os.path.exists(f'{path}/{name}'):
        return f'ERROR: File [{path}/{name}] does NOT exists!'

    model = modelClass
    model.load_state_dict(torch.load(f'{path}/{name}'))
    model.eval()
    return model


def validate(model, loss_fn, holdback_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    sum_loss = 0
    correct = 0

    with torch.no_grad():
        model.eval()
        for x_hb, y_hb in holdback_loader:
            x_hb = x_hb.to(device)
            y_hb = y_hb.to(device)

            y_hat, _ = model(x_hb)
            sum_loss += loss_fn(y_hat, y_hb).item()

            correct += y_hat.argmax(dim=1).eq(y_hb).sum().item()

    avg_loss = sum_loss / len(holdback_loader.dataset)
    accuracy = correct / len(holdback_loader.dataset)
    accuracy_percent = accuracy * 100

    print(f'\nHoldBackSet: Avg. loss: {avg_loss:.4f}, Accuracy: {accuracy} ({accuracy_percent:.1f}%)\n')


def getPerformanceMetrics(cms):
    results = pd.DataFrame(index=['0','1','2','3','4','5','6','7','8','9', 'total'], columns=['accuracy', 'precision', 'recall', 'f1'])

    for i, cm in enumerate(cms):

        tp = cm[0][0]
        fn = cm[0][1]
        fp = cm[1][0]
        tn = cm[1][1]

        accuracy = (tp + tn)/(tp + tn + fp + fn)
        recall = tp/(tp+fn)
        precision = tp/(tp+fp)

        f1 = (precision * recall)/(precision + recall)

        results.loc[f'{i}','accuracy'] = accuracy
        results.loc[f'{i}','precision'] = precision
        results.loc[f'{i}','recall'] = recall
        results.loc[f'{i}','f1'] = f1

        results.loc['total','accuracy'] = results['accuracy'].mean()
        results.loc['total','precision'] = results['precision'].mean()
        results.loc['total','recall'] = results['recall'].mean()
        results.loc['total','f1'] = results['f1'].mean()

    return results.astype(float).round(3)
