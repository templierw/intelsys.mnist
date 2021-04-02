import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.1307    # unnormalize
    npimg = img.numpy()
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
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
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