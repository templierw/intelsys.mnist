import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt

from helpers.analysis.AnalysisUtils import getNumCorrect

# adaptation from https://deeplizard.com/learn/video/0LhiS6yu2qQ
class ConfusionMatrix:

    def __init__(self, model, trainset):
        self.model = model
        self.trainset = trainset

    def get_all_preds(self, loader):
        all_preds = torch.tensor([])
        for batch in loader:
            images, _ = batch

            preds = self.model(images)
            all_preds = torch.cat(
                (all_preds, preds), dim=0
            )

        return all_preds

    def buildMatrix(self, verbose=False):
        with torch.no_grad():
            prediction_loader = torch.utils.data.DataLoader(self.trainset, batch_size=10000)
            train_preds = self.get_all_preds(prediction_loader)

        preds_correct = getNumCorrect(train_preds, self.trainset.targets)
        stacked = torch.stack(
                (self.trainset.targets,
                train_preds.argmax(dim=1)),
                dim=1
            )

        if verbose:
            print('total correct:', preds_correct)
            print('accuracy:', preds_correct / len(self.trainset))

        cmt = torch.zeros(10,10, dtype=torch.int64)

        for p in stacked:
            tl, pl = p.tolist()
            cmt[tl, pl] = cmt[tl, pl] + 1

        return cmt


    def plot_confusion_matrix(self, title='Confusion matrix', cmap=plt.cm.Blues):
        cm = self.buildMatrix(verbose=False)
        classes = self.trainset.classes

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')