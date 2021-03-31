
def getNumCorrect(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()