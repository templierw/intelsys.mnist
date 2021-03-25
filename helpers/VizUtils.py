import matplotlib.pyplot as plt

def plotImages(samples, target, nb):
    if len(samples) != len(target) or nb > len(target):
        print("Cannot display the image(s) - please verify your parameters...")
        return
        
    fig = plt.figure()
    for i in range(nb):
        plt.subplot(nb+1%4,4,i+1)
        plt.tight_layout()
        plt.imshow(samples[i][0], cmap='gray', interpolation='none')
        plt.title(f"Ground Truth: {target[i]}")
        plt.xticks([])
        plt.yticks([])
    
    plt.show()

def plotTrainingCurve(train_counter, train_losses, test_counter, test_losses):
    #test_counter = [i*len(train_dl.dataset) for i in range(epoch + 1)]
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')