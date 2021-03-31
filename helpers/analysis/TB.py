from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


class TB:

    def __init__(self):
        self.tb = SummaryWriter()

    def plotModel(self, model, train_loader):
        
        images, _ = next(iter(train_loader))
        grid = make_grid(images)
        self.tb.add_image('imagestest', grid)
        self.tb.add_graph(model, images)
        self.tb.close()
