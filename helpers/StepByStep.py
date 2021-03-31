import numpy as np
import datetime
import matplotlib.pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter

class StepByStep(object):
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
        self.writer = None

        self.losses = []
        self.test_losses = []
        self.val_losses = []
        self.total_epochs = 0

        self.train_step = self._make_train_step()
        self.test_step = self._make_testval_step()
        self.val_step = self._make_testval_step()
    
    def to(self, device):
        self.device = device
        self.model.to(self.device)

    def set_loaders(self, train_loader, test_loader, val_loader=None):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader

    def set_tensorboard(self, name, folder='runs'):
        suffix = datetime.datetime.now().strftime('%Y/%m/%d-%H:%M:%S')
        self.writer = SummaryWriter(f'{folder}/{name}_{suffix}')


    def _make_train_step(self):
        def perform_train_step(x, y):
            self.model.train()
            yhat = self.model(x)
            loss = self.loss_fn(yhat, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            return loss.item()

        return perform_train_step

    def _make_testval_step(self):
        def perform_testval_step(x, y):
            self.model.eval()
            yhat = self.model(x)
            loss = self.loss_fn(yhat, y)
            return loss.item()

        return perform_testval_step

    def _mini_batch(self, type="train"):
        if type == "train":
            data_loader = self.train_loader
            step = self.train_step
        
        elif type == "test":
            data_loader = self.test_loader
            step = self.test_step

        elif type == "val":
            data_loader = self.val_loader
            step = self.val_step

        else:
            return None
       
        mini_batch_losses = []
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            mini_batch_loss = step(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)
            loss = np.mean(mini_batch_losses)

        return loss

    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)

    def train(self, n_epochs, seed=42):
        # To ensure reproducibility of the training process
        self.set_seed(seed)
        
        for epoch in range(n_epochs):
            self.total_epochs += 1

            loss = self._mini_batch(validation=False)
            self.losses.append(loss)

            with torch.no_grad():
                val_loss = self._mini_batch(type="val")
                self.val_losses.append(val_loss)

            # If a SummaryWriter has been set...
            if self.writer:
                scalars = {'training': loss}
                if val_loss is not None:
                    scalars.update({'validation': val_loss})
                
                self.writer.add_scalars(main_tag='loss',
                                        tag_scalar_dict=scalars,
                                        global_step=epoch)

        if self.writer:
            self.writer.flush()

    def save_checkpoint(self, filename):
        # Builds dictionary with all elements for resuming training
        checkpoint = {'epoch': self.total_epochs,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.losses,
                    'test_loss': self.test_losses,
                    'val_loss': self.val_losses}

        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.total_epochs = checkpoint['epoch']
        self.losses = checkpoint['loss']
        self.test_losses = checkpoint['test_loss']
        self.val_losses = checkpoint['val_loss']

        self.model.train() # always use TRAIN for resuming training 

    def predict(self, x):
        self.model.eval() 
        x_tensor = torch.as_tensor(x).float()
        y_hat_tensor = self.model(x_tensor.to(self.device))
        self.model.train()

        return y_hat_tensor.detach().cpu().numpy()

    def plot_losses(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.losses, label='Training Loss', c='b')
        
        if self.val_loader:
            plt.plot(self.val_losses, label='Validation Loss', c='r')
        
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()

        return fig

    def add_graph(self):
        if self.train_loader and self.writer:
            x_sample, y_sample = next(iter(self.train_loader))
            self.writer.add_graph(self.model, x_sample.to(self.device))
    