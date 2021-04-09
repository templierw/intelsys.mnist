# adapted from https://deeplizard.com/learn/video/NSKghk0pcco

import os
import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from IPython.display import display, clear_output
import pandas as pd

from helpers import VizUtils as vu

from collections import OrderedDict
from collections import namedtuple
from itertools import product

from models.MLP import MLPOne, MLPTwo, MLPZero, MLPZeroReLu

class RunBuilder():
    @staticmethod
    def get_runs(params):

        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs

class RunManager():
    def __init__(self, name):
        self.name = name

        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_val_loss = 0
        self.epoch_num_correct = 0
        self.epoch_num_val_correct = 0

        self.run_params = None
        self.run_count = 0
        self.run_data = []

        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.tb = None

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def setModel(self, run):
        try:
            modelName = run.model

            if modelName == "MLPZero":
                self.model = MLPZero()
                
            if modelName == "MLPZeroReLu":
                self.model = MLPZeroReLu()

            if modelName == "MLPOne":
                self.model = MLPOne()

            if modelName == "MLPTwo":
                self.model = MLPTwo()

        except AttributeError:
            self.model = MLPOne()

        finally:
            self.model.to(self.device)


    def getOptimizer(self, name, modelParams, lr):
        if name == 'sgd' or None:
            return torch.optim.SGD(modelParams, lr=lr)
        if name == 'adam':
            return torch.optim.Adam(modelParams, lr=lr)


    def begin_run(self, run, train_loader, val_loader):
        self.run_params = run
        self.run_count += 1

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tb = SummaryWriter(log_dir=f'runs/xp/{self.name}/{run}')

        images, _ = next(iter(self.train_loader))
        images.to(self.device)
        self.tb.add_graph(self.model, images)


    def end_run(self, val_batch):
        inputs, labels = val_batch
        self.tb.add_figure('predictions vs. actuals',
                            vu.plot_classes_preds(self.model, inputs, labels),
                            global_step=self.epoch_count)
        self.tb.close()
        self.epoch_count = 0
        self.saveModel()
        self.model = None


    def begin_epoch(self):
        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_val_loss = 0
        self.epoch_num_correct = 0
        self.epoch_num_val_correct = 0


    def end_epoch(self):
        loss = self.epoch_loss / len(self.train_loader.dataset)
        val_loss = self.epoch_val_loss / len(self.val_loader.dataset)
        accuracy = self.epoch_num_correct / len(self.train_loader.dataset)
        accuracy_val = self.epoch_num_val_correct / len(self.val_loader.dataset)

        self.tb.add_scalars('Loss', {
            "train_loss": loss,
            "val_loss": val_loss
        }, self.epoch_count)

        self.tb.add_scalars('Accuracy', {
            "train_accuracy": accuracy,
            "val_accuracy": accuracy_val
        }, self.epoch_count)

        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results['loss'] = loss
        results['valLoss'] = val_loss
        results["accuracy"] = accuracy
        results["accuracy_val"] = accuracy_val
        for k,v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)

        df = pd.DataFrame.from_dict(self.run_data, orient='columns')

        clear_output(wait=True)
        display(df)


    def track_loss(self, loss, batch, test=True):
        inc_loss = loss.item() * batch[0].shape[0]
        if test:
            self.epoch_loss += inc_loss
        else:
            self.epoch_val_loss += inc_loss


    def get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()


    def track_num_correct(self, preds, labels, test=True):
        num_correct = self.get_num_correct(preds, labels)
        if test:
            self.epoch_num_correct += num_correct
        else:
            self.epoch_num_val_correct += num_correct

    def save(self):
        pd.DataFrame.from_dict(
            self.run_data, orient='columns'
        ).to_csv(f'results/{self.name}.csv')

    def create_train_step_function(self, loss_fn, optimizer):
        def train_step(x, y):
            self.model.train()
            yhat = self.model(x)

            loss = loss_fn(yhat, y)

            loss.backward()
            optimizer.step()

            optimizer.zero_grad()
            return yhat, loss

        return train_step


    def create_validation_step_function(self, loss_fn):
        def validation_step(x, y):
            self.model.eval()
            yhat = self.model(x)
            val_loss = loss_fn(yhat, y)
            return yhat, val_loss

        return validation_step


    def run(self, epochs, params, loss_fn, train_set, val_set):

        for run in RunBuilder.get_runs(params):

            self.setModel(run)

            train_loader = DataLoader(train_set, batch_size=run.batch_size, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_set, batch_size=run.batch_size, num_workers=2)
            optimizer = self.getOptimizer(run.optim, self.model.parameters(), run.lr)

            train_step = self.create_train_step_function(loss_fn, optimizer)
            validation_step = self.create_validation_step_function(loss_fn)

            self.begin_run(run, train_loader, val_loader)
            for epoch in range(epochs):
                self.begin_epoch()
                for batch in train_loader:
                    x_batch = batch[0].to(self.device)
                    y_batch = batch[1].to(self.device)

                    preds, loss = train_step(x_batch, y_batch)

                    self.track_loss(loss, batch, test=True)
                    self.track_num_correct(preds, y_batch, test=True)

                with torch.no_grad():
                    for val_batch in val_loader:
                        x_val = val_batch[0].to(self.device)
                        y_val = val_batch[1].to(self.device)

                        preds, val_loss = validation_step(x_val, y_val)

                        self.track_loss(val_loss, val_batch, test=False)
                        self.track_num_correct(preds, y_val, test=False)

                self.end_epoch()
            self.end_run(val_batch)
        self.save()


    def saveModel(self,path='./models'):
        #if os.path.exists(f'{path}/{name}'):
        #    return f'ERROR: File [{path}/{name}] already exists!'
        torch.save(self.model.state_dict(), f'{path}/{self.name}_{self.run_params}')


    def loadModel(self, modelClass, name, path='./models'):
        #if not os.path.exists(f'{path}/{name}'):
        #    return f'ERROR: File [{path}/{name}] does NOT exists!'

        model = modelClass
        model.load_state_dict(torch.load(f'{path}/{name}'))
        model.eval()
        return model