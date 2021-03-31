# adapted from https://deeplizard.com/learn/video/NSKghk0pcco

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from IPython.display import display, clear_output
import pandas as pd
import json

from collections import OrderedDict
from collections import namedtuple
from itertools import product

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

    def getOptimizer(self, name, modelParams, lr):

        if name == 'sgd' or None:
            return torch.optim.SGD(modelParams, lr=lr)

        if name == 'adam':
            return torch.optim.Adam(modelParams, lr=lr)

    def begin_run(self, run, model, train_loader, val_loader):

        self.run_params = run
        self.run_count += 1

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tb = SummaryWriter(log_dir=f'runs/xp/{self.name}',comment=f'-{run}')

        images, _ = next(iter(self.train_loader))
        grid = torchvision.utils.make_grid(images)

        self.tb.add_image('images', grid)
        self.tb.add_graph(self.model, images)

    def end_run(self):
        self.tb.close()
        self.epoch_count = 0

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

        self.tb.add_scalar('Loss', loss, self.epoch_count)
        self.tb.add_scalar('ValLoss', val_loss, self.epoch_count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)
        self.tb.add_scalar('AccuracyVal', accuracy_val, self.epoch_count)

        #for name, param in self.model.named_parameters():
        #    self.tb.add_histogram(name, param, self.epoch_count)
        #    self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

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
    
    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    def track_num_correct(self, preds, labels, test=True):
        num_correct = self._get_num_correct(preds, labels)
        if test:
            self.epoch_num_correct += num_correct
        else:
            self.epoch_num_val_correct += num_correct

    def save(self):
        pd.DataFrame.from_dict(
            self.run_data, orient='columns'
        ).to_csv(f'results/{self.name}.csv')

        with open(f'results/{self.name}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)

    def create_train_step_function(self, loss_fn, optimizer):
        def train_step(x, y):
            self.model.train()
            yhat = self.model(x)

            loss = loss_fn(yhat, y)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            return yhat, loss

        return train_step

    def create_validation_step_function(self, loss_fn):
        def validation_step(x, y):
            self.model.eval()
            yhat = self.model(x)
            val_loss = loss_fn(yhat, y)
            return yhat, val_loss

        return validation_step

    def fit(self, model, epochs, params, loss_fn, train_set, test_set):
        
        for run in RunBuilder.get_runs(params):
            train_loader = DataLoader(train_set, batch_size=run.batch_size)
            val_loader = DataLoader(test_set, batch_size=run.batch_size)
            optimizer = self.getOptimizer(run.optim, model.parameters(), run.lr)

            train_step = self.create_train_step_function(loss_fn, optimizer)
            validation_step = self.create_validation_step_function(loss_fn)

            self.begin_run(run, model, train_loader, val_loader)
            for epoch in range(epochs):
                self.begin_epoch()
                for batch in train_loader:
                    x_batch = batch[0].to(self.device)
                    y_batch = batch[1].to(self.device)

                    preds, loss = train_step(x_batch, y_batch)

                    self.track_loss(loss, batch, test=True)
                    self.track_num_correct(preds, y_batch, test=True)

                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        x_val = x_val.to(self.device)
                        y_val = y_val.to(self.device)

                        preds, val_loss = validation_step(x_val, y_val)

                        self.track_loss(val_loss, batch, test=False)
                        self.track_num_correct(preds, y_val, test=False)

                self.end_epoch()
            self.end_run()
        self.save()