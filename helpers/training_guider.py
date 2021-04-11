# adapted from https://deeplizard.com/learn/video/NSKghk0pcco

import os
from collections import OrderedDict
from collections import namedtuple
from itertools import product

import torch
import pandas as pd

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from IPython.display import display, clear_output

from helpers import evaluation as vu
from models.MLP import MLPOne, MLPTwo, MLPZero, MLPZeroReLu


class TrainingGuider:
    PARAMETER_KEYS = ['model', 'learning_rate', 'batch_size', 'optimizer']

    def __init__(self, name):
        self.name = name

        self.run_params = None
        self.run_count = 0
        self.run_data = []

        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.tb = None

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'



    def begin_run(self, run, train_loader, val_loader):
        self.run_params = run
        self.run_count += 1

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tb = SummaryWriter(log_dir=f'runs/xp/{self.name}/{run}')


    def end_run(self, val_batch):
        inputs, labels = val_batch
        self.tb.add_figure('predictions vs. actuals',
                            vu.plot_classes_preds(self.model, inputs, labels),
                            global_step=self.epoch_count)
        self.tb.close()
        self.epoch_count = 0
        self.saveModel()
        self.model = None



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



    def _validate(self, params):
        for key in params.keys():
            if key not in TrainingGuider.PARAMETER_KEYS:
                print(f'ERROR: key [{key}] is not expected by TrainingGuider!')
                return False
        for key in TrainingGuider.PARAMETER_KEYS:
            if key not in params.keys():
                print(f'ERROR: key [{key}] is not found in the parameters!')
                return False
        return True


    def _generate_cases(self, params):
        case = namedtuple('case', params.keys())
        cases = [ case(*values) for values in product(*params.values()) ]
        return cases


    def _getOptimizer(self, name, modelParams, lr):
        if name == 'sgd' or None:
            return torch.optim.SGD(modelParams, lr=lr)
        if name == 'adam':
            return torch.optim.Adam(modelParams, lr=lr)


    def run(self, epochs, params, loss_fn, train_set, val_set):
        if self._validate(params) is False: return

        for case in self._generate_cases(params):
            self.model_trainer = ModelTrainer(case.model)

            train_loader = DataLoader(train_set, batch_size=case.batch_size, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_set, batch_size=case.batch_size, num_workers=2)
            optimizer = self._getOptimizer(case.optimizer, case.model.parameters(), case.learning_rate)

            self.begin_run(run, train_loader, val_loader)
            for state in self.model_trainer.fit()

                self.end_epoch()
            self.end_run(val_batch)
        self.save()


    def saveResult(self):
        pd.DataFrame.from_dict(self.run_data, orient='columns').to_csv(f'results/{self.name}.csv')
