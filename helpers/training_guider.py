import os
from collections import OrderedDict
from collections import namedtuple
from itertools import product

import torch.optim
import pandas as pd

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from IPython.display import display, clear_output

from helpers.evaluation import plot_classes_preds
from helpers.model_trainer import ModelTrainer


# loosly based on https://deeplizard.com/learn/video/NSKghk0pcco
class TrainingGuider:
    PARAMETER_KEYS = ['model', 'learning_rate', 'batch_size', 'optimizer']

    def __init__(self, name, override_models=False):
        self.name = name
        self.override_models = override_models
        self.run_data = []

        self.tb = None
        self.model_trainer = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.case = None
        self.case_model = None
        self.case_name = None
        self.case_counter = 0


    def _validate(self, params):
        for key in params.keys():
            if key not in TrainingGuider.PARAMETER_KEYS:
                print(f'ERROR: key [{key}] is not expected by TrainingGuider!')
                return False
        for key in TrainingGuider.PARAMETER_KEYS:
            if key not in params.keys():
                print(f'ERROR: key [{key}] is not found in the parameters!')
                return False
        for idx, model in enumerate(params['model']):
            if not isinstance(model, type):
                print(f'ERROR: model[{idx}] is not a model class type! (Avoid calling constructors "()" in the model list!)')
                return False
        return True


    def _generate_cases(self, params):
        case = namedtuple('case', params.keys())
        cases = [ case(*values) for values in product(*params.values()) ]
        return cases


    def _get_optimizer(self, name, modelParams, lr):
        if name == 'sgd' or None:
            return torch.optim.SGD(modelParams, lr=lr)
        if name == 'adam':
            return torch.optim.Adam(modelParams, lr=lr)


    def _begin(self, case):
        self.case = case
        self.case_counter += 1
        self._extract_case_name()
        self.tb = SummaryWriter(log_dir=f'runs/xp/{self.name}/{self.case_name}')


    def _extract_case_name(self):
        self.case_name = f'model_{self.case_model}'
        for idx, key in enumerate(TrainingGuider.PARAMETER_KEYS):
            if idx == 0: continue
            self.case_name += f'_{key}_{self.case[idx]}'


    def _end(self, val_loader):
        x_val, y_val = list(val_loader)[-1]
        x_val = x_val.to(self.device)
        y_val = y_val.to(self.device)

        plot = plot_classes_preds(self.model_trainer.model, x_val, y_val)
        epochs = len(self.model_trainer.state['losses'])

        self.tb.add_figure('prediction vs. truth', plot, global_step=epochs)
        self.tb.close()
        self.model_trainer.save_model(self.case_name, self.override_models)
        self.model_trainer = None


    def _display_training_state(self, state):
        epoch = len(state['losses'])

        self.tb.add_scalars('Loss', {
            "train_loss": state['losses'][-1],
            "val_loss": state['val_losses'][-1]
        }, epoch)

        self.tb.add_scalars('Accuracy', {
            "train_accuracy": state['accuracy'][-1],
            "val_accuracy": state['val_accuracy'][-1]
        }, epoch)

        results = OrderedDict()
        results["case"] = self.case_counter
        results["epoch"] = epoch
        results['loss'] = state['losses'][-1]
        results['val_loss'] = state['val_losses'][-1]
        results["accuracy"] = state['accuracy'][-1]
        results["val_accuracy"] = state['val_accuracy'][-1]

        for key, value in self.case._asdict().items():
            if key == 'model': continue
            results[key] = value
        results['model'] = self.case_model

        self.run_data.append(results)

        df = pd.DataFrame.from_dict(self.run_data, orient='columns')
        clear_output(wait=True)
        display(df)


    def run(self, epochs, params, loss_fn, target_loss, train_set, val_set):
        if self._validate(params) is False: return

        for case in self._generate_cases(params):
            self.model_trainer = ModelTrainer(case.model)
            self.case_model = f'{type(self.model_trainer.model).__name__}'

            train_loader = DataLoader(train_set, batch_size=case.batch_size, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_set, batch_size=case.batch_size, num_workers=2)
            optimizer = self._get_optimizer(case.optimizer, self.model_trainer.model.parameters(), case.learning_rate)

            self._begin(case)
            for state in self.model_trainer.fit(epochs, loss_fn, target_loss, optimizer, train_loader, val_loader):
                self._display_training_state(state)
            self._end(val_loader)

        self.save_result()


    def save_result(self):
        pd.DataFrame.from_dict(self.run_data, orient='columns').to_csv(f'results/{self.name}.csv')
