import os
import torch


# Based on article: https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e
class ModelTrainer:
    def __init__(self, modelClass):
        self.model = modelClass
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.state = {
            'losses': [],
            'num_correct': [],
            'val_losses': [],
            'val_num_correct': []
        }

    def _create_train_step_function(self, loss_fn, optimizer):
        def train_step(x, y):
            self.model.train()
            y_hat, _ = self.model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            return y_hat, loss.item()
        return train_step


    def _create_validation_step_function(self, loss_fn):
        def validation_step(x, y):
            self.model.eval()
            y_hat, _ = self.model(x)
            val_loss = loss_fn(y_hat, y)
            return y_hat, val_loss.item()
        return validation_step


    def _get_number_of_correct_labels(self, prediction, truth):
        return prediction.argmax(dim=1).eq(truth).sum().item()


    def fit(self, epochs, loss_fn, target_loss, optimizer, train_loader, val_loader):
        train_step = self._create_train_step_function(loss_fn, optimizer)
        validation_step = self._create_validation_step_function(loss_fn)

        for epoch in range(epochs):
            for x_train, y_train in train_loader:
                x_train = x_train.to(self.device)
                y_train = y_train.to(self.device)

                y_hat, loss = train_step(x_train, y_train)
                self.state['num_correct'].append(_get_number_of_correct_labels(y_hat, y_train))
                self.state['losses'].append(loss * x_train.shape[0])

            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val = x_val.to(self.device)
                    y_val = y_val.to(self.device)

                    y_hat, val_loss = validation_step(x_val, y_val)
                    self.state['num_correct'].append(_get_number_of_correct_labels(y_hat, y_val))
                    self.state['val_losses'].append(val_loss * x_val.shape[0])

            # Check if target error rate is reached to avoid overfitting
            if self.state['val_losses'][-1] <= target_loss:
                return self.state
            else:
                yield self.state
        return self.state


    def saveModel(self, name, override=False, path='./models'):
        if os.path.exists(f'{path}/{name}') and not override:
            return f'ERROR: File [{path}/{name}] already exists!'
        torch.save(self.model.state_dict(), f'{path}/{name}')
