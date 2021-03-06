from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.random
import torch.tensor
from sklearn.metrics import accuracy_score

from data_helper import DataHelper


def generate_data() -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(3)
    mean_1 = [2.0, 0.2];
    cov_1 = [[1, .5], [.5, 2.0]]
    mean_2 = [0.4, -2.0];
    cov_2 = [[1.25, -0.2], [-0.2, 1.75]]
    x_1, y_1 = np.random.multivariate_normal(mean_1, cov_1, 15).T
    x_2, y_2 = np.random.multivariate_normal(mean_2, cov_2, 15).T

    X = np.zeros((30, 2))
    X[0:15, 0] = x_1
    X[0:15, 1] = y_1
    X[15:, 0] = x_2
    X[15:, 1] = y_2

    y = np.zeros(30)
    y[15:] = 1 * np.ones(15)

    y_ = np.zeros((y.shape[0], 2))
    y_[np.arange(y.shape[0]), y.astype(int)] = 1
    return X, y_


class Model:
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, batch_size: int = 1):
        self._model = nn.Sequential(nn.Linear(input_size, hidden_size),
                                    nn.Sigmoid(), nn.Linear(hidden_size, num_classes), nn.Softmax(dim=1))
        self._model.cuda()
        self._criterion = nn.MSELoss()
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=0.01)
        self._batch_size = batch_size

    def _train(self, X, y):
        assert len(X) > 0
        self._model.train()
        # Forward.
        y_pred = self._model(X)

        loss = self._criterion(y_pred, y)

        self._optimizer.zero_grad()

        loss.backward()
        self._optimizer.step()

        return loss

    def _eval(self, X, y):
        self._model.eval()
        with torch.no_grad():
            probs = self._model.forward(X)
            preds = torch.max(probs, 1)[1].cpu().numpy()
            y = torch.max(y, 1)[1].cpu().numpy()
            accuracy = accuracy_score(y, preds)
            return accuracy

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_dev: Optional[np.ndarray] = None, y_dev: Optional[np.ndarray] = None, num_epoch: int = 50) -> None:
        X_train, y_train = torch.from_numpy(X_train).float().cuda(), torch.from_numpy(y_train).float().cuda()
        X_dev, y_dev = torch.from_numpy(X_dev).float().cuda(), torch.from_numpy(y_dev).float().cuda()

        train_accuracy = self._eval(X_train, y_train)
        validate_accuracy = self._eval(X_dev, y_dev)
        print('Epoch: 0, train_accuracy: {}, validate_accuracy: {}'.format(train_accuracy, validate_accuracy))

        for epoch in range(1, num_epoch + 1):
            total_loss = 0
            for i, (x_batch, y_batch) in enumerate(DataHelper.batch_iter(X_train, y_train, self._batch_size, 1)):
                loss = self._train(x_batch, y_batch)
                total_loss += loss
                step = i * len(x_batch)
                if step % 10000 == 0:
                    print('Epoch: {}, step: {}/{}, loss: {}'.format(epoch, step, len(X_train), loss))

            train_accuracy = self._eval(X_train, y_train)
            validate_accuracy = self._eval(X_dev, y_dev)
            # print('train accuracy', train_accuracy)
            # print('validate accuracy', validate_accuracy)
            avg_loss = total_loss / len(X_train)
            print('Epoch: {}, avg loss: {}, train_accuracy: {}, validate_accuracy: {}'.
                  format(epoch, avg_loss, train_accuracy, validate_accuracy))


def test_with_generated_data():
    X, y = generate_data()
    # print('X', X)
    # print('y', y)
    model = Model(input_size=X.shape[1], hidden_size=5, num_classes=y.shape[1], batch_size=1024)
    model.fit(X, y, num_epoch=500)


if __name__ == '__main__':
    test_with_generated_data()
