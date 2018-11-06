__author__ = 'ducbui'
from typing import Union, List, Tuple, Iterable, Any

import numpy as np


class DataHelper:
    @staticmethod
    def batch_iter(data: Union[np.ndarray, List[Any]], labels: Union[np.ndarray, List[Any]],
                   batch_size: int, num_epochs: int) -> Tuple[Iterable[Any], Iterable[Any]]:
        """
        A mini-batch iterator to generate mini-batches for training neural network
        :param data: a list of sentences. each sentence is a vector of integers
        :param labels: a list of labels
        :param batch_size: the size of mini-batch
        :param num_epochs: number of epochs
        :return: a mini-batch iterator
        """
        assert len(data) == len(labels)

        for _ in range(num_epochs):
            start_index = 0
            while start_index < len(data) - 1:
                end_index = min(len(data) - 1, start_index + batch_size)

                xdata = data[start_index: end_index]
                ydata = labels[start_index: end_index]

                yield xdata, ydata

                start_index += batch_size
