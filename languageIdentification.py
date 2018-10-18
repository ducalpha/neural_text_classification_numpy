__author__ = 'Duc Bui'

from pathlib import Path
import sys
from typing import Dict, Tuple, List, Iterable, Any, Union, Optional
import numpy as np
import sklearn.utils
import sklearn.metrics
import pickle
from collections import Counter


class Vocab:
  def __init__(self):
    self._token_to_index: Dict[str, int] = {}
    self._index_to_token: Dict[int, str] = {}

  def _to_one_hot_encoding(self, token: Union[str, List[str]]) -> np.ndarray:
    token_indexes = np.array([self._token_to_index[c] for c in token])
    encoding = np.zeros((len(token), len(self._token_to_index)))
    encoding[np.arange(len(token)), token_indexes] = 1
    encoding = encoding.flatten()
    assert len(encoding) == len(self._token_to_index) * len(token)
    return encoding

  @property
  def vocab_size(self):
    return len(self._token_to_index)

  def decode_label(self, label_idx: int) -> str:
    return self._index_to_token[label_idx]

  def fit(self, train_path: Path, dev_path: Path, test_path: Optional[Path] = None):
    self._fit(train_path, dev_path, test_path)
    self._index_to_token = {v: k for k, v in self._token_to_index.items()}

  def _fit(self, train_path: Path, dev_path: Path, test_path: Optional[Path] = None):
    raise NotImplementedError


class CharVocab(Vocab):
  """a token is a char"""

  def _fit(self, train_path: Path, dev_path: Path, test_path: Optional[Path] = None):
    for path in (train_path, dev_path):
      for _, text in TrainDevDatasetReader(path).iter():
        for c in text:
          if c not in self._token_to_index:
            self._token_to_index[c] = len(self._token_to_index)
    for text in TestDatasetReader(test_path).iter():
      for c in text:
        if c not in self._token_to_index:
          self._token_to_index[c] = len(self._token_to_index)
    # print(self._token_to_index)

  def to_one_hot_encoding(self, token: str) -> np.ndarray:
    return super()._to_one_hot_encoding(token)


class LabelVocab(Vocab):
  """A token is a label"""

  def _fit(self, train_path: Path, dev_path: Path, test_path: Optional[Path] = None):
    for path in (train_path, dev_path):
      for label, _ in TrainDevDatasetReader(path).iter():
        if label not in self._token_to_index:
          self._token_to_index[label] = len(self._token_to_index)
    # print(self._token_to_index)

  def to_one_hot_encoding(self, token: str) -> np.ndarray:
    return super()._to_one_hot_encoding([token])


class DatasetReader:
  def __init__(self, path: Path, encoding='ISO-8859-1'):
    self._path = path
    self._encoding = encoding

  @staticmethod
  def get_char_sequences(text: str, char_seq_len: int = 5):
    for i in range(len(text) - char_seq_len):
      yield text[i:i + char_seq_len]


class TrainDevDatasetReader(DatasetReader):
  def iter(self) -> Tuple[str, str]:
    with self._path.open(encoding=self._encoding) as f:
      for line in f:
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
          label, text = parts
          yield label, text

  @staticmethod
  def read_data(file_path: Path, label_vocab: LabelVocab, char_vocab: CharVocab,
                char_seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    dataset_reader = TrainDevDatasetReader(file_path)
    charseq_label_pairs: List[Tuple[str, str]] = []
    for label, text in dataset_reader.iter():
      for char_seq in DatasetReader.get_char_sequences(text, char_seq_len=char_seq_len):
        charseq_label_pairs.append((char_seq, label))
    char_seq_array = np.array([char_vocab.to_one_hot_encoding(char_seq)
                               for char_seq, _ in charseq_label_pairs])
    label_array = np.array([label_vocab.to_one_hot_encoding(label)
                            for _, label in charseq_label_pairs])
    return char_seq_array, label_array


class TestDatasetReader(DatasetReader):
  def iter(self) -> str:
    with self._path.open(encoding=self._encoding) as f:
      for line in f:
        yield line

  @staticmethod
  def read_line(file_path: Path, char_vocab: CharVocab,
                char_seq_len: int = 5) -> np.ndarray:
    """Return ndarray of char sequences of each line."""
    dataset_reader = TestDatasetReader(file_path)
    for text in dataset_reader.iter():
      char_sequence_array = np.array([char_vocab.to_one_hot_encoding(char_seq)
                                      for char_seq in
                                      DatasetReader.get_char_sequences(text, char_seq_len=char_seq_len)])
      yield char_sequence_array


def sigmoid(x, derivative=False):
  return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))


def row_softmax(x):
  """Column-wise softmax."""
  exp = np.exp(x - np.max(x, axis=1)[:, np.newaxis])  # for stability, reduce by max
  return exp / exp.sum(axis=1)[:, np.newaxis]  # sum axis = 1 argument sums over axis representing columns


def softmax_grad(s):
  return np.diagflat(s) - np.outer(s, s)


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
      while start_index < len(data):
        end_index = min(len(data) - 1, start_index + batch_size)

        xdata = data[start_index: end_index]
        ydata = labels[start_index: end_index]

        yield xdata, ydata

        start_index += batch_size


class Model:
  def __init__(self, input_size: int, hidden_size: int, num_classes: int, learning_rate: float = 0.1):
    # Weights
    self._W1: np.ndarray = np.random.random((input_size, hidden_size))  # each weight for 1 input = 1 column
    self._W2: np.ndarray = np.random.random((hidden_size, num_classes))
    self._b1: np.ndarray = np.random.random(hidden_size)
    self._b2: np.ndarray = np.random.random(num_classes)
    self._h: np.ndarray = None  # intermediate reults
    self._input_size = input_size
    self._hidden_size = hidden_size
    self._num_classes: int = num_classes
    self._lr: float = learning_rate

    # Gradients.
    self._delta_W1_L: np.ndarray = None
    self._delta_b1_L: np.ndarray = None
    self._delta_W2_L: np.ndarray = None
    self._delta_b2_L: np.ndarray = None

  def loss(self, y, y_pred):
    output = np.zeros((y.shape[0]))
    diff = y - y_pred
    for i in range(y.shape[0]):
      output[i] = np.inner(diff[i], diff[i]) / 2
    return np.average(output)

  def forward(self, inputs: np.ndarray) -> np.ndarray:
    batch_size = inputs.shape[0]
    b1_batch = np.repeat(self._b1[np.newaxis, :], batch_size, axis=0)
    assert b1_batch.shape == (batch_size, self._hidden_size)

    hp = inputs.dot(self._W1) + b1_batch
    assert hp.shape == (batch_size, self._hidden_size)

    self._h = sigmoid(hp)
    assert self._h.shape == (batch_size, self._hidden_size)

    b2_batch = np.repeat(self._b2[np.newaxis, :], batch_size, axis=0)
    assert b2_batch.shape == (batch_size, self._num_classes)

    yp = self._h.dot(self._W2) + b2_batch
    assert yp.shape == (batch_size, self._num_classes)

    y_prob = row_softmax(yp)
    assert y_prob.shape == (batch_size, self._num_classes)
    assert np.allclose(np.sum(y_prob, axis=1), 1.0)

    return y_prob

  def backward(self, inputs: np.ndarray, y: np.ndarray, y_prob: np.ndarray):
    batch_size = inputs.shape[0]

    delta_yprob_L = y_prob - y  # \delta_y L
    assert delta_yprob_L.shape == (batch_size, self._num_classes)

    delta_yp_y: np.ndarray = np.zeros((batch_size, self._num_classes, self._num_classes))
    for i in range(y_prob.shape[0]):
      delta_yp_y[i] = softmax_grad(y_prob[i])

    delta_yp_L = np.matmul(delta_yprob_L.reshape((batch_size, 1, self._num_classes)), delta_yp_y)
    assert delta_yp_L.shape == (batch_size, 1, self._num_classes)
    delta_yp_L = delta_yp_L.reshape(batch_size, self._num_classes)
    assert delta_yp_L.shape == (batch_size, self._num_classes)

    self._delta_W2_L = self._h.transpose().dot(delta_yp_L) / batch_size
    assert self._delta_W2_L.shape == (self._hidden_size, self._num_classes)

    self._delta_b2_L = np.average(delta_yp_L, axis=0)
    assert self._delta_b2_L.shape == (self._num_classes,)

    delta_h_L = delta_yp_L.dot(self._W2.transpose())
    assert delta_h_L.shape == (batch_size, self._hidden_size)

    delta_hp_L = np.multiply(delta_h_L, np.multiply(self._h, 1 - self._h))
    assert delta_hp_L.shape == (batch_size, self._hidden_size)

    self._delta_W1_L = inputs.transpose().dot(delta_hp_L) / batch_size
    assert self._delta_W1_L.shape == (self._input_size, self._hidden_size)

    self._delta_b1_L = np.average(delta_hp_L, axis=0)
    assert self._delta_b1_L.shape == (self._hidden_size,)

  def step(self):
    # Descent following the gradients.
    self._W1 -= self._lr * self._delta_W1_L
    self._b1 -= self._lr * self._delta_b1_L
    self._W2 -= self._lr * self._delta_W2_L
    self._b2 -= self._lr * self._delta_b2_L

  def decode(self, probs: np.ndarray) -> np.ndarray:
    """From probabilities like [0.9, 0.1, 0] to predictions (0, 1, 2)"""
    return np.argmax(probs, axis=1)

  def evaluate(self, x: np.ndarray, y_true_probs: np.ndarray) -> float:
    """Return accuracy."""
    y_true = self.decode(y_true_probs)
    y_prob = self.forward(x)
    y_pred = self.decode(y_prob)
    return sklearn.metrics.accuracy_score(y_true, y_pred)


class Trainer:
  def __init__(self, train_path: Path, dev_path: Path, label_vocab: LabelVocab, char_vocab: CharVocab,
               hidden_size: int = 100, learning_rate: float = 0.1, batch_size: int = 1, char_seq_len: int = 5):
    input_size = char_vocab.vocab_size * char_seq_len
    num_classes = label_vocab.vocab_size

    self._x_train, self._y_train, self._x_dev, self._y_dev = \
      self.load_data_maybe_from_disk(label_vocab, char_vocab, char_seq_len)
    assert self._x_train.shape[1] == input_size and self._y_train.shape[1] == num_classes
    assert self._x_dev.shape[1] == input_size and self._y_dev.shape[1] == num_classes
    # shuffle the data
    sklearn.utils.shuffle(self._x_train, self._y_train)

    self._model = Model(input_size, hidden_size, num_classes, learning_rate)
    self._batch_size = batch_size

  def dump(self, data, data_archive_file: Path):
    with data_archive_file.open('wb') as f:
      pickle.dump(data, f, protocol=4)

  def load_data_maybe_from_disk(self, label_vocab: LabelVocab, char_vocab: CharVocab, char_seq_len: int):
    # read data into byte
    data_archive_file = Path.home() / 'tmp' / Path('data.pkl')
    if data_archive_file.exists():
      print('Read data from data archive...')
      with data_archive_file.open('rb') as f:
        x_train, y_train, x_dev, y_dev = pickle.load(f)
      print('Done reading data from data archive')
    else:
      print('Read data from original data files...')
      x_train, y_train = TrainDevDatasetReader.read_data(train_path, label_vocab, char_vocab, char_seq_len)
      x_dev, y_dev = TrainDevDatasetReader.read_data(dev_path, label_vocab, char_vocab, char_seq_len)
      self.dump((x_train, y_train, x_dev, y_dev), data_archive_file)
      print('Done reading data from original data files')
    return x_train, y_train, x_dev, y_dev

  def fit(self, num_epochs: int = 3):
    for epoch in range(num_epochs):
      for i, (x_batch, y_batch) in enumerate(DataHelper.batch_iter(self._x_train, self._y_train, self._batch_size, 1)):
        # Forward.
        y_pred = self._model.forward(x_batch)

        # Compute the loss.
        loss = self._model.loss(y_batch, y_pred)
        print('Epoch: {}, step {}, loss: {}'.format(epoch, i, loss))

        # Find gradients/losses for each layer.
        self._model.backward(x_batch, y_batch, y_pred)

        # Descent.
        self._model.step()

      # Evaluate on dev set.
      train_accuracy = self._model.evaluate(self._x_train, self._y_train)
      validate_accuracy = self._model.evaluate(self._x_dev, self._y_dev)
      print('Epoch: {}, train_accuracy: {}, validate_accuracy: {}'.format(epoch, train_accuracy, validate_accuracy))

  def save_model(self, model_path: str):
    with open(model_path, 'wb') as f:
      pickle.dump(self._model, f)


class Predictor:
  def __init__(self, label_vocab: LabelVocab, char_vocab: CharVocab):
    self._model: Model = None
    self._label_vocab: LabelVocab = label_vocab
    self._char_vocab: CharVocab = char_vocab

  def from_archive(self, model_path: str):
    with open(model_path, 'rb') as f:
      self._model = pickle.load(f)

  def predict(self, test_file_path) -> List[str]:
    line_predictions = []
    for line_batch in TestDatasetReader.read_line(test_file_path, char_vocab=self._char_vocab):
      # line_batch contains an array of
      probs = self._model.forward(line_batch)
      assert probs.shape == (self._label_vocab.vocab_size, len(line_batch))
      preds = self._decode(self._model.decode(probs))
      # preds: should be ['ENGLISH', 'ENGLISH', 'FRENCH']
      pred = Counter(preds).most_common()
      line_predictions.append(pred)
    return line_predictions

  def _decode(self, y_pred: np.ndarray) -> List[str]:
    """y_pred: array of indexes in labels."""
    return [self._label_vocab.decode_label(pred) for pred in y_pred]


class TrainPredictManager:
  MODEL_PATH = 'model.pkl'
  OUTPUT_PATH = 'languageIdentificationPart1.output'

  def maybe_load_from_files(self, train_path: Path, dev_path: Path, test_path: Path,
                            label_vocab_path: Path, char_vocab_path: Path) -> Tuple[LabelVocab, CharVocab]:
    label_vocab: LabelVocab = LabelVocab()
    label_vocab.fit(train_path, dev_path)
    char_vocab: CharVocab = CharVocab()
    char_vocab.fit(train_path, dev_path, test_path)
    return label_vocab, char_vocab

  def write_to_file(self, label_pred: List[str]) -> None:
    with open(self.OUTPUT_PATH, 'w') as f:
      f.writelines('{} {}'.format(i, label) for i, label in enumerate(label_pred))

  def run(self, train_path, dev_path, test_path, test_only=False):
    label_vocab, char_vocab = self.maybe_load_from_files(train_path, dev_path, test_path,
                                                         Path('label_vocab.pkl'), Path('char_vocab.pkl'))

    if not test_only:
      trainer = Trainer(train_path, dev_path, label_vocab, char_vocab, hidden_size=100, learning_rate=0.01,
                        batch_size=1024)
      trainer.fit(num_epochs=3)
      trainer.save_model(self.MODEL_PATH)

    return
    predictor = Predictor(label_vocab, char_vocab)
    predictor.from_archive(self.MODEL_PATH)
    label_pred = predictor.predict(test_path)
    self.write_to_file(label_pred)

  def run_with_torch(self, train_path, dev_path, test_path):
    label_vocab, char_vocab = self.maybe_load_from_files(train_path, dev_path, test_path,
                                                         Path('label_vocab.pkl'), Path('char_vocab.pkl'))
    from pytorch_model import Model
    x_train, y_train = TrainDevDatasetReader.read_data(train_path, label_vocab, char_vocab, char_seq_len=5)
    x_dev, y_dev = TrainDevDatasetReader.read_data(dev_path, label_vocab, char_vocab, char_seq_len=5)
    model = Model(input_size=x_train.shape[1], hidden_size=50, num_classes=y_train.shape[1])
    model.fit(x_train, y_train, x_dev, y_dev, num_epoch=500)


if __name__ == '__main__':
  if len(sys.argv) < 4:
    print('Usage: {} <train_path> <dev_path> <test_path> [only_test]'.format(sys.argv[0]))
    sys.exit(-1)
  train_path = Path(sys.argv[1])
  dev_path = Path(sys.argv[2])
  test_path = Path(sys.argv[3])
  test_only = False if len(sys.argv) < 5 else True
  train_predict_manager = TrainPredictManager()
  train_predict_manager.run(train_path, dev_path, test_path, test_only=test_only)
  # train_predict_manager.run_with_torch(train_path, dev_path, test_path)
