__author__ = 'Duc Bui (ducbui)'

"""This implementation is capable of training with batches, so the forward/backward parts are
more complex than in the assignment description."""

from pathlib import Path
import sys
from typing import Dict, Tuple, List, Iterable, Any, Union, Optional
import numpy as np
import sklearn.utils
import sklearn.metrics
import pickle
from collections import Counter
import string

DEFAULT_ENCODING = 'ISO-8859-1'
MODEL_LATEST_FILE = 'model.latest.pkl'


def clean_text(line):
  line = line.lower()
  removed_chars = set(string.punctuation + ' ')
  return ''.join([c for c in line if c not in removed_chars])


def progress(count, total, status=''):
  """from GitHub: https://gist.github.com/vladignatyev/06860ec2040cb497f0f3"""
  bar_len = 60
  filled_len = int(round(bar_len * count / float(total)))

  percents = round(100.0 * count / float(total), 1)
  bar = '=' * filled_len + '-' * (bar_len - filled_len)

  sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
  sys.stdout.flush()


class Vocab:
  def __init__(self):
    self._token_to_index: Dict[str, int] = {}
    self._index_to_token: Dict[int, str] = {}

  def tokens_to_indexes(self, tokens: List[str]) -> np.ndarray:
    raise NotImplementedError

  def indexes_to_tokens(self, indexes: Union[np.ndarray, List[int]]) -> List[str]:
    return [self._index_to_token[idx] for idx in indexes]

  def to_one_hot_encodings(self, tokens: List[str]) -> np.ndarray:
    raise NotImplementedError

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

  def tokens_to_indexes(self, tokens: List[str]) -> np.ndarray:
    return np.array([[self._token_to_index[c] for c in token] for token in tokens])

  def indexes_to_one_hot_encoding(self, token_indexes: np.ndarray) -> np.ndarray:
    assert len(token_indexes) > 0
    encoding = np.eye(len(self._token_to_index))[token_indexes]
    return encoding.reshape((encoding.shape[0], encoding.shape[1] * encoding.shape[2]))

  def to_one_hot_encodings(self, tokens: List[str]) -> np.ndarray:
    token_indexes: np.ndarray = self.tokens_to_indexes(tokens)
    assert len(token_indexes) > 0
    encoding = self.indexes_to_one_hot_encoding(token_indexes)
    assert encoding.shape == (len(tokens), len(self._token_to_index) * len(tokens[0]))
    return encoding


class LabelVocab(Vocab):
  """A token is a label"""

  def _fit(self, train_path: Path, dev_path: Path, test_path: Optional[Path] = None):
    for path in (train_path, dev_path):
      for label, _ in TrainDevDatasetReader(path).iter():
        if label not in self._token_to_index:
          self._token_to_index[label] = len(self._token_to_index)

  def tokens_to_indexes(self, tokens: List[str]) -> np.ndarray:
    return np.array([self._token_to_index[token] for token in tokens])

  def indexes_to_one_hot_encodings(self, token_indexes: np.ndarray) -> np.ndarray:
    return np.eye(len(self._token_to_index))[token_indexes]

  def to_one_hot_encodings(self, tokens: List[str]) -> np.ndarray:
    token_indexes: np.ndarray = self.tokens_to_indexes(tokens)
    encoding = self.indexes_to_one_hot_encodings(token_indexes)
    assert encoding.shape == (len(tokens), len(self._token_to_index))
    return encoding


class DatasetReader:
  def __init__(self, path: Path, encoding: str = DEFAULT_ENCODING):
    self._path = path
    self._encoding = encoding

  @staticmethod
  def get_char_sequences(text: str, char_seq_len: int = 5):
    for i in range(len(text) - char_seq_len):
      yield text[i:i + char_seq_len]

  # def dump(self, data, data_archive_file: Path):
  # with data_archive_file.open('wb') as f:
  # pickle.dump(data, f, protocol=4)

  @staticmethod
  def load_data_maybe_from_disk(train_path: Path, dev_path: Path, char_seq_len: int):
    # read data into byte
    # data_archive_file = Path.home() / 'tmp' / Path('data.pkl')
    data_archive_file = Path('data.pkl')
    if False and data_archive_file.exists():  # disabled because it's fast to read the data.
      print('Read data from data archive...')
      with data_archive_file.open('rb') as f:
        data_train, label_train, data_dev, label_dev = pickle.load(f)
      print('Done reading data from data archive')
    else:
      print('Read data from original data files...')
      data_train, label_train = TrainDevDatasetReader.read_data(train_path, char_seq_len)
      data_dev, label_dev = TrainDevDatasetReader.read_data(dev_path, char_seq_len)
      # self.dump((data_train, label_train, data_dev, label_dev), data_archive_file)
      print('Done reading data from original data files')
    return data_train, label_train, data_dev, label_dev


class TrainDevDatasetReader(DatasetReader):
  def iter(self, need_clean_text: bool = True) -> Tuple[str, str]:
    with self._path.open(encoding=self._encoding) as f:
      for line in f:
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
          label, text = parts
          if need_clean_text:
            text = clean_text(text)
          if not text:
            continue
          yield label, text

  @staticmethod
  def read_data(file_path: Path, char_seq_len: int = 5) -> Tuple[List[str], List[str]]:
    """Return 5-char-seqs and labels lists: ['abc', 'bcd'], ['ENGLISH', 'FRENCH']"""
    dataset_reader = TrainDevDatasetReader(file_path)
    charseq_label_pairs: List[Tuple[str, str]] = []
    for label, text in dataset_reader.iter():
      for char_seq in DatasetReader.get_char_sequences(text, char_seq_len=char_seq_len):
        charseq_label_pairs.append((char_seq, label))
    char_seq_array = [char_seq for char_seq, _ in charseq_label_pairs]
    label_array = [label for _, label in charseq_label_pairs]
    return char_seq_array, label_array

  @staticmethod
  def read_line(file_path: Path, char_seq_len: int = 5, need_clean_text=True) -> np.ndarray:
    dataset_reader = TrainDevDatasetReader(file_path)
    for label, text in dataset_reader.iter():
      if need_clean_text:
        text = clean_text(text)
      yield list(DatasetReader.get_char_sequences(text, char_seq_len=char_seq_len))


class TestDatasetReader(DatasetReader):
  def iter(self, need_clean_text: bool = True) -> str:
    with self._path.open(encoding=self._encoding) as f:
      for line in f:
        if need_clean_text:
          line = clean_text(line)
        yield line

  @staticmethod
  def read_line(file_path: Path, char_seq_len: int = 5, need_clean_text=True) -> np.ndarray:
    """Return ndarray of char sequences of each line."""
    dataset_reader = TestDatasetReader(file_path)
    for text in dataset_reader.iter():
      if need_clean_text:
        text = clean_text(text)
      if not text:
        continue
      yield list(DatasetReader.get_char_sequences(text, char_seq_len=char_seq_len))


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
                 batch_size: int, num_epochs: int = 1) -> Tuple[Iterable[Any], Iterable[Any]]:
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


class Model:
  def __init__(self, input_size: int, hidden_size: int, num_classes: int, learning_rate: float = 0.1):
    # Weights
    np.random.seed(1024)
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

  def evaluate(self, char_vocab: CharVocab, label_vocab: LabelVocab, x: np.ndarray, y_true: np.ndarray) -> float:
    """Return accuracy."""
    num_corrects = 0
    for x_batch, y_batch_true in DataHelper.batch_iter(x, y_true, batch_size=64):
      y_batch_prob = self.forward(x_batch)
      y_batch_pred = self.decode(y_batch_prob)
      num_corrects += sum(y_batch_pred == y_batch_true)
    # return sklearn.metrics.accuracy_score(y_true, y_pred)
    return num_corrects / len(x)


class Trainer:
  def __init__(self, label_vocab: LabelVocab, char_vocab: CharVocab, train_path: Path, dev_path: Path,
               hidden_size: int = 100, learning_rate: float = 0.1, batch_size: int = 1, char_seq_len: int = 5):
    self._label_vocab = label_vocab
    self._char_vocab = char_vocab
    self._train_path = train_path
    self._dev_path = dev_path

    input_size = self._char_vocab.vocab_size * char_seq_len
    num_classes = self._label_vocab.vocab_size

    data_train, label_train, data_dev, label_dev = \
      DatasetReader.load_data_maybe_from_disk(train_path, dev_path, char_seq_len)
    # shuffle the train data
    data_train, label_train = sklearn.utils.shuffle(data_train, label_train, random_state=1024)

    self._x_train, self._y_train = self._char_vocab.tokens_to_indexes(data_train), \
                                   self._label_vocab.tokens_to_indexes(label_train)
    self._x_dev, self._y_dev = self._char_vocab.tokens_to_indexes(data_dev), \
                               self._label_vocab.tokens_to_indexes(label_dev)
    # keep cached encoding as encoding x is too slow.
    self._x_train = self._char_vocab.indexes_to_one_hot_encoding(self._x_train)
    self._y_train_one_hot_encoded = self._label_vocab.indexes_to_one_hot_encodings(self._y_train)
    self._x_dev = self._char_vocab.indexes_to_one_hot_encoding(self._x_dev)
    self._y_dev_one_hot_encoded = self._label_vocab.indexes_to_one_hot_encodings(self._y_dev)
    # self._y_train = self._label_vocab.indexes_to_one_hot_encoding(self._y_train)
    # assert self._char_vocab.to_one_hot_encodings([self._x_train[0]]).shape[0] == input_size \
    # and self._label_vocab.to_one_hot_encodings([self._y_train[0]]).shape[0] == num_classes
    # assert self._char_vocab.to_one_hot_encodings([self._x_dev[0]]).shape[0] == input_size \
    # and self._label_vocab.to_one_hot_encodings([self._y_dev[0]]).shape[0] == num_classes

    self._model = Model(input_size, hidden_size, num_classes, learning_rate)
    self._batch_size = batch_size
    print('Train data size: {}'.format(len(self._x_train)))
    print('Model: batch size: {}, input size:{}, hidden size: {}, learning rate: {}'
          .format(self._batch_size, input_size, hidden_size, learning_rate))

  def fit(self, num_epochs: int = 3):
    train_accuracy = self._model.evaluate(self._char_vocab, self._label_vocab, self._x_train, self._y_train)
    validate_accuracy = self._model.evaluate(self._char_vocab, self._label_vocab, self._x_dev, self._y_dev)
    print('Epoch 0, train_accuracy: {:.6f}, validate_accuracy: {:.6f}'.format(train_accuracy, validate_accuracy))

    for epoch in range(1, num_epochs + 1):
      print('Training epoch {}/{}'.format(epoch, num_epochs))
      total_loss = 0
      for i, (x_batch, y_batch) in enumerate(
          DataHelper.batch_iter(self._x_train, self._y_train_one_hot_encoded, batch_size=self._batch_size)):
        # Encoding here to save the memory.
        # x_batch = self._char_vocab.to_one_hot_encodings(x_batch)
        # y_batch = self._label_vocab.to_one_hot_encodings(y_batch)

        # Forward.
        y_pred = self._model.forward(x_batch)

        # Compute the loss.
        loss = self._model.loss(y_batch, y_pred)
        total_loss += loss
        # print('Epoch: {}, step {}, loss: {:.6f}'.format(epoch, i, loss))

        # Find gradients/losses for each layer.
        self._model.backward(x_batch, y_batch, y_pred)

        # Descent.
        self._model.step()

        # Progress bar.
        progress(i, len(self._x_train))

      # Evaluate on dev set.
      avg_loss = total_loss / len(self._x_train)
      train_accuracy = self._model.evaluate(self._char_vocab, self._label_vocab, self._x_train, self._y_train)
      validate_accuracy = self._model.evaluate(self._char_vocab, self._label_vocab, self._x_dev, self._y_dev)
      self.save_model('model.{}.pkl'.format(epoch))
      self.save_model(MODEL_LATEST_FILE)

      train_sent_accu = self.test(self._train_path)
      validate_sent_accu = self.test(self._dev_path)
      print('Epoch: {}, avg loss: {:.6f}, train_accuracy: {:.6f}, validate_accuracy: {:.6f},'
            ' train sent-accu: {}, validate sent-accu: {}'.
            format(epoch, avg_loss, train_accuracy, validate_accuracy,
                   train_sent_accu, validate_sent_accu))

  def test(self, file_path: Path):
    predictor = Predictor(self._label_vocab, self._char_vocab)
    predictor.from_archive(MODEL_LATEST_FILE)
    label_true = [label for label, _ in TrainDevDatasetReader(file_path).iter()]
    label_pred = predictor.predict(file_path)
    return sklearn.metrics.accuracy_score(label_true, label_pred)

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

  def predict(self, test_file_path: Path) -> List[str]:
    line_predictions = []
    train_or_dev_file = test_file_path.name.startswith('dev') or test_file_path.name.startswith('train')
    read_data_func = TrainDevDatasetReader.read_line if train_or_dev_file else TestDatasetReader.read_line
    for line_batch in read_data_func(test_file_path):
      # line_batch contains an array of
      line_batch = self._char_vocab.to_one_hot_encodings(line_batch)
      probs = self._model.forward(line_batch)
      assert probs.shape == (len(line_batch), self._label_vocab.vocab_size)
      preds = self._decode(self._model.decode(probs))
      # preds: should be ['ENGLISH', 'ENGLISH', 'FRENCH']
      pred = Counter(preds).most_common()[0][0]
      line_predictions.append(pred)
    return line_predictions

  def _decode(self, y_pred: np.ndarray) -> List[str]:
    """y_pred: array of indexes in labels."""
    return [self._label_vocab.decode_label(pred) for pred in y_pred]


class TrainPredictManager:
  PREDICT_OUTPUT_PATH = Path('languageIdentificationPart1.output')

  def maybe_load_from_files(self, train_path: Path, dev_path: Path, test_path: Path,
                            label_vocab_path: Path, char_vocab_path: Path) -> Tuple[LabelVocab, CharVocab]:
    label_vocab: LabelVocab = LabelVocab()
    label_vocab.fit(train_path, dev_path)
    char_vocab: CharVocab = CharVocab()
    char_vocab.fit(train_path, dev_path, test_path)
    return label_vocab, char_vocab

  def write_to_file(self, label_pred: List[str], output_file_path: Path) -> None:
    with output_file_path.open('w') as f:
      f.writelines('Line{} {}\n'.format(i + 1, label) for i, label in enumerate(label_pred))

  def try_to_evaluate_test_result(self, test_path, pred_path):
    evaluation_module = 'evaluate_test_output'
    evaluation_script_file = Path('{}.py'.format(evaluation_module))
    if evaluation_script_file.exists():
      from evaluate_test_output import compute_metrics
      compute_metrics(test_path, pred_path)

  def run(self, train_path: Path, dev_path: Path, test_path: Path, test_only=False, hypertune=False):
    label_vocab, char_vocab = self.maybe_load_from_files(train_path, dev_path, test_path,
                                                         Path('label_vocab.pkl'), Path('char_vocab.pkl'))

    params = [(100, 0.1)] if not hypertune else [(50, 0.1), (200, 0.1), (300, 0.1), (200, 0.01), (300, 0.01)]

    for hidden_size, learning_rate in params:
      if not test_only:
        trainer = Trainer(label_vocab, char_vocab, train_path, dev_path, hidden_size=hidden_size,
                          learning_rate=learning_rate,
                          batch_size=1)
        trainer.fit(num_epochs=5)

      predictor = Predictor(label_vocab, char_vocab)
      predictor.from_archive(MODEL_LATEST_FILE)
      label_pred = predictor.predict(test_path)
      self.write_to_file(label_pred, self.PREDICT_OUTPUT_PATH)
      self.try_to_evaluate_test_result(test_path.parent / 'test_solutions', self.PREDICT_OUTPUT_PATH)

  def run_with_torch(self, train_path, dev_path, test_path):
    label_vocab, char_vocab = self.maybe_load_from_files(train_path, dev_path, test_path,
                                                         Path('label_vocab.pkl'), Path('char_vocab.pkl'))
    from pytorch_model import Model
    # read data.
    data_train, label_train, data_dev, label_dev = \
      DatasetReader.load_data_maybe_from_disk(train_path, dev_path, char_seq_len=5)
    # shuffle the train data
    data_train, label_train = sklearn.utils.shuffle(data_train, label_train, random_state=1024)

    x_train, y_train = char_vocab.tokens_to_indexes(data_train), label_vocab.tokens_to_indexes(label_train)
    x_dev, y_dev = char_vocab.tokens_to_indexes(data_dev), label_vocab.tokens_to_indexes(label_dev)

    # keep cached encoding as encoding x is too slow.
    x_train = char_vocab.indexes_to_one_hot_encoding(x_train)
    y_train_one_hot_encoded = label_vocab.indexes_to_one_hot_encodings(y_train)
    x_dev = char_vocab.indexes_to_one_hot_encoding(x_dev)
    y_dev_one_hot_encoded = label_vocab.indexes_to_one_hot_encodings(y_dev)

    model = Model(input_size=x_train.shape[1], hidden_size=50, num_classes=y_train_one_hot_encoded.shape[1],
                  batch_size=1)
    model.fit(x_train, y_train_one_hot_encoded, x_dev, y_dev_one_hot_encoded, num_epoch=4)


if __name__ == '__main__':
  if len(sys.argv) < 4:
    print('Usage: {} <train_path> <dev_path> <test_path> [only_test] [cuda] hypertune'.format(sys.argv[0]))
    sys.exit(-1)
  train_path = Path(sys.argv[1])
  dev_path = Path(sys.argv[2])
  test_path = Path(sys.argv[3])
  test_only = len(sys.argv) >= 5 and sys.argv[4] == '1'
  use_cuda = len(sys.argv) >= 6 and sys.argv[5] == 'cuda'
  hypertune = len(sys.argv) >= 7 and sys.argv[6] == 'hypertune'

  train_predict_manager = TrainPredictManager()
  if use_cuda:
    train_predict_manager.run_with_torch(train_path, dev_path, test_path)
  else:
    train_predict_manager.run(train_path, dev_path, test_path, test_only=test_only, hypertune=hypertune)
