__author__ = 'Duc'

from languageIdentification import *


def test_char_vocab(train_path, dev_path, test_path):
  char_vocab = CharVocab()
  char_vocab.fit(train_path, dev_path, test_path)
  encoding = char_vocab.to_one_hot_encoding('Apra')
  print(encoding)
  for i in range(len(encoding)):
    assert encoding[i] == (1 if i in [0, 1, 2, 5] else 0)


def test_label_vocab(train_path, dev_path):
  label_vocab = LabelVocab()
  label_vocab.fit(train_path, dev_path)
  encoding = label_vocab.to_one_hot_encoding('ENGLISH')
  print(encoding)
  for i in range(len(encoding)):
    assert encoding[i] == (1 if i in [0] else 0)


if __name__ == '__main__':
  if len(sys.argv) < 4:
    print('Usage: {} <train_path> <dev_path> <test_path>'.format(sys.argv[0]))
    sys.exit(-1)
  train_path = Path(sys.argv[1])
  dev_path = Path(sys.argv[2])
  test_path = Path(sys.argv[3])
  test_char_vocab(train_path, dev_path, test_path)
  test_label_vocab(train_path, dev_path)
