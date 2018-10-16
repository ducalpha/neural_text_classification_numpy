__author__ = 'Duc'

import sys
from pathlib import Path
import sklearn

def read_output_file(output_file: Path):
  outputs = []
  with output_file.open() as f:
    for line in f:
      ith, label = line.split()
      outputs.append(label)
      assert ith == len(outputs)
  return outputs

def compute_metrics(expected_output_file: Path, predicted_output_file: Path):
  expected_values = read_output_file(expected_output_file)
  predicted_values = read_output_file(predicted_output_file)
  assert len(expected_values) == len(predicted_values)
  precision, recall, fscore, support = sklearn.metrics.precision_recall_fscore_support(expected_values, predicted_values)
  assert support == len(expected_values)
  print('Support: {}'.format(support))
  print('Precision | recall | F1: {:.4f} {:.4f} {:.4f}'.format(precision, recall, fscore))

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print('Usage: {} <expected_output> <predicted>'.format(sys.argv[0]))
    sys.exit(-1)
  expected_output_file = Path(sys.argv[1])
  predicted_output_file = Path(sys.argv[2])
  compute_metrics(expected_output_file, predicted_output_file)