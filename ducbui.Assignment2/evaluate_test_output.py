__author__ = 'Duc'

import sys
from pathlib import Path
import sklearn

def read_output_file(output_file: Path, encoding: str = 'utf-8'):
  outputs = []
  with output_file.open(encoding=encoding) as f:
    for line in f:
      ith, label = line.split()
      outputs.append(label)
      if ith.startswith('Line'):
        ith = ith[4:] # format: LineXXX
      assert int(ith) == len(outputs)
  return outputs

def compute_metrics(expected_output_file: Path, predicted_output_file: Path):
  expected_values = read_output_file(expected_output_file, encoding='ISO-8859-1')
  expected_values = [v.upper() for v in expected_values]
  predicted_values = read_output_file(predicted_output_file)
  assert len(expected_values) == len(predicted_values)
  accuracy = sklearn.metrics.accuracy_score(expected_values, predicted_values)
  print('Testing accuracy: {}'.format(accuracy))

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print('Usage: {} <expected_output> <predicted>'.format(sys.argv[0]))
    sys.exit(-1)
  expected_output_file = Path(sys.argv[1])
  predicted_output_file = Path(sys.argv[2])
  compute_metrics(expected_output_file, predicted_output_file)
