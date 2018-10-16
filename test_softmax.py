import numpy as np
from languageIdentification import row_softmax

def softmax(x):
  """Compute softmax values for each sets of scores in x."""
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum()

def test():
  shape = (5, 6)
  a = np.random.random(shape)
  a_softmax = np.zeros(shape)
  for i in range(a_softmax.shape[0]):
    a_softmax[i] = softmax(a[i])
  a_softmax_by_row = row_softmax(a)
  print(a_softmax)
  print(a_softmax_by_row)

  assert np.allclose(a_softmax, a_softmax_by_row)
  assert np.allclose(np.sum(a_softmax, axis=1), 1.0)

if __name__ == '__main__':
  test()

