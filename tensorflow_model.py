from typing import Optional

import numpy as np
import tensorflow as tf

from data_helper import DataHelper


def prob_to_class_idx(prob: tf.Tensor):
    return tf.argmax(prob, 1)


class Model:
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, batch_size: int):
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._num_classes = num_classes
        self._learning_rate = 0.1
        self._batch_size = batch_size

        self._W1 = tf.Variable(tf.random_normal([self._input_size, self._hidden_size]))
        self._W2 = tf.Variable(tf.random_normal([self._input_size, self._num_classes]))
        self._b1 = tf.Variable(tf.random_normal([self._hidden_size]))
        self._b2 = tf.Variable(tf.random_normal([self._num_classes]))

        self._input = tf.placeholder("float", [None, self._input_size])
        self._y = tf.placeholder("float", [None, self._num_classes])  # [[1,0,0], [0,1,0]...]

        self._h_p = tf.add(tf.matmul(self._input, self._W1), self._b1)
        self._h = tf.sigmoid(self._h_p)
        self._y_p = tf.add(tf.matmul(self._h, self._W2), self._b2)
        self._y_prob = tf.sparse_softmax(self._y_p)
        self._y_pred = prob_to_class_idx(self._y_prob)

        # Loss.
        self._loss = tf.nn.l2_loss(self._y_pred - self._y)

        # Optimizer.
        self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._learning_rate)
        self._train_op = self._optimizer.minimize(self._loss)

        # Accuracy.
        self._y_corrects = tf.equal(self._y_pred, self._y)
        self._accuracy = tf.reduce_mean(tf.cast(self._y_corrects, dtype=tf.int8))

    def _train(self, sess: tf.Session, x_batch, y_batch):
        # forward.
        loss, _ = sess.run([self._loss, self._train_op])
        return loss

    # model.fit(x_train, y_train_one_hot_encoded, x_dev, y_dev_one_hot_encoded, num_epoch=4)
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_dev: Optional[np.ndarray] = None, y_dev: Optional[np.ndarray] = None, num_epoch: int = 50) -> None:
        X_train, y_train = tf.convert_to_tensor(X_train), tf.convert_to_tensor(y_train)
        X_dev, y_dev = tf.convert_to_tensor(X_dev), tf.convert_to_tensor(y_dev)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            for epoch in range(num_epoch + 1):
                total_loss = 0
                for i, (x_batch, y_batch) in enumerate(DataHelper.batch_iter(X_train, y_train, self._batch_size, 1)):
                    loss = self._train(sess, x_batch, y_batch)
                    total_loss += loss
                    step = i * len(x_batch)
                    if step % 100000 == 0:
                        print('Epoch: {}, step {}/{}, loss: {}'.format(epoch, step, len(X_train), loss))

                train_acc = sess.run([self._accuracy], feed_dict={self._input: X_dev, self._y: y_dev})
                validate_acc = sess.run([self._accuracy], feed_dict={self._input: X_dev, self._y: y_dev})
                avg_loss = total_loss / len(X_train)
                print('Epoch: {}, avg train loss: {}, train accuracy: {}, validate accuracy: {}'.format(epoch, avg_loss, train_acc, validate_acc))
