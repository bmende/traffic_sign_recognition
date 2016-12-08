import tensorflow as tf
from tensorflow.python.framework import dtypes
import readTrafficSigns as rt
import numpy as np


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def leNet_like_traffic_network(dataset=None, test_data=None):
    """
    this is from deep mnist for experts tutorial. build a cnn!
    """

    sess = tf.InteractiveSession()

    if dataset == None:
        raise Exception("You must pass in a dataset! You can't train on nothing!")
    else:
        output_size = 43
        depth_multiplier = 3

    x = tf.placeholder(tf.float32, [None, 784*depth_multiplier])
    y_prime = tf.placeholder(tf.float32, [None, output_size])

    # first layer is 5x5 conv on 1 input channel with 32 output channels
    W_conv1 = weight_variable([5, 5, depth_multiplier, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 28, 28, depth_multiplier])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # second layer is 5x5 conv on 32 input channels with 64 output channels
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # fully connected layer from 7x7x64 image into 1024 neurons
    W_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout stuff to prevent overfitting
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # final layer for readout of prediction!
    W_fc2 = weight_variable([1024, output_size])
    b_fc2 = bias_variable([output_size])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # and now train!
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_prime))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_prime, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.global_variables_initializer())


    def eval_in_batches(images, labels, batch_size, session):
        num_test_images = len(images)
        accuracy_to_avg = list()
        for begin in range(0, num_test_images, batch_size):
            end = begin + batch_size
            batch_preds = session.run(accuracy, feed_dict={x: images[begin:end],
                                                           y_prime: labels[begin:end],
                                                           keep_prob: 1.0})
            accuracy_to_avg.append(batch_preds)
        return sum(accuracy_to_avg) / len(accuracy_to_avg)

    for i in range(80000):
        batch = dataset.next_batch(50)

        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: batch[0],
                                                           y_prime: batch[1],
                                                           keep_prob: 1.0})
            print "step {step_num}, training accuracy {t_a}".format(step_num=i, t_a=train_accuracy)

        if i % 5000 == 0:
            print ""
            print "full train accuracy %g" % eval_in_batches(dataset.images, dataset.labels, 50, sess)
            print "test accuracy %g" % eval_in_batches(test_data.images, test_data.labels, 50, sess)
            print ""

        sess.run(train_step, feed_dict={x: batch[0],
                                        y_prime: batch[1],
                                        keep_prob: 0.5})




    print ""
    print "full train accuracy %g" % eval_in_batches(dataset.images, dataset.labels, 50, sess)
    print "test accuracy %g" % eval_in_batches(test_data.images, test_data.labels, 50, sess)
    print ""


def get_the_stuff():
    print 'about to get training data'
    images, label_strings = rt.readTrafficSigns('GTSRB/Final_Training/Images')
    print 'data got.'

    print "getting test data"
    test_im, test_label_str = rt.readTrafficSigns_test('GTSRB/Final_Test/Images')
    print 'test data got'

    images = np.asarray(images)
    label_ints = [int(label) for label in label_strings] # its only 40000, so it doesn't take long!
    labels = np.zeros([len(label_ints), 43])
    for example_number, label in enumerate(label_ints):
        labels[example_number][label] = 1
    gtsrb_dataset = DataSet(images, labels, one_hot=True)

    test_im = np.asarray(test_im)
    test_label_ints = [int(label) for label in test_label_str]
    test_labels = np.zeros([len(test_label_ints), 43])
    for example_number, label in enumerate(test_label_ints):
        test_labels[example_number][label] = 1
    test_dataset = DataSet(test_im, test_labels, one_hot=True)
    return gtsrb_dataset, test_dataset
    print 'data reshaped. now training and other magiks'





class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.

    Copied from mnist input data tutorial
    """
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2] * images.shape[3])
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

if __name__ == "__main__":

    gtsrb_dataset, test_dataset = get_the_stuff()
    leNet_like_traffic_network(dataset=gtsrb_dataset, test_data=test_dataset)
    print 'done!'
