from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
from .model import model_fn
from .input import input_fn

import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam
from tensorflow.contrib.distribute import MirroredStrategy, CollectiveAllReduceStrategy
from tensorflow.logging import info, debug

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--job-dir', type=str, required=True,
    help='location to write checkpoints and export models')
  parser.add_argument('--batch-size', default=128, type=int,
    help='number of records to read during each training step, default=128')
  parser.add_argument('--learning-rate', default=.001, type=float,
    help='learning rate for gradient descent, default=.001')
  parser.add_argument('--verbosity',
    choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
    default='INFO')
  parser.add_argument('--num-gpus', default=0, type=int,
    help='num gpus to user per worker')

  parser.add_argument('--dist',
    choices=['mirror', 'collective'],
    default='mirror')

  parser.add_argument('--subtract-mean', dest='subtract_mean', action='store_true')
  parser.add_argument('--no-subtract-mean', dest='subtract_mean', action='store_false')
  parser.set_defaults(subtract_mean=True)

  return parser.parse_args()

def train_and_evaluate(hparams):
  (train_images, train_labels), (test_images, test_labels) = \
        tf.keras.datasets.cifar10.load_data()

  img_width, img_height, img_channels = 32, 32, 3
  label_dimensions = 10

  n_train_images = len(train_images)
  n_test_images = len(test_images)

  train_images = np.asarray(train_images, dtype=np.float32) / 255
  test_images = np.asarray(test_images, dtype=np.float32) / 255

  if hparams.subtract_mean:
    train_images_mean = np.mean(train_images, axis=0)
    train_images -= train_images_mean
    test_images -= train_images_mean

  train_images = train_images.reshape((-1, img_width, img_height, img_channels))
  test_images = test_images.reshape((-1, img_width, img_height, img_channels))


  debug("shape train_images %s" % (train_images.shape,))
  debug("shape train_labels %s" % (train_labels.shape,))
  debug("shape test_images %s" % (test_images.shape,))
  debug("shape test_labels %s" % (test_labels.shape,))

  train_labels  = tf.keras.utils.to_categorical(train_labels, label_dimensions)
  test_labels = tf.keras.utils.to_categorical(test_labels, label_dimensions)

  train_labels = train_labels.astype(np.float32)
  test_labels = test_labels.astype(np.float32)

  model = model_fn(img_width, img_height, img_channels, label_dimensions, hparams.learning_rate)
  model.summary()

  strategy = {
          'mirror': MirroredStrategy(num_gpus=hparams.num_gpus),
          'collective': CollectiveAllReduceStrategy(num_gpus_per_worker=hparams.num_gpus)
          }[hparams.dist]
  config = tf.estimator.RunConfig(train_distribute=strategy, save_checkpoints_secs=2000)

  estimator = tf.keras.estimator.model_to_estimator(model, \
          model_dir=hparams.job_dir, config=config)

  train_input_fn = lambda: input_fn(train_images, train_labels, hparams.batch_size)
  test_input_fn = lambda: input_fn(test_images, test_labels, hparams.batch_size)

  train_labels = np.asarray(train_labels).astype('int').reshape((-1, label_dimensions))
  train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)

  test_labels = np.asarray(test_labels).astype('int').reshape((-1, label_dimensions))
  eval_spec = tf.estimator.EvalSpec(
          input_fn=test_input_fn,
          steps=n_test_images/hparams.batch_size,
          start_delay_secs=10,
          throttle_secs=10)

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ == '__main__':
  args = get_args()
  tf.logging.set_verbosity(args.verbosity)

  hparams = hparam.HParams(**args.__dict__)
  train_and_evaluate(hparams)
