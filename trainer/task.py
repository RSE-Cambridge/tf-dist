from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
from model import model_fn
from input import input_fn

import sys
sys.path.append('/home/pc620/proj/tf-dist/')
import os

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

  parser.add_argument('--dataset',
    choices=['cifar10', 'ecoset-tfr64', 'ecoset-tfr', 'ecoset', 'imagenet', 'ecoset-h5', 'imagenet-h5'],
    default='cifar10')

  return parser.parse_args()

def get_dataset(dataset):
  if dataset == 'cifar10':
      import cifar10
      return (32, 32, 3), 10, \
              cifar10.test_input_fn, \
              cifar10.train_input_fn
  elif dataset.endswith('-h5'):
      import filesh5
      filename = "%s/rds/rds-hpc-support/rse/full_%s.h5" % (os.environ['HOME'], dataset[:-3])
      return (64, 64, 3), filesh5.num_classes(filename), \
              filesh5.make_input_fn(filename, "test"), \
              filesh5.make_input_fn(filename, "train")
  elif dataset.endswith('-tfr64'):
      import tfrecords
      filename = "%s/rds/rds-hpc-support/rse/tfrecord_%s_64_64" % (os.environ['HOME'], dataset[:-6])
      return (64, 64, 3), tfrecords.num_classes(filename), \
              tfrecords.make_input_fn(filename, "test"), \
              tfrecords.make_input_fn(filename, "train")
  elif dataset.endswith('-tfr'):
      import tfrecords
      filename = "%s/rds/rds-hpc-support/rse/tfrecord_%s" % (os.environ['HOME'], dataset[:-4])
      return (64, 64, 3), tfrecords.num_classes(filename), \
              tfrecords.make_input_fn(filename, "test"), \
              tfrecords.make_input_fn(filename, "train")
  else:
      basepath = "%s/rds/rds-hpc-support/rse/full_%s" % (os.environ['HOME'], dataset)
      import files
      return (64, 64, 3), files.num_classes("%s/test"%basepath), \
              files.make_input_fn("%s/test"%basepath), \
              files.make_input_fn("%s/train"%basepath) 

def train_and_evaluate(hparams):
  img_shape, num_classes, test_input_fn, train_input_fn = get_dataset(hparams.dataset)  

  model = model_fn(img_shape, num_classes, hparams.learning_rate)

#  model.summary()
  model.load_weights('my_model_weights.h5')
  strategy = {
          'mirror': MirroredStrategy(num_gpus=hparams.num_gpus, prefetch_on_device=True),
          'collective': CollectiveAllReduceStrategy(num_gpus_per_worker=hparams.num_gpus)
          }[hparams.dist]

#  config = tf.estimator.RunConfig(train_distribute=strategy, save_checkpoints_steps=500)
   
  dist_config = tf.contrib.distribute.DistributeConfig(
           train_distribute=strategy,
           eval_distribute=strategy,
           remote_cluster=None
           )
  session_config = tf.ConfigProto(
           inter_op_parallelism_threads=0,
           intra_op_parallelism_threads=0,
           allow_soft_placement=True
           )
  config = tf.estimator.RunConfig(
           save_checkpoints_steps=2000,
           session_config=session_config,
           experimental_distribute=dist_config
           )

  estimator = tf.keras.estimator.model_to_estimator(model,
          model_dir=hparams.job_dir,
          config=config)
  
  train_spec = tf.estimator.TrainSpec(
          input_fn=lambda: train_input_fn(hparams.batch_size),
          max_steps=8000)

  eval_spec = tf.estimator.EvalSpec(
          input_fn=lambda: test_input_fn(hparams.batch_size),
          steps=50,
          start_delay_secs=10,
          throttle_secs=10)

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ == '__main__':
  args = get_args()
  tf.logging.set_verbosity(args.verbosity)

  hparams = hparam.HParams(**args.__dict__)
  train_and_evaluate(hparams)
