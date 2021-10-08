#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script for training multiplane image (MPI) network.
"""
from __future__ import division
import tensorflow as tf

from crossmpi.sequence_data_loader import SequenceDataLoader
from crossmpi.mpi import MPI

# Note that the flags below are a subset of all flags. The remainder (data
# loading relevant) are defined in loader.py.
flags = tf.app.flags
flags.DEFINE_string('checkpoint_dir', 'checkpoints',
                    'Location to save the models.')
flags.DEFINE_string('pretrained_hr_syn_checkpoint_dir', None,
                    'Location of mpi pretrained model.')
flags.DEFINE_string('cameras_glob', 'train/????????????????.txt',
                    'Glob string for training set camera files.')
flags.DEFINE_string('image_dir', 'images',
                    'Path to training image directories.')
flags.DEFINE_string('valid_cameras_glob', 'train/????????????????.txt',
                    'Glob string for training set camera files.')
flags.DEFINE_string('valid_image_dir', 'images',
                    'Path to training image directories.')
flags.DEFINE_string('experiment_name', '', 'Name for the experiment to run.')
flags.DEFINE_string('which_model_predict', 'ASPP_f_psv_f_psv', 'Name for model to predict.')
flags.DEFINE_integer('random_seed', 8964, 'Random seed.')
flags.DEFINE_string(
    'which_loss', 'pixel', 'Which loss to use to compare '
    'rendered and ground truth images. '
    'Can be "pixel" or "VGG".')
flags.DEFINE_float('learning_rate', 0.0002, 'Learning rate')
flags.DEFINE_float('beta1', 0.9, 'beta1 hyperparameter for Adam optimizer.')
flags.DEFINE_integer('max_steps', 10000000, 'Maximum number of training steps.')
flags.DEFINE_integer('summary_freq', 200, 'Logging frequency.')
# flags.DEFINE_integer('summary_freq', 50, 'Logging frequency.')
flags.DEFINE_integer(
    'save_latest_freq', 2000, 'Frequency with which to save the model '
    '(overwrites previous model).')
flags.DEFINE_boolean('continue_train', False,
                     'Continue training from previous checkpoint.')
flags.DEFINE_integer('num_source', 2, 'Number of source images.')
flags.DEFINE_integer(
    'shuffle_seq_length', 4,
    'Length of sequences to be sampled from each video clip. '
    'Each sequence is shuffled, and then the first '
    'num_source + 1 images from the shuffled sequence are '
    'selected as a training instance. Increasing this number '
    'results in more varied baselines in training data.')
flags.DEFINE_float('min_depth', 1, 'Minimum scene depth.')
flags.DEFINE_float('max_depth', 100, 'Maximum scene depth.')
flags.DEFINE_integer('num_psv_planes', 32, 'Number of planes for plane sweep '
                     'volume (PSV).')
flags.DEFINE_string(
    'vgg_model_file', 'imagenet-vgg-verydeep-19.mat',
    'Location of vgg model file used to compute perceptual '
    '(VGG) loss.')
flags.DEFINE_integer('resize_factor', 8, 'The resize factor you need in your stereo cross super resolution')  # -- Mayza
flags.DEFINE_boolean('color_augmentation', False, 'color_augmentation')
flags.DEFINE_boolean('pretrain_hr_syn', False, 'if you want to pretrain the hr_syn part')
flags.DEFINE_boolean('pretrain_lowres', False, 'if you want to pretrain the hr_syn lowres part')

FLAGS = flags.FLAGS

# comment path check lines in


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)  # print INFO to the screen (DEBUG < INFO < WARN < ERROR < FATAL)
  tf.set_random_seed(FLAGS.random_seed)  # graph-level seed (repeat in different session), and there is also op seed
  FLAGS.checkpoint_dir += '/%s/' % FLAGS.experiment_name
  if not tf.gfile.IsDirectory(FLAGS.checkpoint_dir):
    tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

  tf.logging.info('pretrain_hr_syn == %s' % FLAGS.pretrain_hr_syn)
  tf.logging.info('which_model_predict == %s' % FLAGS.which_model_predict)

  # form two data_loader_iterators
  data_loader = SequenceDataLoader(FLAGS.cameras_glob, FLAGS.image_dir, True,
                                   FLAGS.num_source, FLAGS.shuffle_seq_length,
                                   FLAGS.random_seed, FLAGS.resize_factor, FLAGS.color_augmentation)  # -- Mayza
  valid_data_loader = SequenceDataLoader(FLAGS.valid_cameras_glob, FLAGS.valid_image_dir, False,
                                         FLAGS.num_source, FLAGS.shuffle_seq_length,
                                         FLAGS.random_seed, FLAGS.resize_factor, FLAGS.color_augmentation)  # -- Mayza

  train_examples = data_loader.sample_batch()
  valid_examples = valid_data_loader.sample_batch()

  batch_now = data_loader.form_placeholders()


  model = MPI()
  train_op = model.build_train_graph(
      batch_now, FLAGS.min_depth, FLAGS.max_depth, FLAGS.num_psv_planes, FLAGS.which_loss,
      FLAGS.learning_rate, FLAGS.beta1, FLAGS.vgg_model_file, pretrain_hr_syn=FLAGS.pretrain_hr_syn,
      pretrain_lowres=FLAGS.pretrain_lowres, which_model_predict=FLAGS.which_model_predict)

  model.train(train_op, train_examples, valid_examples, FLAGS.checkpoint_dir, FLAGS.continue_train,
              FLAGS.summary_freq, FLAGS.save_latest_freq, FLAGS.max_steps,
              pretrained_hr_syn_checkpoint_dir=FLAGS.pretrained_hr_syn_checkpoint_dir)

  # run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)
  # model.compile(options=run_opts)


if __name__ == '__main__':
  tf.app.run()
