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
"""Class definition of the data loader.
"""
from __future__ import division

import os.path
import tensorflow as tf
from tensorflow import flags
import datasets
import loader
import cv2  # -- Mayza
import numpy as np
from PIL import Image

FLAGS = flags.FLAGS


class SequenceDataLoader(object):
  """Loader for video sequence data."""

  def __init__(self,
               cameras_glob='train/????????????????.txt',
               image_dir='images',
               training=True,
               num_source=2,
               shuffle_seq_length=10,
               random_seed=8964,
               resize_factor=2.0,  # --Mayza
               color_aug_flag = False,
               map_function=None):
    self.num_source = num_source
    self.random_seed = random_seed
    self.shuffle_seq_length = shuffle_seq_length
    self.batch_size = FLAGS.batch_size
    self.image_height = FLAGS.image_height
    self.image_width = FLAGS.image_width
    self.resize_factor = resize_factor  # -- Mayza
    self.color_aug_flag = color_aug_flag

    self.datasets = loader.create_from_flags(
        cameras_glob=cameras_glob,
        image_dir=image_dir,
        training=training,
        map_function=map_function)
    # return a object Loader() from loader.py

  def set_shapes(self, examples):
    """Set static shapes of the mini-batch of examples.

    Args:
      examples: a batch of examples
    Returns:
      examples with correct static shapes
    """
    b = self.batch_size
    h = self.image_height
    w = self.image_width
    s = self.num_source
    r = self.resize_factor
    examples['tgt_image'].set_shape([b, h, w, 3])
    examples['lr_image'].set_shape([b, int(h/r), int(w/r), 3])
    examples['ref_image'].set_shape([b, h, w, 3])
    examples['src_images'].set_shape([b, h, w, 3 * s])
    examples['tgt_pose'].set_shape([b, 4, 4])
    examples['ref_pose'].set_shape([b, 4, 4])
    examples['src_poses'].set_shape([b, s, 4, 4])
    examples['intrinsics'].set_shape([b, 6, 3])
    return examples

  def form_placeholders(self):
    """Set static shapes of the mini-batch of examples.

    Args:
      examples: a batch of examples
    Returns:
      examples with correct static shapes
    """
    b = self.batch_size
    h = self.image_height
    w = self.image_width
    s = self.num_source
    r = self.resize_factor
    examples = {}
    examples['tgt_image'] = tf.placeholder(tf.float32, shape=[b, h, w, 3], name='input_tgt_image')
    examples['lr_image'] = tf.placeholder(tf.float32, shape=[b, int(h/r), int(w/r), 3], name='input_lr_image')
    examples['ref_image'] = tf.placeholder(tf.float32, shape=[b, h, w, 3], name='input_ref_image')
    examples['src_images'] = tf.placeholder(tf.float32, shape=[b, h, w, 3 * s], name='input_src_images')
    examples['tgt_pose'] = tf.placeholder(tf.float32, shape=[b, 4, 4], name='input_tgt_pose')
    examples['ref_pose'] = tf.placeholder(tf.float32, shape=[b, 4, 4], name='input_ref_pose')
    examples['src_poses'] = tf.placeholder(tf.float32, shape=[b, s, 4, 4], name='input_src_poses')
    examples['intrinsics'] = tf.placeholder(tf.float32, shape=[b, 6, 3], name='input_intrinsics')
    return examples

  def sample_batch(self):
    """Samples a batch of examples for training / testing.

    Returns:
      A batch of examples.
    """
    example = self.datasets.sequences.map(self.format_for_mpi())
    iterator = example.make_one_shot_iterator()
    # iterator = example.make_initializable_iterator() # create the iterator
    # iterator = tf.data.Iterator.from_structure(example.output_types, example.output_shapes)  # create a common iterator
    return self.set_shapes(iterator.get_next())  # , iterator

  def format_for_mpi(self):
    """Format the sampled sequence for MPI training/inference.
    """

    def make_intrinsics_matrix(fx, fy, cx, cy):
      # Assumes batch input.
      batch_size = fx.get_shape().as_list()[0]
      zeros = tf.zeros_like(fx)
      r1 = tf.stack([fx, zeros, cx], axis=1)
      r2 = tf.stack([zeros, fy, cy], axis=1)
      r3 = tf.constant([0., 0., 1.], shape=[1, 3])
      r3 = tf.tile(r3, [batch_size, 1])
      intrinsics = tf.stack([r1, r2, r3], axis=1)
      return intrinsics

    def format_sequence(sequence):
      # resize module -- Mayza
      def img_processing_lr_syn(img):
        img_op = img.copy()
        for i in range(self.batch_size):
          img_temp = img_op[i, ...]
          img_temp_pil = Image.fromarray((img_temp * 255.0).astype('uint8'))
          img_downsampled = img_temp_pil.resize((int(self.image_width/self.resize_factor),
                                                 int(self.image_height/self.resize_factor)),
                                                resample=Image.BICUBIC)
          img_upsampled = img_downsampled.resize((self.image_width, self.image_height), resample=Image.BICUBIC)
          img_op[i, ...] = np.asarray(img_upsampled, dtype=np.float32) / 255.0
        return img_op

      # downsampling module -- Mayza
      def img_pil_downsampling(img):
        img_op = img.copy()
        img_down = None
        for i in range(self.batch_size):
          img_temp = img_op[i, ...]
          img_temp_pil = Image.fromarray((img_temp * 255.0).astype('uint8'))
          img_downsampled = img_temp_pil.resize((int(self.image_width / self.resize_factor),
                                                 int(self.image_height / self.resize_factor)),
                                                resample=Image.BICUBIC)
          img_downsampled = np.asarray(img_downsampled, dtype=np.float32) / 255.0
          if i == 0:
            img_down = img_downsampled[np.newaxis, ...]
          else:
            img_down = np.concatenate((img_down, img_downsampled), axis=0)
        return img_down

      def color_augmentation_function(image):
        image = tf.image.random_brightness(image, max_delta=0.5, seed=self.random_seed)
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8, seed=self.random_seed)
        image = tf.image.random_hue(image, max_delta=0.3, seed=self.random_seed)
        image = tf.image.random_saturation(image, lower=0.2, upper=1.8, seed=self.random_seed)
        return image

      tgt_idx = tf.random_uniform(
          [],
          maxval=self.shuffle_seq_length,
          dtype=tf.int32,
          seed=self.random_seed)
      shuffled_inds = tf.random_shuffle(
          tf.range(self.shuffle_seq_length), seed=self.random_seed)
      src_inds = shuffled_inds[:self.num_source]

      ref_idx = src_inds[0]
      tgt_idx = ref_idx  # for super resolution lr source(1) ref(0) image -- Mayza

      images = sequence.image
      images.set_shape([
          self.batch_size, self.shuffle_seq_length, self.image_height,
          self.image_width, 3
      ])
      poses = sequence.pose
      poses.set_shape([self.batch_size, self.shuffle_seq_length, 3, 4])
      intrinsics = sequence.intrinsics
      intrinsics.set_shape([self.batch_size, self.shuffle_seq_length, 4])

      tgt_image = images[:, tgt_idx]
      ref_image = images[:, ref_idx]  # for super resolution lr source(1) ref(0) image -- Mayza

      # resize the reference image
      lr_image = tf.py_func(img_pil_downsampling, [ref_image], tf.float32)
      ref_image = tf.py_func(img_processing_lr_syn, [ref_image], tf.float32)

      src_images = tf.gather(images, src_inds, axis=1)
      # Reshape src_images into [batch, height, width, 3*#sources]
      src_images = tf.transpose(src_images, [0, 2, 3, 1, 4])
      src_images = tf.reshape(src_images, [self.batch_size, self.image_height, self.image_width, -1])

      # make sure the ref image is resized  -- Mayza
      ref_in_src_image = src_images[..., :3]
      ref_in_src_image = tf.py_func(img_processing_lr_syn, [ref_in_src_image], tf.float32)

      if self.color_aug_flag:
        tgt_image = color_augmentation_function(tgt_image)
        ref_image = color_augmentation_function(ref_image)
        ref_in_src_image = color_augmentation_function(ref_in_src_image)

      src_images = tf.concat([ref_in_src_image, src_images[..., 3:]], axis=3)  # all been resized

      # Make the pose matrix homogeneous.
      filler = tf.constant(
          [0., 0., 0., 1.], dtype=tf.float32, shape=[1, 1, 1, 4])
      filler = tf.tile(filler, [self.batch_size, self.shuffle_seq_length, 1, 1])
      poses_h = tf.concat([poses, filler], axis=2)
      ref_pose = poses_h[:, ref_idx]
      tgt_pose = poses_h[:, tgt_idx]
      src_poses = tf.gather(poses_h, src_inds, axis=1)
      intrinsics = intrinsics[:, ref_idx]
      intrinsics = make_intrinsics_matrix(intrinsics[:, 0] * self.image_width,
                                          intrinsics[:, 1] * self.image_height,
                                          intrinsics[:, 2] * self.image_width,
                                          intrinsics[:, 3] * self.image_height)
      src_timestamps = tf.gather(sequence.timestamp, src_inds, axis=1)
      ref_timestamp = tf.gather(sequence.timestamp, ref_idx, axis=1)
      tgt_timestamp = tf.gather(sequence.timestamp, tgt_idx, axis=1)

      # Put everything into a dictionary.
      instance = {}
      instance['tgt_image'] = tgt_image  # gt
      instance['lr_image'] = lr_image  # low-res input
      instance['ref_image'] = ref_image  # low-res input upsampled
      instance['src_images'] = src_images  # low-res input upsampled and hr reference
      instance['tgt_pose'] = tgt_pose  # input pose
      instance['src_poses'] = src_poses  # [input pose, reference pose]
      instance['intrinsics'] = tf.concat([intrinsics, intrinsics], axis=1)
      instance['ref_pose'] = ref_pose  # input pose
      instance['ref_name'] = sequence.id
      instance['src_timestamps'] = src_timestamps
      instance['ref_timestamp'] = ref_timestamp
      instance['tgt_timestamp'] = tgt_timestamp
      return self.set_shapes(instance)

    return format_sequence
