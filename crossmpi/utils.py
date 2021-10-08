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
"""Collection of utility functions."""

import math
import numpy as np
from PIL import Image
import PIL.Image as pil
from scipy import signal
import tensorflow as tf


def write_image(filename, image):
  """Save image to disk."""
  byte_image = np.clip(image, 0, 255).astype('uint8')
  image_pil = pil.fromarray(byte_image)
  with tf.gfile.GFile(filename, 'w') as fh:
    image_pil.save(fh)


def write_pose(filename, pose):
  with tf.gfile.GFile(filename, 'w') as fh:
    for i in range(3):
      for j in range(4):
        fh.write('%f ' % (pose[i, j]))


def write_intrinsics(fh, intrinsics):
  fh.write('%f ' % intrinsics[0, 0])
  fh.write('%f ' % intrinsics[1, 1])
  fh.write('%f ' % intrinsics[0, 2])
  fh.write('%f ' % intrinsics[1, 2])
  fh.write('%f ' % intrinsics[3, 0])
  fh.write('%f ' % intrinsics[4, 1])
  fh.write('%f ' % intrinsics[3, 2])
  fh.write('%f ' % intrinsics[4, 2])


def build_matrix(elements):
  """Stacks elements along two axes to make a tensor of matrices.

  Args:
    elements: [n, m] matrix of tensors, each with shape [...].

  Returns:
    [..., n, m] tensor of matrices, resulting from concatenating
      the individual tensors.
  """
  rows = [tf.stack(row_elements, axis=-1) for row_elements in elements]
  return tf.stack(rows, axis=-2)


# downsampling module -- Mayza
def img_pil_downsampling(img, resize_factor):
  batch_size, img_height, img_width, _ = img.shape
  img_op = img.copy()
  img_down = None
  for i in range(batch_size):
    img_temp = img_op[i, ...]
    img_temp_pil = Image.fromarray(((img_temp + 1) / 2.0 * 255.0).astype('uint8'))  # (the network input is -1~1)
    img_downsampled = img_temp_pil.resize((int(img_width / resize_factor),
                                           int(img_height / resize_factor)),
                                          resample=Image.BICUBIC)
    img_downsampled = (np.asarray(img_downsampled, dtype=np.float32) / 255.0) * 2 - 1
    if i == 0:
      img_down = img_downsampled[np.newaxis, ...]
    else:
      img_down = np.concatenate((img_down, img_downsampled), axis=0)
  return img_down

