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
"""Script to generate a multiplane image (MPI) from an image pair related by a
3D translation."""

from __future__ import division
import os
import sys
import tensorflow as tf
import glob
import numpy as np
from PIL import Image

from crossmpi.mpi import MPI
from crossmpi.utils import build_matrix
from crossmpi.utils import write_image
from crossmpi.utils import write_intrinsics
from crossmpi.utils import write_pose

flags = tf.app.flags

# Input flags
flags.DEFINE_string('image_dir', '', 'The image dir of giga images.')
flags.DEFINE_integer('frame_start', 0, 'Which frame id to start the processing')
flags.DEFINE_integer('frame_len', 100, 'Which frame id to end the processing')
flags.DEFINE_integer('image_height', 752, 'Image height in pixels.')
flags.DEFINE_integer('image_width', 1008, 'Image width in pixels.')

flags.DEFINE_float('fx', 0.5, 'Focal length as a fraction of image width.')
flags.DEFINE_float('fy', 0.5, 'Focal length as a fraction of image height.')
flags.DEFINE_float('cx', 0.5, 'Principle point cx.')
flags.DEFINE_float('cy', 0.5, 'Principle point cy.')

flags.DEFINE_float('min_depth', 1, 'Minimum scene depth.')
flags.DEFINE_float('max_depth', 100, 'Maximum scene depth.')

flags.DEFINE_string('pose1', '',
                    ('Camera pose for first image (if not identity).'
                     ' Twelve space- or comma-separated floats, forming a 3x4'
                     ' matrix in row-major order.'))
flags.DEFINE_string('pose2', '',
                    ('Pose for second image (if not identity).'
                     ' Twelve space- or comma-separated floats, forming a 3x4'
                     ' matrix in row-major order. If pose2 is specified, then'
                     ' xoffset/yoffset/zoffset flags will be used for rendering'
                     ' output views only.'))
# Output flags
flags.DEFINE_string('output_dir', '/tmp/', 'Directory to write MPI output.')
flags.DEFINE_string('test_outputs', 'src_images',
                    ('Which outputs to save. Can concat the following with _'
                     ' [src_images, ref_image, psv, fgbg, poses,'
                     ' intrinsics, blend_weights, rgba_layers]'))

# Rendering images.
flags.DEFINE_boolean('render', True,
                     'Render output images at multiples of input offset.')

# Model flags. Defaults are the model described in the SIGGRAPH 2018 paper.  See
# README for more details.
flags.DEFINE_string('model_root', 'models/',
                    'Root directory for model checkpoints.')
flags.DEFINE_string('model_name', 'siggraph_model_20180701',
                    'Name of the model to use for inference.')
flags.DEFINE_string('which_color_pred', 'bg',
                    'Color output format: [alpha_only,single,bg,fgbg,all].')
flags.DEFINE_integer('num_psv_planes', 32, 'Number of planes for PSV.')
flags.DEFINE_integer('num_mpi_planes', 32, 'Number of MPI planes to infer.')
flags.DEFINE_boolean('deconvolution', True,
                     'Use deconvolution or not.')
flags.DEFINE_string('which_model_predict', 'ASPP_f_psv_f_psv',
                    'Name for model to predict.')

FLAGS = flags.FLAGS

def shift_image(image, x, y):
  """Shift an image x pixels right and y pixels down, filling with zeros."""
  shape = tf.shape(image)
  height = shape[0]
  width = shape[1]
  x = int(round(x))
  y = int(round(y))
  dtype = image.dtype
  if x > 0:
    image = tf.concat(
        [tf.zeros([height, x, 3], dtype=dtype), image[:, :(width - x)]], axis=1)
  elif x < 0:
    image = tf.concat(
        [image[:, -x:], tf.zeros([height, -x, 3], dtype=dtype)], axis=1)
  if y > 0:
    image = tf.concat(
        [tf.zeros([y, width, 3], dtype=dtype), image[:(height - y), :]], axis=0)
  elif y < 0:
    image = tf.concat(
        [image[-y:, :], tf.zeros([-y, width, 3], dtype=dtype)], axis=0)
  return image


def crop_to_multiple(image, size):
  """Crop image to a multiple of size in height and width."""
  # Compute how much we need to remove.
  shape = tf.shape(image)
  height = shape[0]
  width = shape[1]
  new_width = width - (width % size)
  new_height = height - (height % size)
  # Crop amounts. Extra pixel goes on the left side.
  left = (width % size) // 2
  right = new_width + left
  top = (height % size) // 2
  bottom = new_height + top
  return image[top:bottom, left:right]


def crop_to_size(image, width, height):
  """Crop image to specified size."""
  shape = tf.shape(image)
  # crop_to_multiple puts the extra pixel on the left, so here
  # we make sure to remove the extra pixel from the left.
  left = (shape[1] - width + 1) // 2
  top = (shape[0] - height + 1) // 2
  right = left + width
  bottom = top + height
  return image[top:bottom, left:right]


def load_image_and_resize(f):
  """Load an image, pad, and shift it."""
  contents = tf.read_file(f)
  raw = tf.image.decode_image(contents)
  image = tf.image.convert_image_dtype(raw, tf.float32)
  resized = tf.squeeze(
      tf.image.resize_area(tf.expand_dims(image, axis=0), [FLAGS.image_height, FLAGS.image_width]),
      axis=0)
  resized.set_shape([FLAGS.image_height, FLAGS.image_width, 3])  # RGB images have 3 channels.
  return resized


def pose_from_flag(flag):
  if flag:
    values = [float(x) for x in flag.replace(',', ' ').split()]
    assert len(values) == 12
    return [values[0:4], values[4:8], values[8:12], [0.0, 0.0, 0.0, 1.0]]
  else:
    return [[1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]]


def get_inputs(global_filename, local_filename):
  """Get images, poses and intrinsics in required format."""
  inputs = {}
  image1 = load_image_and_resize(global_filename)
  image2 = load_image_and_resize(local_filename)

  # Add batch dimension (size 1).
  image1 = image1[tf.newaxis, ...]
  image2 = image2[tf.newaxis, ...]

  pose_one = pose_from_flag(FLAGS.pose1)
  pose_two = pose_from_flag(FLAGS.pose2)

  pose_one = build_matrix(pose_one)[tf.newaxis, ...]
  pose_two = build_matrix(pose_two)[tf.newaxis, ...]

  fx = tf.multiply(tf.to_float(FLAGS.image_width), FLAGS.fx)
  fy = tf.multiply(tf.to_float(FLAGS.image_height), FLAGS.fy)
  cx = tf.multiply(tf.to_float(FLAGS.image_width), FLAGS.cx)
  cy = tf.multiply(tf.to_float(FLAGS.image_height), FLAGS.cy)
  intrinsics = build_matrix([[fx, 0.0, cx], [0.0, fy, cy],
                             [0.0, 0.0, 1.0]])[tf.newaxis, ...]

  inputs['ref_image'] = image1
  inputs['ref_pose'] = pose_one
  inputs['src_images'] = tf.concat([image1, image2], axis=-1)
  inputs['src_poses'] = tf.stack([pose_one, pose_two], axis=1)
  inputs['intrinsics'] = tf.concat([intrinsics, intrinsics], axis=1)
  return inputs


def main(_):
  # Set up the inputs.
  # Maybe we don't need to pad the inputs
  image_dir = FLAGS.image_dir
  assert tf.gfile.IsDirectory(image_dir)  # Ensure the provided path is valid.
  assert tf.gfile.ListDirectory(image_dir) > 0  # Ensure that some data exists.
  assert FLAGS.frame_len > 0

  tf.reset_default_graph()

  global_filenames = glob.glob(os.path.join(image_dir, '*global.png'))
  local_filenames = glob.glob(os.path.join(image_dir, '*local.png'))

  global_filenames.sort()
  local_filenames.sort()

  global_filenames = tf.constant(global_filenames[FLAGS.frame_start: FLAGS.frame_start + FLAGS.frame_len])
  local_filenames = tf.constant(local_filenames[FLAGS.frame_start: FLAGS.frame_start + FLAGS.frame_len])

  dataset = tf.data.Dataset.from_tensor_slices((global_filenames, local_filenames))
  dataset = dataset.map(get_inputs)

  iterator = dataset.make_one_shot_iterator()
  inputs = iterator.get_next()

  batch = 1
  channels = 3
  mpi_height = FLAGS.image_height
  mpi_width = FLAGS.image_width

  print('MPI size: width=%d, height=%d' % (mpi_width, mpi_height))
  inputs['ref_image'].set_shape([batch, mpi_height, mpi_width, channels])
  inputs['src_images'].set_shape([batch, mpi_height, mpi_width, channels * 2])
  # Build the MPI.
  model = MPI()
  psv_planes = model.inv_depths(FLAGS.min_depth, FLAGS.max_depth,
                                FLAGS.num_psv_planes)
  print('Inferring MPI...')

  outputs = model.attention_ref_sr(
      inputs['src_images'], inputs['ref_image'], inputs['ref_pose'],
      inputs['src_poses'], inputs['intrinsics'], psv_planes, which_model_predict=FLAGS.which_model_predict)

  out_to_run = dict()
  out_to_run['hr_syn'] = model.deprocess_image(outputs['hr_syn'])
  out_to_run['psv'] = outputs['psv']  # add psv
  out_to_run['attention_map'] = outputs['attention_map']  # add attention
  if 'init_syn' in outputs.keys():
      out_to_run['init_syn'] = model.deprocess_image(outputs['init_syn'])
  out_to_run['sr_out'] = model.deprocess_image(outputs['sr_out'])

  depth_map = tf.expand_dims(tf.argmax(outputs['attention_map'], axis=-1), -1)
  out_to_run['depth_map'] = tf.image.convert_image_dtype(tf.cast(depth_map, tf.float32) /
                                                         tf.cast(tf.reduce_max(depth_map), tf.float32), dtype=tf.uint8)

  init_depth_map = tf.expand_dims(tf.argmax(outputs['init_attention_map'], axis=-1), -1)
  out_to_run['init_depth_map'] = tf.image.convert_image_dtype(tf.cast(init_depth_map, tf.float32) /
                                                              tf.cast(tf.reduce_max(init_depth_map), tf.float32),
                                                              dtype=tf.uint8)
  saver = tf.train.Saver([var for var in tf.model_variables()])
  ckpt_dir = os.path.join(FLAGS.model_root, FLAGS.model_name)
  ckpt_file = tf.train.latest_checkpoint(ckpt_dir)
  sv = tf.train.Supervisor(logdir=ckpt_dir, saver=None)
  config = tf.ConfigProto()

  config.gpu_options.allow_growth = True  # control the usage of GPU

  with sv.managed_session(config=config) as sess:
      saver.restore(sess, ckpt_file)
      for frame_id in range(FLAGS.frame_start, FLAGS.frame_start + FLAGS.frame_len):

          print("Start calculating frame: No.%03d/%03d" % (frame_id, FLAGS.frame_start + FLAGS.frame_len))
          ins, outs = sess.run([inputs, out_to_run])

          output_dir = os.path.join(FLAGS.output_dir, FLAGS.model_name, '%03d' % frame_id)
          if not tf.gfile.IsDirectory(output_dir):
              tf.gfile.MakeDirs(output_dir)

          print('Saving results to %s' % output_dir)

          # Write results to disk.
          write_image(output_dir + '/syn_image.png', outs['hr_syn'][0])
          write_image(output_dir + '/output_image.png', outs['sr_out'][0])
          write_image(output_dir + '/depth_map.png', outs['depth_map'][0, ..., 0])
          write_image(output_dir + '/init_depth_map.png', outs['init_depth_map'][0, ..., 0])
          if 'init_syn' in outs.keys():
              write_image(output_dir + '/init_syn.png', outs['init_syn'][0])
          # write_image(output_dir + '/lr_image.png', ins['lr_image'][0] * 255.0)
          if 'poses' in FLAGS.test_outputs:
              write_pose(output_dir + '/tgt_pose.txt', ins['tgt_pose'][0])
          if 'intrinsics' in FLAGS.test_outputs:
              with open(output_dir + '/intrinsics.txt', 'w') as fh:
                  write_intrinsics(fh, ins['intrinsics'][0])
          if 'src_images' in FLAGS.test_outputs:
              for i in range(2):
                  write_image(output_dir + '/src_image_%d.png' % i,
                              ins['src_images'][0, :, :, i * 3:(i + 1) * 3] * 255.0)
                  if 'poses' in FLAGS.test_outputs:
                      write_pose(output_dir + '/src_pose_%d.txt' % i, ins['src_poses'][0, i])
          if 'psv' in FLAGS.test_outputs:
              for j in range(FLAGS.num_psv_planes):
                  plane_img = (outs['psv'][0, :, :, j * 3:(j + 1) * 3] + 1.) / 2. * 255
                  write_image(output_dir + '/psv_plane_%.3d.png' % j, plane_img)
          if 'rgba_layers' in FLAGS.test_outputs:
              for i in range(FLAGS.num_mpi_planes):
                  alpha_img = outs['rgba_layers'][0, :, :, i, 3] * 255.0
                  rgb_img = (outs['rgba_layers'][0, :, :, i, :3] + 1.) / 2. * 255
                  write_image(output_dir + '/mpi_alpha_%.2d.png' % i, alpha_img)
                  write_image(output_dir + '/mpi_rgb_%.2d.png' % i, rgb_img)

          with open(output_dir + '/README', 'w') as fh:
              fh.write(
                  'This directory was generated by crossmpi_from_images. Command-line:\n\n')
              fh.write('%s \\\n' % sys.argv[0])
              for arg in sys.argv[1:-1]:
                  fh.write('  %s \\\n' % arg)
              fh.write('  %s\n' % sys.argv[-1])

          print('Done.')


if __name__ == '__main__':
  tf.app.run()
