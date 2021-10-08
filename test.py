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

"""Main script for evaluating multiplane image (MPI) network on a test set.
"""
from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
from skimage.measure import compare_psnr, compare_ssim

from crossmpi.mpi import MPI
from crossmpi.sequence_data_loader import SequenceDataLoader
from crossmpi.utils import write_image
from crossmpi.utils import write_intrinsics
from crossmpi.utils import write_pose

# Note that the flags below are a subset of all flags. The remainder (data
# loading relevant) are defined in loader.py.
flags = tf.app.flags
flags.DEFINE_string('model_root', 'models',
                    'Root directory for model checkpoints.')
flags.DEFINE_string('model_name', 'siggraph_model_20180701',
                    'Name of the model to use for inference.')
flags.DEFINE_string('data_split', 'test',
                    'Which split to run ("train" or "test").')
flags.DEFINE_integer('num_runs', 20, 'number of runs')
flags.DEFINE_string('cameras_glob', 'test/????????????????.txt',
                    'Glob string for test set camera files.')
flags.DEFINE_string('image_dir', 'images', 'Path to test image directories.')
flags.DEFINE_integer('random_seed', 8964, 'Random seed')
flags.DEFINE_string('output_root', '/tmp/results',
                    'Root of directory to write results.')
flags.DEFINE_integer('num_source', 2, 'Number of source images.')
flags.DEFINE_integer(
    'shuffle_seq_length', 10,
    'Length of sequences to be sampled from each video clip. '
    'Each sequence is shuffled, and then the first '
    'num_source + 1 images from the shuffled sequence are '
    'selected as a test instance. Increasing this number '
    'results in more varied baselines in training data.')
flags.DEFINE_string('which_model_predict', 'ASPP_f_psv_f_psv',
                    'Name for model to predict.')
flags.DEFINE_float('min_depth', 1, 'Minimum scene depth.')
flags.DEFINE_float('max_depth', 100, 'Maximum scenen depth.')
flags.DEFINE_integer('num_psv_planes', 32, 'Number of planes for plane sweep '
                     'volume (PSV).')
flags.DEFINE_integer('num_mpi_planes', 32, 'Number of MPI planes to predict.')
flags.DEFINE_string(
    'test_outputs', 'rgba_layers_src_images_tgt_image',
    'Which outputs to save. Can concat the following with "_": '
    '[src_images, ref_image, tgt_image, psv, fgbg, poses,'
    ' intrinsics, blend_weights, rgba_layers].')
flags.DEFINE_integer('resize_factor', 8, 'The resize factor you need in your stereo cross super resolution')  # -- Mayza
flags.DEFINE_boolean('pretrain_hr_syn', False, 'if pretrain the hr_syn part')
flags.DEFINE_boolean('save_image', True, 'if save images')
flags.DEFINE_string(
    'vgg_model_file', 'imagenet-vgg-verydeep-19.mat',
    'Location of vgg model file used to compute perceptual '
    '(VGG) loss.')

FLAGS = flags.FLAGS


def main(_):
  assert FLAGS.batch_size == 1, 'Currently, batch_size must be 1 when testing.'

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.reset_default_graph()
  tf.set_random_seed(FLAGS.random_seed)

  data_loader = SequenceDataLoader(FLAGS.cameras_glob, FLAGS.image_dir, False,
                                   FLAGS.num_source, FLAGS.shuffle_seq_length,
                                   FLAGS.random_seed, FLAGS.resize_factor)

  inputs = data_loader.sample_batch()
  model = MPI()
  psv_planes = model.inv_depths(FLAGS.min_depth, FLAGS.max_depth,
                                FLAGS.num_psv_planes)
  if 'vgg' in FLAGS.which_model_predict:
      vgg_to_extract = FLAGS.vgg_model_file
  else:
      vgg_to_extract = None
  outputs = model.attention_ref_sr(
      inputs['src_images'], inputs['ref_image'], inputs['ref_pose'],
      inputs['src_poses'], inputs['intrinsics'], psv_planes, vgg_file=vgg_to_extract, pretrain_hr_syn=FLAGS.pretrain_hr_syn, which_model_predict=FLAGS.which_model_predict)
  out_to_run = dict()
  out_to_run['hr_syn'] = model.deprocess_image(outputs['hr_syn'])
  out_to_run['psv'] = outputs['psv']  # add psv
  out_to_run['attention_map'] = outputs['attention_map']  # add attention
  if 'init_syn' in outputs.keys():
      out_to_run['init_syn'] = model.deprocess_image(outputs['init_syn'])
  if not FLAGS.pretrain_hr_syn:
      out_to_run['sr_out'] = model.deprocess_image(outputs['sr_out'])

  depth_map = tf.expand_dims(tf.argmax(outputs['attention_map'], axis=-1), -1)
  out_to_run['depth_map'] = tf.image.convert_image_dtype(tf.cast(depth_map, tf.float32) /
                                                      tf.cast(tf.reduce_max(depth_map), tf.float32), dtype=tf.uint8)

  init_depth_map = tf.expand_dims(tf.argmax(outputs['init_attention_map'], axis=-1), -1)
  out_to_run['init_depth_map'] = tf.image.convert_image_dtype(tf.cast(init_depth_map, tf.float32) /
                                                              tf.cast(tf.reduce_max(init_depth_map), tf.float32), dtype=tf.uint8)

  saver = tf.train.Saver([var for var in tf.model_variables()])
  ckpt_dir = os.path.join(FLAGS.model_root, FLAGS.model_name)
  ckpt_file = tf.train.latest_checkpoint(ckpt_dir)
  iter_num = int(ckpt_file.split('-')[-1])
  sv = tf.train.Supervisor(logdir=ckpt_dir, saver=None)
  config = tf.ConfigProto()

  with sv.managed_session(config=config) as sess:
    saver.restore(sess, ckpt_file)
    psnr_sum = 0
    ssim_sum = 0
    psnr_max = 0
    ssim_max = 0
    psnr_min = float('inf')
    ssim_min = float('inf')
    psnr_bicubic_sum = 0
    ssim_bicubic_sum = 0
    time_sum = 0
    num = 0
    metrics = dict()
    for run in range(FLAGS.num_runs):
        # nun_runs > sequences in test data
        try:
            tf.logging.info('Progress: %d/%d' % (run, FLAGS.num_runs))
            start = time.time()
            ins, outs = sess.run([inputs, out_to_run])
            end = time.time()
            execution_time = end-start

            # Output directory name: [scene]_[1st src file]_[2nd src file]_[tgt file].
            dirname = ins['ref_name'][0].split('/')[0]
            for i in range(FLAGS.num_source):
                dirname += '_%s' % (
                    os.path.basename(
                        ins['src_timestamps'][0][i]).split('.')[0].split('_')[-1])
            dirname += '_%s' % (
                os.path.basename(
                    ins['tgt_timestamp'][0]).split('.')[0].split('_')[-1])

            # evaluation
            if FLAGS.pretrain_hr_syn:
                pred_image = outs['hr_syn'][0]
            else:
                pred_image = outs['sr_out'][0]
            lr_up_image = ins['ref_image'][0] * 255.0
            tgt_image = ins['tgt_image'][0] * 255.0

            pred_image = np.clip(pred_image, 0, 255).astype('uint8')
            lr_up_image = np.clip(lr_up_image, 0, 255).astype('uint8')
            tgt_image = np.clip(tgt_image, 0, 255).astype('uint8')

            psnr = compare_psnr(tgt_image, pred_image)
            ssim = compare_ssim(tgt_image, pred_image, multichannel=True)
            if np.sum(np.abs(tgt_image-lr_up_image)) < 1e-5:
                continue
            psnr_bicubic = compare_psnr(tgt_image, lr_up_image)
            ssim_bicubic = compare_ssim(tgt_image, lr_up_image, multichannel=True)

            # record to a dict()
            metrics[dirname] = [psnr, ssim, execution_time]
            tf.logging.info('psnr = %f, ssim = %f and takes: %f s.' % (psnr, ssim, execution_time))
            psnr_sum = psnr_sum + psnr
            ssim_sum = ssim_sum + ssim
            if psnr > psnr_max:
                psnr_max = psnr
            if ssim > ssim_max:
                ssim_max = ssim
            if psnr < psnr_min:
                psnr_min = psnr
            if ssim < ssim_min:
                ssim_min = ssim
            psnr_bicubic_sum = psnr_bicubic_sum + psnr_bicubic
            ssim_bicubic_sum = ssim_bicubic_sum + ssim_bicubic
            if run > 0:
                time_sum = time_sum + execution_time
            num = num + 1  # how many runs

            if FLAGS.save_image:
                output_dir = os.path.join(FLAGS.output_root, FLAGS.model_name,
                                          FLAGS.data_split + '_iter%d' % iter_num, dirname)
                if not tf.gfile.IsDirectory(output_dir):
                    tf.gfile.MakeDirs(output_dir)

                with tf.gfile.GFile(output_dir + '/metric.txt', 'w') as fh:
                    fh.write('(psnr, ssim, time) = (%f, %f, %f)' % (psnr, ssim, execution_time))

                # Write results to disk.
                if 'intrinsics' in FLAGS.test_outputs:
                    with open(output_dir + '/intrinsics.txt', 'w') as fh:
                        write_intrinsics(fh, ins['intrinsics'][0])

                if 'src_images' in FLAGS.test_outputs:
                    for i in range(FLAGS.num_source):
                        timestamp = ins['src_timestamps'][0][i]
                        write_image(output_dir + '/src_image_%d_%s.png' % (i, timestamp),
                                    ins['src_images'][0, :, :, i * 3:(i + 1) * 3] * 255.0)
                        if 'poses' in FLAGS.test_outputs:
                            write_pose(output_dir + '/src_pose_%d.txt' % i,
                                       ins['src_poses'][0, i])

                if 'tgt_image' in FLAGS.test_outputs:
                    timestamp = ins['tgt_timestamp'][0]
                    write_image(output_dir + '/tgt_image_%s.png' % timestamp,
                                ins['tgt_image'][0] * 255.0)
                    write_image(output_dir + '/syn_image_%s.png' % timestamp,
                                outs['hr_syn'][0])
                    if not FLAGS.pretrain_hr_syn:
                        write_image(output_dir + '/output_image_%s.png' % timestamp,
                                    outs['sr_out'][0])
                    write_image(output_dir + '/depth_map_%s.png' % timestamp,
                                outs['depth_map'][0, ..., 0])
                    write_image(output_dir + '/init_depth_map_%s.png' % timestamp,
                                outs['init_depth_map'][0, ..., 0])
                    if 'init_syn' in outs.keys():
                        write_image(output_dir + '/init_syn_%s.png' % timestamp,
                                    outs['init_syn'][0])
                    write_image(output_dir + '/lr_image_%s.png' % timestamp,
                                ins['lr_image'][0] * 255.0)
                    if 'poses' in FLAGS.test_outputs:
                        write_pose(output_dir + '/tgt_pose.txt', ins['tgt_pose'][0])

                if 'attention_map' in FLAGS.test_outputs:
                    for i in range(FLAGS.num_psv_planes):
                        alpha_img = outs['attention_map'][0, :, :, i] * 255.0
                        rgb_img = (outs['psv'][0, :, :, i * 3:(i + 1) * 3] + 1.) / 2. * 255
                        write_image(output_dir + '/mpi_alpha_%.2d.png' % i, alpha_img)
                        write_image(output_dir + '/mpi_rgb_%.2d.png' % i, rgb_img)

                if 'ref_image' in FLAGS.test_outputs:
                    fname = os.path.basename(ins['ref_name'][0])
                    write_image(output_dir + '/ref_image_%s.png' % fname, ins['ref_image'][0] * 255.0)
                    write_pose(output_dir + '/ref_pose.txt', ins['ref_pose'][0])

        except Exception:
            psnr_avg = psnr_sum / num
            ssim_avg = ssim_sum / num
            psnr_bicubic_avg = psnr_bicubic_sum / num
            ssim_bicubic_avg = ssim_bicubic_sum / num
            time_avg = time_sum / (num - 1)  # except the first one
            tf.logging.info('metrics dict length = %d', len(metrics))
            np.save(os.path.join(FLAGS.output_root, FLAGS.model_name, FLAGS.data_split  + '_iter%d' % iter_num)
                                + '_metrics.npy', metrics)
            tf.logging.info('psnr_avg = %f, ssim_avg = %f, time_avg = %f (except the first one), for %d pairs.' % (psnr_avg, ssim_avg, time_avg, num))
            with tf.gfile.GFile(os.path.join(FLAGS.output_root, FLAGS.model_name, FLAGS.data_split  + '_iter%d' % iter_num)
                                + '_metric.txt', 'w') as fh:
                fh.write('total_pairs = %d\n' % num)
                fh.write('(psnr_avg, ssim_avg, time_avg) = (%f, %f, %f)\n' % (psnr_avg, ssim_avg, time_avg))
                fh.write('(psnr_max, ssim_max, psnr_min, ssim_min) = (%f, %f, %f, %f)\n' % (psnr_max, ssim_max, psnr_min, ssim_min))
                fh.write('(psnr_bicubic_avg, ssim_bicubic_avg, time_avg) = (%f, %f, %f)' % (psnr_bicubic_avg, ssim_bicubic_avg, time_avg))

            exit(0)


if __name__ == '__main__':
  tf.app.run()
