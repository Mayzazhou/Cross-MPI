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
"""Functions for learning multiplane images (MPIs).
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
import tensorflow as tf
import geometry.projector as pj
from third_party.vgg import build_vgg19
from nets import f_extract_net, att_hr_syn_net, guided_att_hr_syn_net, psv_guided_att_hr_syn_net, psv_guided_att_decoder, res_fuse_net
from utils import img_pil_downsampling

class MPI(object):
  """Class definition for MPI learning module.
  """

  def __init__(self):
    pass

  def attention_ref_sr(self,
                raw_src_images,
                raw_ref_image,
                ref_pose,
                src_poses,
                intrinsics,
                psv_planes,
                pretrain_hr_syn=False,
                vgg_file=None,
                which_model_predict='ASPP_f_psv_f_psv'):
    """Construct the MPI inference graph.

    Args:
      raw_src_images: stack of source images [batch, height, width, 3*#source]
      raw_ref_image: reference image [batch, height, width, 3] or raw_lr_image [batch, height/8, width/8, 3]
      ref_pose: reference frame pose (world to camera) [batch, 4, 4]
      src_poses: source frame poses (world to camera) [batch, #source, 4, 4]
      intrinsics: camera intrinsics [batch, 6, 3]
      psv_planes: list of depth of plane sweep volume (PSV) planes
      pretrain_hr_syn: pretrain hr syn or not
      vgg_file: if use vgg to extract features
      which_model_predict: which model we use

    Returns:
      outputs: a collection of output tensors.
    """
    batch_size, img_height, img_width, _ = raw_src_images.get_shape().as_list()
    with tf.name_scope('preprocessing'):
      src_images = self.preprocess_image(raw_src_images)
      ref_image = self.preprocess_image(raw_ref_image)

    with tf.name_scope('format_network_input'):
      # Note: we assume the first src image/pose is the reference.
      # So the ref_image is the real input of the network  -- Mayza
      net_input = self.format_network_input(ref_image, src_images[:, :, :, 3:],
                                              ref_pose, src_poses[:, 1:],
                                              psv_planes, intrinsics)
      # mask synthesis
      """be careful! these lines have some problems......"""
      # psv_max = tf.reduce_max(net_input[:, :, :, 3:], axis=-1, keepdims=True)  # [b, h, w, 1]
      # psv_mask = tf.greater(psv_max, -1 + 1e-5)  # elements > -1 is kept, others set as -1 by tf.where
      # psv_mask = tf.tile(psv_mask, (1, 1, 1, 3))  # [b, h, w, 3], bool

      # lr psv synthesis
      if 'F_PSV_LR' in which_model_predict:
        tf.logging.info('psv_lr synthesis...')
        net_input_lr_syn = self.psvs_lr_syn(net_input, len(psv_planes))

      if 'lowres' in which_model_predict:
        net_input_lowres = self.psvs_lowres_syn(net_input, len(psv_planes))

    with tf.name_scope('hr_synthesis_prediction'):
      if 'denseASPP' in which_model_predict:
        dense_res_aspp = True
        tf.logging.info('Use denseASPP!!!!')
      else:
        dense_res_aspp = False
      # extract lr features
      if 'lowres' in which_model_predict:
        f_lr = f_extract_net(net_input_lowres[..., :3], dense_res_aspp=dense_res_aspp)
      else:
        f_lr = f_extract_net(ref_image, vgg_filepath=vgg_file, dense_res_aspp=dense_res_aspp)  # f_lr: [b, h, w, c] and f_extract_net should be 'tf.AUTO_REUSE'

      # extract psv features
      if 'f_psv' in which_model_predict:
        tf.logging.info('Generating f_psv!!')
        f_psv = []
        for i in range(len(psv_planes)):
          f_map = f_extract_net(net_input[..., (i + 1) * 3: (i + 2) * 3], vgg_filepath=vgg_file, dense_res_aspp=dense_res_aspp)  # [b, h, w, c]
          f_psv.append(f_map)
        f_psv = tf.stack(f_psv, axis=-1)  # [b, h, w, c, n]

      # extract psv_lr features
      if 'lowres' in which_model_predict:
        tf.logging.info('Generating f_lowres_psv!!')
        f_lowres_psv = []
        for i in range(len(psv_planes)):
          f_map = f_extract_net(net_input_lowres[..., (i + 1) * 3: (i + 2) * 3], vgg_filepath=vgg_file, dense_res_aspp=dense_res_aspp)  # [b, h, w, c]
          f_lowres_psv.append(f_map)
        f_lowres_psv = tf.stack(f_lowres_psv, axis=-1)  # [b, h, w, c, n]

      if 'F_PSV_LR' in which_model_predict:
        tf.logging.info('Generating f_psv_lr!!')
        f_psv_lr = []
        for i in range(len(psv_planes)):
          f_map = f_extract_net(net_input_lr_syn[..., (i + 1) * 3: (i + 2) * 3], vgg_filepath=vgg_file, dense_res_aspp=dense_res_aspp)  # [b, h, w, c]
          f_psv_lr.append(f_map)
        f_psv_lr = tf.stack(f_psv_lr, axis=-1)  # [b, h, w, c, n]

      if 'image_psv' in which_model_predict:
        tf.logging.info('Generating image_psv!!')
        image_psv = tf.reshape(net_input[..., 3:], [batch_size, img_height, img_width, len(psv_planes), 3])
        image_psv = tf.transpose(image_psv, [0, 1, 2, 4, 3])  # [b, h, w, 3, n]

        lowres_psv_to_transfer = tf.reshape(net_input_lowres[..., 3:], [batch_size, int(img_height / 4.), int(img_width / 4.), len(psv_planes), 3])
        lowres_psv_to_transfer = tf.transpose(lowres_psv_to_transfer, [0, 1, 2, 4, 3])  # [b, h, w, 3, n]


      # network execution
      if which_model_predict == 'guided_ASPP_lowres_image_psv':
        tf.logging.info('guided_ASPP_lowres_image_psv')
        hr_syn, attention_map, init_attention_map = guided_att_hr_syn_net(f_lr, f_lowres_psv, net_input, image_psv)

      if which_model_predict == 'guided_nnup_ASPP_lowres_image_psv':
        tf.logging.info('guided_nnup_ASPP_lowres_image_psv')
        hr_syn, attention_map, init_attention_map = guided_att_hr_syn_net(f_lr, f_lowres_psv, net_input, image_psv, nnup=True)

      if which_model_predict == 'only_psv_guided_nnup_denseASPP_lowres_image_psv':
        tf.logging.info('only_psv_guided_nnup_denseASPP_lowres_image_psv')
        hr_syn, attention_map, init_attention_map = guided_att_hr_syn_net(f_lr, f_lowres_psv, net_input[..., 3:], image_psv, nnup=True)

      if which_model_predict == 'psv_guided_ASPP_lowres_image_psv':
        tf.logging.info('psv_guided_ASPP_lowres_image_psv')
        hr_syn, attention_map, init_attention_map = psv_guided_att_hr_syn_net(f_lr, f_lowres_psv, image_psv)

      if which_model_predict == 'psv_guided_nnup_denseASPP_lowres_image_psv':
        tf.logging.info('psv_guided_nnup_denseASPP_lowres_image_psv\nnnup=True!!!')
        hr_syn, attention_map, init_attention_map = psv_guided_att_hr_syn_net(f_lr, f_lowres_psv, image_psv, nnup=True)

      if which_model_predict == 'guided_nnup_denseASPP_lowres_image_psv_lowresL1':
        tf.logging.info('guided_nnup_denseASPP_lowres_image_psv\nnnup=True!!!')
        hr_syn, attention_map, init_syn, init_attention_map = guided_att_hr_syn_net(f_lr, f_lowres_psv, net_input, image_psv, lowres_psv_to_transfer, nnup=True)

      if which_model_predict == 'non_guided_nnup_denseASPP_lowres_image_psv_lowresL1':
        tf.logging.info('non_guided_nnup_denseASPP_lowres_image_psv_lowresL1\nnnup=True!!!')
        hr_syn, attention_map, init_syn, init_attention_map = guided_att_hr_syn_net(f_lr, f_lowres_psv, None, image_psv, lowres_psv_to_transfer, nnup=True)

      if which_model_predict == 'no_att_guided_nnup_denseASPP_lowres_image_psv_lowresL1':
        tf.logging.info('no_att_guided_nnup_denseASPP_lowres_image_psv_lowresL1\nnnup=True!!!')
        hr_syn, attention_map, init_syn, init_attention_map = guided_att_hr_syn_net(f_lr, f_lowres_psv, net_input, image_psv, lowres_psv_to_transfer, attention_flag=False, nnup=True)

      if which_model_predict == 'no_att_guided_nnup_denseASPP_lowres_image_psv_lowresL1_pretrain':
        tf.logging.info('no_att_guided_nnup_denseASPP_lowres_image_psv_lowresL1_pretrain\nnnup=True!!!')
        init_syn, init_attention_map = guided_att_hr_syn_net(f_lr, f_lowres_psv, net_input, image_psv, lowres_psv_to_transfer, lowres_pretrain=True, attention_flag=False, nnup=True)

      if which_model_predict == 'guided_nnup_denseASPP_lowres_image_psv_lowresL1_pretrain':
        tf.logging.info('guided_nnup_denseASPP_lowres_image_psv\nnnup=True!!!')
        init_syn, init_attention_map = guided_att_hr_syn_net(f_lr, f_lowres_psv, net_input, image_psv, lowres_psv_to_transfer, lowres_pretrain=True, nnup=True)

      if which_model_predict == 'guided_nnup_denseASPP_lowres_image_psv':
        tf.logging.info('guided_nnup_denseASPP_lowres_image_psv\nnnup=True!!!')
        hr_syn, attention_map, init_attention_map = guided_att_hr_syn_net(f_lr, f_lowres_psv, net_input, image_psv, nnup=True)

      if which_model_predict == 'shared_psv_guided_ASPP_lowres_image_psv':
        tf.logging.info('shared_psv_guided_ASPP_lowres_image_psv')
        hr_syn, attention_map, init_attention_map = self.shared_psv_guided_att_decoder(f_lr, f_lowres_psv, image_psv)

      if which_model_predict == 'ASPP_lowres_image_psv' or which_model_predict == 'denseASPP_lowres_image_psv':
        tf.logging.info('ASPP_lowres_image_psv or denseASPP_lowres_image_psv')
        f_hr_syn, hr_syn, attention_map, init_attention_map = att_hr_syn_net(f_lr, f_lowres_psv, image_psv, attention_decoder=True)

      if which_model_predict == 'nnup_denseASPP_lowres_image_psv':
        tf.logging.info('nnup_denseASPP_lowres_image_psv\nnnup=True!!!')
        f_hr_syn, hr_syn, attention_map, init_attention_map = att_hr_syn_net(f_lr, f_lowres_psv, image_psv, attention_decoder=True, nnup=True)

      if which_model_predict == 'vgg_F_PSV_LR_image_psv':
        tf.logging.info('vgg_F_PSV_LR_image_psv')
        f_hr_syn, hr_syn, attention_map, init_attention_map = att_hr_syn_net(f_lr, f_psv_lr, image_psv, attention_decoder=True)
                                                                        # attention_map: [b, h, w, n]
                                                                        # f_hr_syn: [b, h, w, c]
                                                                        # hr_syn: [b, h, w, 3]

      if which_model_predict == 'ASPP_f_psv_f_psv':
        tf.logging.info('ASPP_f_psv_f_psv')
        f_hr_syn, hr_syn, attention_map = att_hr_syn_net(f_lr, f_psv, f_psv)

      if which_model_predict == 'ASPP_f_psv_image_psv':
        tf.logging.info('ASPP_f_psv_image_psv')
        f_hr_syn, hr_syn, attention_map = att_hr_syn_net(f_lr, f_psv, image_psv)

      if which_model_predict == 'ASPP_F_PSV_LR_f_psv':
        tf.logging.info('ASPP_F_PSV_LR_f_psv')
        f_hr_syn, hr_syn, attention_map = att_hr_syn_net(f_lr, f_psv_lr, f_psv)

      if which_model_predict == 'ASPP_F_PSV_LR_image_psv':
        tf.logging.info('ASPP_F_PSV_LR_image_psv')
        f_hr_syn, hr_syn, attention_map = att_hr_syn_net(f_lr, f_psv_lr, image_psv)


    with tf.name_scope('residue_aggregation'):
      if not pretrain_hr_syn:
        sr_out = res_fuse_net(ref_image, hr_syn)

    # Collect output tensors
    pred = dict()
    if 'attention_map' in vars():
      pred['attention_map'] = attention_map  # E(0, 1), [b, h, w, n]
    if 'init_attention_map' in vars():
      pred['init_attention_map'] = init_attention_map
    if 'init_syn' in vars():
      pred['init_syn'] = init_syn
    pred['psv'] = net_input[:, :, :, 3:]
    # for debug:
    pred['f_lr'] = f_lr  # E(0, +oo), [b, h, w, c]
    # pred['f0'] = f0 * 2 - 1  # E(-1, +oo), [b, h, w, c]
    # pred['f1'] = f1 * 2 - 1  # E(-1, +oo), [b, h, w, c]
    # pred['f_psv'] = f_psv  # E(0, +oo), [b, h, w, c, n]
    if 'hr_syn' in vars():
      pred['hr_syn'] = hr_syn
    if not pretrain_hr_syn:
      pred['sr_out'] = sr_out

    # pred['psv_mask'] = psv_mask
    return pred

  def build_train_graph(self,
                        inputs,
                        min_depth,
                        max_depth,
                        num_psv_planes,
                        which_loss='pixel',
                        learning_rate=0.0002,
                        beta1=0.9,
                        vgg_model_file=None,
                        pretrain_hr_syn=False,
                        pretrain_lowres=False,
                        # use_psv_mask = False,
                        which_model_predict='ASPP_f_psv_f_psv'):
    """Construct the training computation graph.

    Args:
      inputs: dictionary of tensors (see 'input_data' below) needed for training
      min_depth: minimum depth for the plane sweep volume (PSV) and MPI planes
      max_depth: maximum depth for the PSV and MPI planes
      num_psv_planes: number of PSV planes for network input
      which_loss: which loss function to use (vgg or pixel)
      learning_rate: learning rate
      beta1: hyperparameter for ADAM
      vgg_model_file: path to VGG weights (required when VGG loss is used)
      pretrain_hr_syn: pretrain hr_syn_net or not
      pretrain_lowres: pretrain lowres attention estimation or not
      # use_psv_mask: use psv_mask when calculating depth
      which_model_predict: which model we use
    Returns:
      A train_op to be used for training.
    """
    with tf.name_scope('setup'):
      psv_planes = self.inv_depths(min_depth, max_depth, num_psv_planes)

    with tf.name_scope('input_data'):
      raw_tgt_image = inputs['tgt_image']
      raw_lr_image = inputs['lr_image']
      raw_ref_image = inputs['ref_image']
      raw_src_images = inputs['src_images']
      ref_pose = inputs['ref_pose']
      src_poses = inputs['src_poses']
      intrinsics = inputs['intrinsics']
      _, num_source, _, _ = src_poses.get_shape().as_list()

    with tf.name_scope('inference'):
      if 'vgg' in which_model_predict:
        vgg_to_extract = vgg_model_file
      else:
        vgg_to_extract = None
      pred = self.attention_ref_sr(raw_src_images, raw_ref_image, ref_pose, src_poses,
                                   intrinsics, psv_planes, pretrain_hr_syn=pretrain_hr_syn,
                                   vgg_file=vgg_to_extract, which_model_predict=which_model_predict)

      if pretrain_lowres:
        output_image = pred['init_syn']
        raw_tgt_image = tf.py_func(img_pil_downsampling, [self.preprocess_image(raw_tgt_image), 4.0], tf.float32)  # E(-1. 1)
        raw_tgt_image = (raw_tgt_image + 1.) / 2.  # E(0, 1) tf.float32
      elif pretrain_hr_syn:
        output_image = pred['hr_syn']
        # if use_psv_mask:
        #   raw_tgt_image = tf.where(pred['psv_mask'], raw_tgt_image, tf.zeros_like(raw_tgt_image))
      else:
        output_image = pred['sr_out']

    with tf.name_scope('loss'):
      """add accuracy for training while testing"""
      tgt_image = self.preprocess_image(raw_tgt_image)
      l2_accuracy = tf.reduce_mean(tf.nn.l2_loss(output_image - tgt_image))

      """train: total_loss"""
      if which_loss == 'vgg':
        # Normalized VGG loss (from
        # https://github.com/CQFIO/PhotographicImageSynthesis)
        def compute_error(real, fake):
          return tf.reduce_mean(tf.abs(fake - real))
        # It seems that raw_tgt_image belongs to [0, 1] -- Mayza
        vgg_real = build_vgg19(raw_tgt_image * 255.0, vgg_model_file)
        # vgg_real = build_vgg19(raw_tgt_image, vgg_model_file)
        rescaled_output_image = (output_image + 1.) / 2. * 255.0
        vgg_fake = build_vgg19(
            rescaled_output_image, vgg_model_file, reuse=True)
        p0 = compute_error(vgg_real['input'], vgg_fake['input'])
        p1 = compute_error(vgg_real['conv1_2'], vgg_fake['conv1_2']) / 2.6
        p2 = compute_error(vgg_real['conv2_2'], vgg_fake['conv2_2']) / 4.8
        p3 = compute_error(vgg_real['conv3_2'], vgg_fake['conv3_2']) / 3.7
        p4 = compute_error(vgg_real['conv4_2'], vgg_fake['conv4_2']) / 5.6
        p5 = compute_error(vgg_real['conv5_2'], vgg_fake['conv5_2']) * 10 / 1.5
        total_loss = p0 + p1 + p2 + p3 + p4 + p5

      if which_loss == 'vgg_lowresL1':
        tf.logging.info('which_loss == vgg_lowresL1')
        init_hr_syn = (pred['init_syn'] + 1.) / 2. * 255.0  # E(0, 255)
        lowres_raw_tgt_image = tf.py_func(img_pil_downsampling, [self.preprocess_image(raw_tgt_image), 4.0], tf.float32)  # E(-1, 1)
        lowres_raw_tgt_image = (lowres_raw_tgt_image + 1.) / 2. * 255.0  # E(0, 255)
        # Normalized VGG loss (from
        # https://github.com/CQFIO/PhotographicImageSynthesis)
        def compute_error(real, fake):
          return tf.reduce_mean(tf.abs(fake - real))
        # It seems that raw_tgt_image belongs to [0, 1] -- Mayza
        vgg_real = build_vgg19(raw_tgt_image * 255.0, vgg_model_file)
        # vgg_real = build_vgg19(raw_tgt_image, vgg_model_file)
        rescaled_output_image = (output_image + 1.) / 2. * 255.0
        vgg_fake = build_vgg19(rescaled_output_image, vgg_model_file, reuse=True)
        p0 = compute_error(vgg_real['input'], vgg_fake['input'])
        p1 = compute_error(vgg_real['conv1_2'], vgg_fake['conv1_2']) / 2.6
        p2 = compute_error(vgg_real['conv2_2'], vgg_fake['conv2_2']) / 4.8
        p3 = compute_error(vgg_real['conv3_2'], vgg_fake['conv3_2']) / 3.7
        p4 = compute_error(vgg_real['conv4_2'], vgg_fake['conv4_2']) / 5.6
        p5 = compute_error(vgg_real['conv5_2'], vgg_fake['conv5_2']) * 10 / 1.5

        p_lowres = compute_error(lowres_raw_tgt_image, init_hr_syn)
        # total_loss = 2 * p0 + p1 + p2 + p3 + p4 + p5 + 0.2 * p_lowres
        total_loss = 2 * p0 + p1 + p2 + p3 + p4 + p5 + 0.01 * p_lowres


      if which_loss == 'pixel':
        tgt_image = self.preprocess_image(raw_tgt_image)
        total_loss = tf.reduce_mean(tf.nn.l2_loss(output_image - tgt_image))
        # tf.reduce_mean(): Computes the mean of elements across dimensions of a tensor.

    with tf.name_scope('train_op'):
      train_vars = [var for var in tf.trainable_variables()]  # can specialize which variable to train
      optim = tf.train.AdamOptimizer(learning_rate, beta1)
      grads_and_vars = optim.compute_gradients(total_loss, var_list=train_vars)
      train_op = optim.apply_gradients(grads_and_vars)

    # Summaries
    # save training procedure and parameters on tensorboard -- Mayza
    # tf.summary.scaler('total_loss')
    tf.summary.scalar('training/total_loss', total_loss, collections=['train'])
    tf.summary.scalar('training/L2_accuracy', l2_accuracy, collections=['train'])
    tf.summary.scalar('test/L2_accuracy', l2_accuracy, collections=['test'])

    # Source images
    for i in range(num_source):
      src_image = raw_src_images[:, :, :, i * 3:(i + 1) * 3]
      tf.summary.image('src_image_%d' % i, src_image, collections=['train'])

    # for debug

    # tf.summary.image('inpts', self.deprocess_image(pred['inpts']))
    tf.summary.image('f_lr_0', self.deprocess_image(pred['f_lr'][..., 0:1] * 2 - 1), collections=['train'])
    # tf.summary.image('f_psv_0_0', self.deprocess_image(pred['f_psv'][..., 0:1, 0] * 2 - 1))
    # tf.summary.image('f_psv_8_1', self.deprocess_image(pred['f_psv'][..., 1:2, 8] * 2 - 1))
    # tf.summary.image('f_psv_16_2', self.deprocess_image(pred['f_psv'][..., 2:3, 16] * 2 - 1))
    # tf.summary.image('f_psv_31_3', self.deprocess_image(pred['f_psv'][..., 3:4, 31] * 2 - 1))
    # lr upscaled image
    tf.summary.image('lr_image', raw_lr_image, collections=['train'])
    # Output image
    if 'init_syn' in pred.keys():
      tf.summary.image('init_syn/train', self.deprocess_image(pred['init_syn']), collections=['train'])
      tf.summary.image('init_syn/test', self.deprocess_image(pred['init_syn']), collections=['test'])
    if 'hr_syn' in pred.keys():
      tf.summary.image('hr_syn/train', self.deprocess_image(pred['hr_syn']), collections=['train'])
      tf.summary.image('hr_syn/test', self.deprocess_image(pred['hr_syn']), collections=['test'])
    if not pretrain_hr_syn:
      tf.summary.image('output_image', self.deprocess_image(pred['sr_out']), collections=['train'])
    # Target image
    tf.summary.image('tgt_image/train', raw_tgt_image, collections=['train'])
    tf.summary.image('tgt_image/test', raw_tgt_image, collections=['test'])
    # Reference image
    tf.summary.image('ref_image/train', raw_ref_image, collections=['train'])
    tf.summary.image('ref_image/test', raw_ref_image, collections=['test'])
    # psv_mask add by Mayza
    # psv_mask = tf.image.convert_image_dtype(tf.cast(pred['psv_mask'], tf.float32), dtype=tf.uint8)
    # tf.summary.image('psv_mask', psv_mask, collections=['train'])
    # Depth map add by Mayza
    if 'attention_map' in pred.keys():
      depth_map = tf.expand_dims(tf.argmax(pred['attention_map'], axis=-1), -1)
      depth_map = tf.image.convert_image_dtype(tf.cast(depth_map, tf.float32) /
                                               tf.cast(tf.reduce_max(depth_map), tf.float32),
                                               dtype=tf.uint8)  # convert image dtype uint8:0-255
      tf.summary.image('depth_map_visualization', depth_map, collections=['train'])

      # visualize the attention map
      for i in range(0, num_psv_planes, 8):
        attention_i = tf.expand_dims(pred['attention_map'][:, :, :, i], -1)
        tf.summary.image('attention_layer_%d' % i, attention_i, collections=['train'])  # E(0, 1)

    # for i in range(0, num_psv_planes, 8):
    #   psv_i = self.deprocess_image(pred['psv'][:, :, :, i * 3: (i + 1) * 3])
    #   tf.summary.image('psv_%d' % i, psv_i, collections=['train'])  # E(0, 1)

    if 'init_attention_map' in pred.keys():
      depth_map = tf.expand_dims(tf.argmax(pred['init_attention_map'], axis=-1), -1)
      depth_map = tf.image.convert_image_dtype(tf.cast(depth_map, tf.float32) /
                                               tf.cast(tf.reduce_max(depth_map), tf.float32),
                                               dtype=tf.uint8)  # convert image dtype uint8:0-255
      tf.summary.image('init_depth_map_visualization', depth_map, collections=['train'])

      for i in range(0, num_psv_planes, 8):
        attention_i = tf.expand_dims(pred['init_attention_map'][:, :, :, i], -1)
        tf.summary.image('init_attention_layer_%d' % i, attention_i, collections=['train'])  # E(0, 1)

    return train_op

  def train(self, train_op, train_examples, valid_examples, checkpoint_dir, continue_train, summary_freq,
            save_latest_freq, max_steps, pretrained_hr_syn_checkpoint_dir=None):
            # save_latest_freq, max_steps, cameras_glob_, image_dir_, valid_cameras_glob_, valid_image_dir_, pretrained_hr_syn_checkpoint_dir=None):
    """Runs the training procedure.

    Args:
      train_op: op for training the network
      train_examples: .make_initializer()
      valid_examples: .make_initializer()
      checkpoint_dir: where to save the checkpoints and summaries
      continue_train: whether to restore training from previous checkpoint
      summary_freq: summary frequency
      save_latest_freq: Frequency of model saving (overwrites old one)
      max_steps: maximum training steps
      # cameras_glob_: as name
      # image_dir_: as name
      # valid_cameras_glob_: as name
      # valid_image_dir_: as name
      pretrained_hr_syn_checkpoint_dir: to restore the pretrained parameters
    """

    # collecting summaries
    s_training = tf.summary.merge_all('train')
    s_test = tf.summary.merge_all('test')

    parameter_count = tf.reduce_sum(
        [tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])  # tf.reduce_xxx() means dimension reduction
    global_step = tf.Variable(0, name='global_step', trainable=False)
    incr_global_step = tf.assign(global_step, global_step + 1)  # where global_step += 1
    saver = tf.train.Saver(
        [var for var in tf.model_variables()] + [global_step], max_to_keep=84)
    hr_syn_saver = tf.train.Saver([var for var in tf.model_variables() if 'net' in var.name or 'f_extract_net' in var.name or 'decoder' in var.name])
    sv = tf.train.Supervisor(
        logdir=checkpoint_dir, save_summaries_secs=0, saver=None)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with sv.managed_session(config=config) as sess:
      tf.logging.info('Trainable variables: ')
      for var in tf.trainable_variables():
        tf.logging.info(var.name)
      tf.logging.info('parameter_count = %d' % sess.run(parameter_count))

      if continue_train:
        if pretrained_hr_syn_checkpoint_dir is not None:
          hr_syn_checkpoint = tf.train.latest_checkpoint(pretrained_hr_syn_checkpoint_dir)
          if hr_syn_checkpoint is not None:
            tf.logging.info('Resume training from previous checkpoint from --pretrained_hr_syn_checkpoint_dir:%s' % pretrained_hr_syn_checkpoint_dir)
            hr_syn_saver.restore(sess, hr_syn_checkpoint)
        else:
          checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
          if checkpoint is not None:
            tf.logging.info('Resume training from previous checkpoint from --checkpoint_dir:%s' % checkpoint_dir)
            saver.restore(sess, checkpoint)

      for step in range(1, max_steps):
        start_time = time.time()
        fetches = {
            # 'train': train_op,
            'global_step': global_step,
            'incr_global_step': incr_global_step,
        }

        input_tgt_image = tf.get_default_graph().get_tensor_by_name('input_tgt_image:0')
        input_lr_image = tf.get_default_graph().get_tensor_by_name('input_lr_image:0')
        input_ref_image = tf.get_default_graph().get_tensor_by_name('input_ref_image:0')
        input_src_images = tf.get_default_graph().get_tensor_by_name('input_src_images:0')
        input_tgt_pose = tf.get_default_graph().get_tensor_by_name('input_tgt_pose:0')
        input_ref_pose = tf.get_default_graph().get_tensor_by_name('input_ref_pose:0')
        input_src_poses = tf.get_default_graph().get_tensor_by_name('input_src_poses:0')
        input_intrinsics = tf.get_default_graph().get_tensor_by_name('input_intrinsics:0')

        if step % summary_freq == 1:  # summary is added to fetches every certain interval
          fetches['summary'] = s_training

        # during validation
        if step % summary_freq == 0:   # summary is added to fetches every certain interval
          fetches['valid_summary'] = s_test
          valid_batch = sess.run(valid_examples)
          results = sess.run(fetches, feed_dict={input_tgt_image: valid_batch['tgt_image'],
                                                 input_lr_image: valid_batch['lr_image'],
                                                 input_ref_image: valid_batch['ref_image'],
                                                 input_src_images: valid_batch['src_images'],
                                                 input_tgt_pose: valid_batch['tgt_pose'],
                                                 input_ref_pose: valid_batch['ref_pose'],
                                                 input_src_poses: valid_batch['src_poses'],
                                                 input_intrinsics: valid_batch['intrinsics']})

        else:
          # during training
          train_batch = sess.run(train_examples)
          fetches['train'] = train_op
          results = sess.run(fetches, feed_dict={input_tgt_image: train_batch['tgt_image'],
                                                 input_lr_image: train_batch['lr_image'],
                                                 input_ref_image: train_batch['ref_image'],
                                                 input_src_images: train_batch['src_images'],
                                                 input_tgt_pose: train_batch['tgt_pose'],
                                                 input_ref_pose: train_batch['ref_pose'],
                                                 input_src_poses: train_batch['src_poses'],
                                                 input_intrinsics: train_batch['intrinsics']})

        gs = results['global_step']

        # during validation
        if step % summary_freq == 0:
          sv.summary_computed(sess, results['valid_summary'], gs)

        # during training
        if step % summary_freq == 1:
          sv.summary_computed(sess, results['summary'], gs)
          tf.logging.info(
              '[Step %.8d] time: %4.4f/it' % (gs, time.time() - start_time))

        if step % save_latest_freq == 0:
          tf.logging.info(' [*] Saving checkpoint to %s...' % checkpoint_dir)
          saver.save(sess, os.path.join(checkpoint_dir, 'model.latest'), global_step=gs)
          # saver.save(sess, os.path.join(checkpoint_dir, 'model.latest'))

      sess.graph.finalize()  # M: in case extra variables are defined during training

  def psvs_lr_syn(self, net_input, num_mpi_planes, resize_factor=8.0):
    """

    :param net_input: lr_input|hr_psvs
    :param num_mpi_planes: num_mpi_planes
    :param resize_factor: resize factor
    :return: lr_input|lr_psvs
    """
    # resize module -- Mayza
    def img_processing_lr_syn(img):
      batch_size, img_height, img_width, _ = img.shape
      img_op = img.copy()
      for i in range(batch_size):
        img_temp = img_op[i, ...]
        img_temp_pil = Image.fromarray(((img_temp + 1) / 2.0 * 255.0).astype('uint8'))  # (the network input is -1~1)
        img_downsampled = img_temp_pil.resize((int(img_width / resize_factor),
                                               int(img_height / resize_factor)),
                                              resample=Image.BICUBIC)
        img_upsampled = img_downsampled.resize((img_width, img_height), resample=Image.BICUBIC)
        img_op[i, ...] = (np.asarray(img_upsampled, dtype=np.float32) / 255.0 ) * 2 - 1
      return img_op

    net_input_lr = []
    net_input_lr.append(net_input[:, :, :, :3])
    for i in range(num_mpi_planes):
      curr_rgb = net_input[:, :, :, (i + 1) * 3: (i + 2) * 3]
      net_input_lr.append(tf.py_func(img_processing_lr_syn, [curr_rgb], tf.float32))
    net_input_lr = tf.concat(net_input_lr, axis=3)
    net_input_lr.set_shape(net_input.get_shape().as_list())
    return net_input_lr

  def psvs_lowres_syn(self, net_input, num_mpi_planes, resize_factor=4.0):
    """

    :param net_input: lr_input|hr_psvs
    :param num_mpi_planes: num_mpi_planes
    :param resize_factor: resize factor
    :return: lr_input|lr_psvs
    """
    net_input_lr = []
    net_input_lr.append(tf.py_func(img_pil_downsampling, [net_input[:, :, :, :3], resize_factor], tf.float32))
    for i in range(num_mpi_planes):
      curr_rgb = net_input[:, :, :, (i + 1) * 3: (i + 2) * 3]
      net_input_lr.append(tf.py_func(img_pil_downsampling, [curr_rgb, resize_factor], tf.float32))
    net_input_lr = tf.concat(net_input_lr, axis=3)
    b, h, w, c = net_input.get_shape().as_list()
    net_input_lr.set_shape([b, int(h/4), int(w/4), c])
    return net_input_lr

  def format_network_input(self, ref_image, psv_src_images, ref_pose,
                           psv_src_poses, planes, intrinsics):
    """Format the network input (reference source image + PSV of the 2nd image).

    Args:
      ref_image: reference source image [batch, height, width, 3]
      psv_src_images: stack of source images (excluding the ref image)
                      [batch, height, width, 3*(num_source -1)]
      ref_pose: reference world-to-camera pose (where PSV is constructed)
                [batch, 4, 4]
      psv_src_poses: input poses (world to camera) [batch, num_source-1, 4, 4]
      planes: list of scalar depth values for each plane
      intrinsics: camera intrinsics [batch, 6, 3]
    Returns:
      net_input: [batch, height, width, (num_source-1)*#planes*3 + 3]
    """
    _, num_psv_source, _, _ = psv_src_poses.get_shape().as_list()
    net_input = []
    net_input.append(ref_image)
    for i in range(num_psv_source):
      curr_pose = tf.matmul(psv_src_poses[:, i], tf.matrix_inverse(ref_pose))
      curr_image = psv_src_images[:, :, :, i * 3:(i + 1) * 3]
      curr_psv = pj.plane_sweep(curr_image, planes, curr_pose, intrinsics)
      net_input.append(curr_psv)
    net_input = tf.concat(net_input, axis=3)
    return net_input

  def preprocess_image(self, image):
    """Preprocess the image for CNN input.

    Args:
      image: the input image in either float [0, 1] or uint8 [0, 255]
    Returns:
      A new image converted to float with range [-1, 1]
    """
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image * 2 - 1

  def deprocess_image(self, image):
    """Undo the preprocessing.

    Args:
      image: the input image in float with range [-1, 1]
    Returns:
      A new image converted to uint8 [0, 255]
    """
    image = (image + 1.) / 2.
    return tf.image.convert_image_dtype(image, dtype=tf.uint8)

  def inv_depths(self, start_depth, end_depth, num_depths, baseline=1.0):
    """Sample reversed, sorted inverse depths between a near and far plane.

    Args:
      start_depth: The first depth (i.e. near plane distance).
      end_depth: The last depth (i.e. far plane distance).
      num_depths: The total number of depths to create. start_depth and
          end_depth are always included and other depths are sampled
          between them uniformly according to inverse depth.
      baseline: the baseline of cameras/frames
    Returns:
      The depths sorted in descending order (so furthest first). This order is
      useful for back to front compositing.
    """
    inv_start_depth = baseline / start_depth
    inv_end_depth = baseline / end_depth
    depths = [start_depth, end_depth]
    for i in range(1, num_depths - 1):
      fraction = float(i) / float(num_depths - 1)
      inv_depth = inv_start_depth + (inv_end_depth - inv_start_depth) * fraction
      depths.append(baseline / inv_depth)
    depths = sorted(depths)
    return depths[::-1]

  def shared_psv_guided_att_decoder(self, f_lr, f_psv_to_attention, psv_to_transfer):
    """
    :param f_lr: input lr features [b, h, w, c]
    :param f_psv_to_attention: stack of psv features [b, h, w, c, 32] to generate attention
    :param psv_to_transfer: which psv format transferred to LR [b, h_t, w_t, c_t, n]
    :return: hr_syn, attention_map, init_attention_map
    """
    b, h, w, c, n = f_psv_to_attention.get_shape().as_list()
    _, h_t, w_t, c_t, _ = psv_to_transfer.get_shape().as_list()
    tf.logging.info('Num of channels of f_psv_to_attention = %d' % c)
    tf.logging.info('Num of channels of psv_to_transfer = %d' % c_t)

    # Construction of f_lr_wave
    f_lr_wave = tf.reshape(f_lr, [b * h * w, c])
    f_lr_wave = tf.reshape(f_lr_wave, [b * h * w, 1, c])  # [bhw, 1, c]

    # Construction of f_psv_wave
    f_psv_to_attention = tf.reshape(f_psv_to_attention, [b * h * w, c, n])

    # Construction of A
    attention_map = tf.matmul(f_lr_wave, f_psv_to_attention)  # [bhw, 1, n]
    attention_map = tf.reshape(attention_map, [b * h * w, n])  # [bhw, n]
    attention_map = tf.reshape(attention_map, [b, h, w, n])  # [b, h, w, n]
    init_attention_map = attention_map

    # Shared psv guided attention map decoder
    attention_map_hr = []
    for i in range(n):
      attention_map_hr_i = psv_guided_att_decoder(attention_map[..., i:(i+1)], psv_to_transfer[..., i])
      attention_map_hr.append(attention_map_hr_i)
      # attention_map_hr.append(psv_guided_att_decoder(attention_map[..., i:(i+1)], psv_to_transfer[..., i]))
    attention_map = tf.stack(attention_map_hr, axis=-1)
    del attention_map_hr
    attention_map = tf.reshape(attention_map, [b, h_t, w_t, n])

    # apply softmax to attention map
    attention_map = tf.nn.softmax(attention_map, axis=-1)
    attention_map = tf.reshape(attention_map, [b * h_t * w_t, n])

    # Transfer psv to construct hr_syn
    psv_to_transfer = tf.reshape(psv_to_transfer, [b * h_t * w_t, c_t, n])  # c_t == 3 for image psv,                                                                           # c_t == c for f_psv

    hr_syn = []
    for i in range(c_t):
      psv_tt_ci = psv_to_transfer[:, i, :]  # [bhw, n]
      hr_syn_ci = tf.multiply(psv_tt_ci, attention_map)  # [bhw, n]
      hr_syn_ci = tf.reduce_sum(hr_syn_ci, axis=-1)  # [bhw]
      hr_syn.append(hr_syn_ci)
    hr_syn = tf.stack(hr_syn, axis=-1)
    hr_syn = tf.reshape(hr_syn, [b, h_t, w_t, c_t])

    # reshape attention map
    attention_map = tf.reshape(attention_map, [b, h_t, w_t, n])

    return hr_syn, attention_map, init_attention_map

  def mpi_render_view(self, rgba_layers, tgt_pose, planes, intrinsics):
    """Render a target view from an MPI representation.

    Args:
      rgba_layers: input MPI [batch, height, width, #planes, 4]
      tgt_pose: target pose to render from [batch, 4, 4]
      planes: list of depth for each plane
      intrinsics: camera intrinsics [batch, 3, 3]
    Returns:
      rendered view [batch, height, width, 3]
    """
    batch_size, _, _ = tgt_pose.get_shape().as_list()
    depths = tf.constant(planes, shape=[len(planes), 1])
    depths = tf.tile(depths, [1, batch_size])
    rgba_layers = tf.transpose(rgba_layers, [3, 0, 1, 2, 4])
    proj_images = pj.projective_forward_homography(rgba_layers, intrinsics,
                                                   tgt_pose, depths)
    proj_images_list = []
    for i in range(len(planes)):
      proj_images_list.append(proj_images[i])
    output_image = pj.over_composite(proj_images_list)
    return output_image

  def refsr_render_view(self, rgbs, alphas, tgt_pose, planes, intrinsics):
    """Render a target view from an MPI representation.

    Args:
      rgbs: input rgbs [batch, height, width, 3*planes]
      alphas: input alphas [batch, height, width, planes]
      tgt_pose: target pose to render from [batch, 4, 4]
      planes: list of depth for each plane
      intrinsics: camera intrinsics [batch, 6, 3]
    Returns:
      rendered view [batch, height, width, 3]
    """
    _, _, _, color_ch = rgbs.get_shape().as_list()
    batch_size, image_height, image_width, planes_num = alphas.get_shape().as_list()
    depths = tf.constant(planes, shape=[len(planes), 1])
    depths = tf.tile(depths, [1, batch_size])
    for i in range(planes_num):
      if color_ch > 3:
        curr_rgba = tf.concat([rgbs[..., i * 3: (i + 1) * 3], alphas[..., i: (i + 1)]], axis=3)
      else:
        curr_rgba = tf.concat([rgbs, alphas[..., i: (i + 1)]], axis=3)
      if i == 0:
        rgba_layers = curr_rgba
      else:
        rgba_layers = tf.concat([rgba_layers, curr_rgba], axis=3)
    rgba_layers = tf.reshape(rgba_layers, [batch_size, image_height, image_width, planes_num, 4])

    rgba_layers = tf.transpose(rgba_layers, [3, 0, 1, 2, 4])
    proj_images = pj.projective_forward_homography(rgba_layers, intrinsics,
                                                   tgt_pose, depths)
    # direct multiply and summation
    for i in range(planes_num):
      rgb = proj_images[i, :, :, :, 0:3]
      alpha = proj_images[i, :, :, :, 3:]
      if i == 0:
        output = rgb
      else:
        rgb_by_alpha = rgb * alpha
        output = rgb_by_alpha + output * (1 - alpha)
    return output
