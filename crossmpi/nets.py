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
"""Network definitions for multiplane image (MPI) prediction networks.
"""
from __future__ import division
import numpy as np
import tensorflow as tf
import scipy.io as sio
from tensorflow.contrib import slim
from crossmpi.ops import *
from utils import img_pil_downsampling



def get_weight_bias(vgg_layers, i):
  weights = vgg_layers[i][0][0][2][0][0]
  weights = tf.get_variable('weights%02d' % i, weights.shape, initializer=tf.constant_initializer(weights))

  bias = vgg_layers[i][0][0][2][0][1]
  bias = tf.get_variable('bias%02d' % i, [bias.size], initializer=tf.constant_initializer(bias))
  return [weights, bias]

# def f_extract_net(inputs, ngf=64, vscope='f_extract_net'):
#   """Network definition for multiplane image (MPI) inference.
#   reuse=tf.AUTO_REUSE: for weight sharing
#
#   Args:
#     inputs: stack of input images [batch, height, width, input_channels]
#     num_outputs: number of output channels
#     ngf: number of features for the first conv layer
#     vscope: variable scope
#   Returns:
#     pred: network output at the same spatial resolution as the inputs.
#   """
#   with tf.variable_scope(vscope, reuse=tf.AUTO_REUSE):
#     with slim.arg_scope([slim.conv2d], normalizer_fn=slim.layer_norm):
#       cnv1_1 = slim.conv2d(inputs, ngf, [3, 3], scope='conv1_1', stride=1)
#       cnv1_2 = slim.conv2d(cnv1_1, ngf * 2, [3, 3], scope='conv1_2', stride=2)
#
#       cnv2_1 = slim.conv2d(cnv1_2, ngf * 2, [3, 3], scope='conv2_1', stride=1)
#       cnv2_2 = slim.conv2d(cnv2_1, ngf * 4, [3, 3], scope='conv2_2', stride=2)
#
#       cnv3_1 = slim.conv2d(cnv2_2, ngf * 4, [3, 3], scope='conv3_1', stride=1)
#       cnv3_2 = slim.conv2d(cnv3_1, ngf * 4, [3, 3], scope='conv3_2', stride=1)
#       cnv3_3 = slim.conv2d(cnv3_2, ngf * 8, [3, 3], scope='conv3_3', stride=2)
#
#       cnv4_1 = slim.conv2d(
#           cnv3_3, ngf * 8, [3, 3], scope='conv4_1', stride=1, rate=2)
#       cnv4_2 = slim.conv2d(
#           cnv4_1, ngf * 8, [3, 3], scope='conv4_2', stride=1, rate=2)
#       cnv4_3 = slim.conv2d(
#           cnv4_2, ngf * 8, [3, 3], scope='conv4_3', stride=1, rate=2)
#
#       # Adding skips
#       skip = tf.concat([cnv4_3, cnv3_3], axis=3)
#       cnv6_1_0 = tf.image.resize_images(skip, 2 * tf.shape(skip)[1:3], method=1)
#       cnv6_1 = slim.conv2d(cnv6_1_0, ngf * 4, [4, 4], scope='conv6_1_nodeconv', stride=1)
#
#       cnv6_2 = slim.conv2d(cnv6_1, ngf * 4, [3, 3], scope='conv6_2', stride=1)
#       cnv6_3 = slim.conv2d(cnv6_2, ngf * 4, [3, 3], scope='conv6_3', stride=1)
#
#       skip = tf.concat([cnv6_3, cnv2_2], axis=3)
#       cnv7_1_0 = tf.image.resize_images(skip, 2 * tf.shape(skip)[1:3], method=1)
#       cnv7_1 = slim.conv2d(cnv7_1_0, ngf * 2, [4, 4], scope='conv7_1_nodeconv', stride=1)
#
#       cnv7_2 = slim.conv2d(cnv7_1, ngf * 2, [3, 3], scope='conv7_2', stride=1)
#
#       skip = tf.concat([cnv7_2, cnv1_2], axis=3)
#       cnv8_1_0 = tf.image.resize_images(skip, 2 * tf.shape(skip)[1:3], method=1)
#       cnv8_1 = slim.conv2d(cnv8_1_0, ngf, [4, 4], scope='conv8_1_nodeconv', stride=1)
#
#       cnv8_2 = slim.conv2d(cnv8_1, ngf, [3, 3], scope='conv8_2', stride=1)
#
#
#       feat = cnv8_2
#       return feat

# def f_extract_net(inputs, ngf=64, vscope='f_extract_net', vgg_filepath=None):
#   """Network definition for multiplane image (MPI) inference.
#   reuse=tf.AUTO_REUSE: for weight sharing
#
#   Args:
#     inputs: stack of input images [batch, height, width, input_channels]
#     num_outputs: number of output channels
#     ngf: number of features for the first conv layer
#     vscope: variable scope
#   Returns:
#     pred: network output at the same spatial resolution as the inputs.
#   """
#   with tf.variable_scope(vscope, reuse=tf.AUTO_REUSE):
#     with slim.arg_scope([slim.conv2d], normalizer_fn=None):
#       conv1_1 = slim.conv2d(inputs, ngf, [3, 3], scope='conv1_1', stride=1)
#       conv2_1 = slim.conv2d(conv1_1, ngf, [3, 3], scope='conv2_1', stride=1)
#       conv3_1 = slim.conv2d(conv2_1, ngf, [3, 3], scope='conv3_1', stride=1)
#       conv4_1 = slim.conv2d(conv3_1, ngf//4, [1, 1], scope='conv4_1', stride=1, activation_fn=None)
#       return conv4_1
#       # return inputs, conv1_1, conv2_1, conv3_1

# def f_extract_net(inputs, ngf=64, vscope='f_extract_net'):
#   """Network definition for multiplane image (MPI) inference.
#   reuse=tf.AUTO_REUSE: for weight sharing
#
#   Args:
#     inputs: stack of input images [batch, height, width, input_channels]
#     num_outputs: number of output channels
#     ngf: number of features for the first conv layer
#     vscope: variable scope
#   Returns:
#     pred: network output at the same spatial resolution as the inputs.
#   """
#   with tf.variable_scope(vscope, reuse=tf.AUTO_REUSE):
#     with slim.arg_scope([slim.conv2d], normalizer_fn=None):
#     # with slim.arg_scope([slim.conv2d], normalizer_fn=slim.layer_norm):
#       # ASPP
#       cnv1_1 = slim.conv2d(inputs, ngf, [3, 3], scope='conv1_1', stride=1)
#       cnv2_1 = slim.conv2d(cnv1_1, ngf, [3, 3], scope='conv2_1', stride=1)
#       cnv2_2 = slim.conv2d(cnv1_1, ngf, [3, 3], scope='conv2_2', stride=1, rate=4)
#       cnv2_3 = slim.conv2d(cnv1_1, ngf, [3, 3], scope='conv2_3', stride=1, rate=8)
#       cnv3_1 = slim.conv2d(tf.concat([cnv2_1, cnv2_2, cnv2_3], axis=-1), ngf, [1, 1], scope='', stride=1)
#       cnv4_1 = tf.add(cnv1_1, cnv3_1)
#
#       net = slim.conv2d(cnv4_1, ngf, [3, 3], scope='conv5_1', stride=1)
#       # ResBlock
#       net_ = slim.conv2d(net, ngf, [3, 3], scope='conv6_1', stride=1)
#       net_ = slim.conv2d(net_, ngf, [3, 3], scope='conv6_2', stride=1)
#
#       feat = tf.add(net, net_)
#       feat = slim.conv2d(feat, ngf // 8, [1, 1], scope='conv7_1', stride=1, activation_fn=None)
#       return feat

def f_extract_net(inputs, ngf=64, vgg_filepath=None, vscope='f_extract_net', dense_res_aspp=False):
  """Network definition for multiplane image (MPI) inference.
  reuse=tf.AUTO_REUSE: for weight sharing

  Args:
    inputs: stack of input images [batch, height, width, input_channels]
    ngf: number of features for the first conv layer
    vgg_filepath: vgg19 model filepath
    vscope: variable scope
    dense_res_aspp: use dense_res_ASPP or not
  Returns:
    pred: network output at the same spatial resolution as the inputs.
  """
  with tf.variable_scope(vscope, reuse=tf.AUTO_REUSE):
      if dense_res_aspp:
          tf.logging.info('Using Dense resASPP+resBlock as feature extractor!!')
          with slim.arg_scope([slim.conv2d], normalizer_fn=None):
              # ASPP
              net = slim.conv2d(inputs, ngf, [3, 3], scope='conv_in', stride=1)
              # ResBlock
              net_ = slim.conv2d(net, ngf, [3, 3], scope='res_in/c1', stride=1)
              net_ = slim.conv2d(net_, ngf, [3, 3], scope='res_in/c2', stride=1)
              net = tf.add(net, net_)
              # residue aspp block + res_block
              for i in range(1):
                  aspp_out = net
                  for j in range(3):
                      cnv2_1 = slim.conv2d(aspp_out, ngf, [3, 3], scope='module%s/aspp%s/c1d1' % (i,j), stride=1)
                      cnv2_2 = slim.conv2d(aspp_out, ngf, [3, 3], scope='module%s/aspp%s/c2d4' % (i,j), stride=1, rate=4)
                      cnv2_3 = slim.conv2d(aspp_out, ngf, [3, 3], scope='module%s/aspp%s/c3d8' % (i,j), stride=1, rate=8)
                      aspp_out = slim.conv2d(tf.concat([cnv2_1, cnv2_2, cnv2_3], axis=-1), ngf, [1, 1],
                                             scope='aspp%s/co' % i, stride=1)
                      net = tf.add(net, aspp_out)
                  # ResBlock
                  net_ = slim.conv2d(net, ngf, [3, 3], scope='res%s/c1' % i, stride=1)
                  net_ = slim.conv2d(net_, ngf, [3, 3], scope='res%s/c2' % i, stride=1)
                  net = tf.add(net, net_)
                  net = slim.conv2d(net, ngf, [3, 3], scope='res%s/co' % i, stride=1)

              feat = slim.conv2d(net, ngf // 8, [1, 1], scope='conv_out', stride=1, activation_fn=None)

      elif vgg_filepath is not None:
          tf.logging.info('Using VGG19 as feature extractor!!')

          # normalize to vgg input p.s. inputs E (-1, 1)
          imagenet_mean = tf.constant([0.485, 0.456, 0.406], shape=[1, 1, 1, 3])
          imagenet_std = tf.constant([0.229, 0.224, 0.225], shape=[1, 1, 1, 3])
          inputs = (inputs - (imagenet_mean * 2 - 1)) / (2 * imagenet_std)

          with open(vgg_filepath, 'r') as f:
              vgg_rawnet = sio.loadmat(f)
          vgg_layers = vgg_rawnet['layers'][0]

          # conv1_1, c=64
          wb_1_1 = get_weight_bias(vgg_layers, 0)
          conv1_1 = tf.nn.conv2d(inputs, wb_1_1[0], strides=[1, 1, 1, 1], padding='SAME', name='conv1_1')
          conv1_1 = tf.nn.relu(conv1_1 + wb_1_1[1])

          # conv1_2, c=64
          wb_1_2 = get_weight_bias(vgg_layers, 2)
          conv1_2 = tf.nn.conv2d(conv1_1, wb_1_2[0], strides=[1, 1, 1, 1], padding='SAME', name='conv1_2')
          conv1_2 = tf.nn.relu(conv1_2 + wb_1_2[1])

          pool1 = tf.nn.avg_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

          # conv2_1, c=128
          wb_2_1 = get_weight_bias(vgg_layers, 5)
          conv2_1 = tf.nn.conv2d(pool1, wb_2_1[0], strides=[1, 1, 1, 1], padding='SAME', name='conv2_1')
          conv2_1 = tf.nn.relu(conv2_1 + wb_2_1[1])

          # conv2_2, c=128
          wb_2_2 = get_weight_bias(vgg_layers, 7)
          conv2_2 = tf.nn.conv2d(conv2_1, wb_2_2[0], strides=[1, 1, 1, 1], padding='SAME', name='conv2_2')
          conv2_2 = tf.nn.relu(conv2_2 + wb_2_2[1])

          pool2 = tf.nn.avg_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

          # conv3_1, c=256
          wb_3_1 = get_weight_bias(vgg_layers, 10)
          conv3_1 = tf.nn.conv2d(pool2, wb_3_1[0], strides=[1, 1, 1, 1], padding='SAME', name='conv3_1')
          conv3_1 = tf.nn.relu(conv3_1 + wb_3_1[1])

          # conv3_2, c=256
          # wb_3_2 = get_weight_bias(vgg_layers, 12)
          # conv3_2 = tf.nn.conv2d(conv3_1, wb_3_2[0], strides=[1, 1, 1, 1], padding='SAME', name='conv3_2')
          # conv3_2 = tf.nn.relu(conv3_2 + wb_3_2[1])


          # c=32
          feat = slim.conv2d(conv3_1, ngf // 2, [3, 3], scope='conv4', stride=1)
          feat = slim.conv2d(feat, ngf // 4, [1, 1], scope='1x1_out', stride=1, activation_fn=None)
          # feat = slim.conv2d(feat, ngf // 2, [1, 1], scope='1x1_out', stride=1, activation_fn=None)
      else:
          tf.logging.info('Using ASPP+resBlock as feature extractor!!')
          with slim.arg_scope([slim.conv2d], normalizer_fn=None):
              # ASPP
              cnv1_1 = slim.conv2d(inputs, ngf, [3, 3], scope='conv1_1', stride=1)
              cnv2_1 = slim.conv2d(cnv1_1, ngf, [3, 3], scope='conv2_1', stride=1)
              cnv2_2 = slim.conv2d(cnv1_1, ngf, [3, 3], scope='conv2_2', stride=1, rate=4)
              cnv2_3 = slim.conv2d(cnv1_1, ngf, [3, 3], scope='conv2_3', stride=1, rate=8)
              cnv3_1 = slim.conv2d(tf.concat([cnv2_1, cnv2_2, cnv2_3], axis=-1), ngf, [1, 1], scope='', stride=1)
              cnv4_1 = tf.add(cnv1_1, cnv3_1)

              net = slim.conv2d(cnv4_1, ngf, [3, 3], scope='conv5_1', stride=1)
              # ResBlock
              net_ = slim.conv2d(net, ngf, [3, 3], scope='conv6_1', stride=1)
              net_ = slim.conv2d(net_, ngf, [3, 3], scope='conv6_2', stride=1)

              feat = tf.add(net, net_)
              feat = slim.conv2d(feat, ngf // 8, [1, 1], scope='conv7_1', stride=1, activation_fn=None)

      return feat

def att_hr_syn_net(f_lr, f_psv_to_attention, psv_to_transfer, attention_decoder=False, ngf=32, vscope='net', reuse_weights=False, nnup=False):
  """Network definition for hr_syn inference.

  Args:
    f_lr: input lr features [b, h, w, c]
    f_psv_to_attention: stack of psv features [b, h, w, c, 32] to generate attention
    psv_to_transfer: which psv format transferred to LR
    attention_decoder: true for use attention_decoder
    ngf: number of features for the first conv layer
    vscope: variable scope
    reuse_weights: whether to reuse weights (for weight sharing)
    nnup: whether use nn_resize + conv
  Returns:
    pred: network output at the same spatial resolution as the inputs.
  """

  # def my_matmul(mat_m_n, mat_n):
  #     """
  #     This function operate mat_m = tf.matmul(mat_m_n, mat_n_1)
  #     :param mat_m_n: tensor mat shape like [b, m, n]
  #     :param mat_n: tensor mat shape like [b, n]
  #     :return: mat_m: tensor mat shape like[b, m]
  #     """
  #     _, m_, _ = mat_m_n.get_shape().as_list()
  #     mat_m = []
  #     for i_ in range(m_):
  #         mat_m_i = mat_m_n[:, i_, :]  # [batch_size, n]
  #         mat_m_i = tf.multiply(mat_m_i, mat_n)  # [batch_size, n]
  #         mat_m_i = tf.reduce_sum(mat_m_i, axis=-1)  # [batch_size]
  #         mat_m.append(mat_m_i)
  #     mat_m = tf.stack(mat_m, axis=-1) # [batch_size, m]
  #     return mat_m

  with tf.variable_scope(vscope, reuse=reuse_weights):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], normalizer_fn=None):
        b, h, w, c, n = f_psv_to_attention.get_shape().as_list()
        _, h_t, w_t, c_t, _ = psv_to_transfer.get_shape().as_list()
        tf.logging.info('Num of channels of f_psv_to_attention = %d' % c)
        tf.logging.info('Num of channels of psv_to_transfer = %d' % c_t)

        # Construction of f_lr_wave
        f_lr_wave = tf.reshape(f_lr, [b * h * w, c])
        f_lr_wave = tf.reshape(f_lr_wave, [b * h * w, 1, c])  # [bhw, 1, c]
        # f_lr_wave = tf.expand_dims(f_lr_wave, axis=1)

        # Construction of f_psv_wave
        f_psv_to_attention = tf.reshape(f_psv_to_attention, [b * h * w, c, n])
        # f_psv_to_attention = tf.transpose(f_psv_to_attention, [0, 2, 1])  # [bhw, n, c]

        # Construction of A
        attention_map = tf.matmul(f_lr_wave, f_psv_to_attention)  # [bhw, 1, n]
        # attention_map = my_matmul(f_psv_to_attention, f_lr_wave)  # [bhw, n]

        attention_map = tf.reshape(attention_map, [b * h * w, n])  # [bhw, n]

        init_attention_map = None
        if attention_decoder:
            attention_map = tf.reshape(attention_map, [b, h, w, n])  # [b, h, w, n]
            init_attention_map = attention_map

            # attention_map = tf.image.resize_images(attention_map, [h_t, w_t])
            tf.logging.info('attention_decoder!!!!!!')
            # -------------2x up-------------
            net = slim.conv2d(attention_map, ngf, [3, 3], scope='conv1_1', stride=1)
            temp = net
            # ======residue blocks(1)======
            for i in range(2):
                net_ = slim.conv2d(net, ngf, [3, 3], scope='conv1_1/b%s/c1' % i, stride=1)
                net_ = slim.conv2d(net_, ngf, [3, 3], scope='conv1_1/b%s/c2' % i, stride=1)
                net_ = tf.add(net, net_)
                net = net_
            net = slim.conv2d(net, ngf, [3, 3], scope='conv1_2', stride=1)
            net = tf.add(temp, net)

            # ======subpixel upsampling(1)======
            if not nnup:
                # Note that, the number of input channels == (scale x scale) x The number of output channels.
                net = conv2(net, 3, 4 * ngf, 1, scope='suppixelconv_1')
                net = pixelShuffler(net, scale=2)
                # net = prelu_tf(net)
                net = tf.nn.relu(net)  # can't find the definition of alpha in parameter relu, so change to relu
            else:
                net = tf.image.resize_images(net, 2 * tf.shape(net)[1:3], method=1)

            # -------------4x up-------------
            net = slim.conv2d(net, ngf, [3, 3], scope='conv2_1', stride=1)
            temp = net
            # ======residue blocks(2)======
            for i in range(2):
                net_ = slim.conv2d(net, ngf, [3, 3], scope='conv2_1/b%s/c1' % i, stride=1)
                net_ = slim.conv2d(net_, ngf, [3, 3], scope='conv2_1/b%s/c2' % i, stride=1)
                net_ = tf.add(net, net_)
                net = net_
            net = slim.conv2d(net, ngf, [3, 3], scope='conv2_2', stride=1)
            net = tf.add(temp, net)

            # ======subpixel upsampling(2)======
            if not nnup:
                net = conv2(net, 3, 4 * ngf, 1, scope='suppixelconv_2')
                net = pixelShuffler(net, scale=2)
                net = tf.nn.relu(net)
            else:
                net = tf.image.resize_images(net, 2 * tf.shape(net)[1:3], method=1)

            # net = slim.conv2d(net, ngf, [3, 3], scope='conv3_1', stride=1)  # or there will be grids
            # temp = net
            # # ======residue blocks(2)======
            # for i in range(2):
            #     net_ = slim.conv2d(net, ngf, [3, 3], scope='conv3_1/b%s/c1' % i, stride=1)
            #     net_ = slim.conv2d(net_, ngf, [3, 3], scope='conv3_1/b%s/c2' % i, stride=1)
            #     net_ = tf.add(net, net_)
            #     net = net_
            # net = slim.conv2d(net, ngf, [3, 3], scope='conv3_2', stride=1)
            # net = tf.add(temp, net)

            attention_map = slim.conv2d(net, n, [1, 1], scope='conv4_1', stride=1, activation_fn=None)  # [b, h_t, w_t, n]

        # apply softmax to attention map
        attention_map = tf.nn.softmax(attention_map, axis=-1)
        attention_map = tf.reshape(attention_map, [b * h_t * w_t, n])
        # attention_map = tf.reshape(attention_map, [b * h_t * w_t, n, 1])

        # Transfer psv to construct hr_syn
        psv_to_transfer = tf.reshape(psv_to_transfer, [b * h_t * w_t, c_t, n])  # c_t == 3 for image psv,
                                                                                # c_t == c for f_psv
        # f_hr_syn = tf.matmul(psv_to_transfer, attention_map)  # [bhw, c_t, 1]
        # f_hr_syn = my_matmul(psv_to_transfer, attention_map)
        f_hr_syn = []
        for i in range(c_t):
            psv_tt_ci = psv_to_transfer[:, i, :]  # [bhw, n]
            f_hr_syn_ci = tf.multiply(psv_tt_ci, attention_map)  # [bhw, n]
            f_hr_syn_ci = tf.reduce_sum(f_hr_syn_ci, axis=-1)  # [bhw]
            f_hr_syn.append(f_hr_syn_ci)
        f_hr_syn = tf.stack(f_hr_syn, axis=-1)

        f_hr_syn = tf.reshape(f_hr_syn, [b * h_t * w_t, c_t])
        f_hr_syn  = tf.reshape(f_hr_syn, [b, h_t, w_t, c_t])
        if c_t == 3:
            hr_syn = f_hr_syn
        else:
            # feature aggregation
            hr_syn = slim.conv2d(f_hr_syn, 3, [1, 1], stride=1, activation_fn=tf.nn.tanh, normalizer_fn=None,
                                 scope='color_pred')  # [b, h, w, 3]

        # reshape attention map
        attention_map = tf.reshape(attention_map, [b, h_t, w_t, n])

        if init_attention_map is not None:
            return f_hr_syn, hr_syn, attention_map, init_attention_map
        else:
            return f_hr_syn, hr_syn, attention_map

def guided_att_hr_syn_net(f_lr, f_psv_to_attention, guidance_volume, psv_to_transfer, lowres_psv_to_transfer=None,
                          lowres_pretrain=False, attention_flag=True, ngf=64, vscope='net', reuse_weights=False, nnup=False):
  """Network definition for hr_syn inference.

  Args:
    f_lr: input lr features [b, h, w, c]
    f_psv_to_attention: stack of psv features [b, h, w, c, 32] to generate attention
    guidance_volume: [lr, psv] [b, h_t, w_t, 3*(n+1)]
    psv_to_transfer: which psv format transferred to LR [b, h_t, w_t, c_t, n]
    lowres_psv_to_transfer: lowres net input [b, h, w, c_t, n]
    attention_flag: use attention or not
    lowres_pretrain: if pretrain lowres attention map or not
    ngf: number of features for the first conv layer
    vscope: variable scope
    reuse_weights: whether to reuse weights (for weight sharing)
    nnup: whether use nn_resize+conv
  Returns:
    pred: network output at the same spatial resolution as the inputs.
  """
  tf.logging.info('guided_att_hr_syn_net: ngf=%d' % ngf)
  def my_residue_block(midd, name, repeat=3, ngf_=64):
      midd = slim.conv2d(midd, ngf_, [3, 3], scope='%s/in' % name, stride=1)
      for j in range(repeat):
          midd_ = slim.conv2d(midd, ngf_, [3, 3], scope='%s/b%s/c1' % (name, j), stride=1)
          midd_ = slim.conv2d(midd_, ngf_, [3, 3], scope='%s/b%s/c2' % (name, j), stride=1)
          midd_ = slim.conv2d(midd_, ngf_, [3, 3], scope='%s/b%s/c3' % (name, j), stride=1)
          midd_ = tf.add(midd, midd_)
          midd = midd_
      midd = slim.conv2d(midd, ngf_, [3, 3], scope='%s/out' % name, stride=1)
      return midd

  with tf.variable_scope(vscope, reuse=reuse_weights):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], normalizer_fn=None):
        b, h, w, c, n = f_psv_to_attention.get_shape().as_list()
        _, h_t, w_t, c_t, _ = psv_to_transfer.get_shape().as_list()
        tf.logging.info('Num of channels of f_psv_to_attention = %d' % c)
        tf.logging.info('Num of channels of psv_to_transfer = %d' % c_t)

        if attention_flag:
            # Construction of f_lr_wave
            f_lr_wave = tf.reshape(f_lr, [b * h * w, c])
            f_lr_wave = tf.reshape(f_lr_wave, [b * h * w, 1, c])  # [bhw, 1, c]

            # Construction of f_psv_wave
            f_psv_to_attention = tf.reshape(f_psv_to_attention, [b * h * w, c, n])

            # Construction of A
            attention_map = tf.matmul(f_lr_wave, f_psv_to_attention)  # [bhw, 1, n]
            attention_map = tf.reshape(attention_map, [b * h * w, n])  # [bhw, n]

        else:
            f_psv_ = tf.transpose(f_psv_to_attention, [0, 1, 2, 4, 3])  # [b, h, w, n, c]
            f_psv_ = tf.reshape(f_psv_, [b, h, w, n * c])
            attention_map = slim.conv2d(tf.concat([f_lr, f_psv_], axis=3), ngf, [3, 3], scope='replace_attention', stride=1)

        if lowres_psv_to_transfer is not None:
            if attention_flag:
                attention_map = tf.reshape(attention_map, [b, h, w, n])  # [b, h, w, n]
                attention_map = slim.conv2d(attention_map, ngf, [3, 3], scope='attention_out_0', stride=1)
                attention_map = slim.conv2d(attention_map, ngf, [3, 3], scope='attention_out_1', stride=1)
                attention_map = slim.conv2d(attention_map, ngf, [3, 3], scope='attention_out_2', stride=1)
                attention_map = slim.conv2d(attention_map, n, [1, 1], scope='attention_out', stride=1, activation_fn=None)
            else:
                attention_map = slim.conv2d(attention_map, ngf, [3, 3], scope='replace_attention_1', stride=1)
                attention_map = slim.conv2d(attention_map, ngf, [3, 3], scope='replace_attention_2', stride=1)
                attention_map = slim.conv2d(attention_map, n, [1, 1], scope='replace_attention_out', stride=1, activation_fn=None)

            attention_map = tf.nn.softmax(attention_map, axis=-1)
            attention_map = tf.reshape(attention_map, [b * h * w, n])  # [bhw, n]

            # lowres hr synthesis
            lowres_psv_to_transfer = tf.reshape(lowres_psv_to_transfer, [b * h * w, c_t, n])
            f_hr_syn = []
            for i in range(c_t):
                psv_tt_ci = lowres_psv_to_transfer[:, i, :]  # [bhw, n]
                f_hr_syn_ci = tf.multiply(psv_tt_ci, attention_map)  # [bhw, n]
                f_hr_syn_ci = tf.reduce_sum(f_hr_syn_ci, axis=-1)  # [bhw]
                f_hr_syn.append(f_hr_syn_ci)
            init_syn = tf.stack(f_hr_syn, axis=-1)

            init_syn = tf.reshape(init_syn, [b * h * w, c_t])
            init_syn = tf.reshape(init_syn, [b, h, w, c_t])
        # else:
        #     attention_map = tf.nn.softmax(attention_map, axis=-1)

        # attention map decoder!
        attention_map = tf.reshape(attention_map, [b, h, w, n])  # [b, h, w, n]
        init_attention_map = attention_map

  # with tf.variable_scope('decoder', reuse=reuse_weights):
  #   with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], normalizer_fn=None):
        ################## Guidance Encoder ##################
        if guidance_volume is not None:
            guid_lv0 = my_residue_block(guidance_volume, name='guid_lv0_res', ngf_=ngf)
            guid_lv1 = slim.conv2d(guid_lv0, ngf, [3, 3], scope='guid_lv1_down', stride=2)
            guid_lv1 = my_residue_block(guid_lv1, name='guid_lv1_res', ngf_=ngf)
            guid_lv2 = slim.conv2d(guid_lv1, ngf, [3, 3], scope='guid_lv2_down', stride=2)
            guid_lv2 = my_residue_block(guid_lv2, name='guid_lv2_res', ngf_=ngf)

        #################### A upsampling ####################
        tf.logging.info('attention_decoder!!!!!!')
        # -------------level 2-------------
        # residue blocks
        a_lv2 = my_residue_block(attention_map, name='a_lv2_res0', ngf_=ngf)
        # concat -> R
        if guidance_volume is not None:
            a_lv2 = my_residue_block(tf.concat((a_lv2, guid_lv2), axis=-1), name='a_lv2_res_com2', ngf_=ngf)
        else:
            a_lv2 = my_residue_block(a_lv2, name='a_lv2_res_com2', ngf_=ngf)

        # -------------level 1-------------
        if not nnup:
            # subpixel upsampling
            a_lv1 = conv2(a_lv2, 3, 4 * ngf, 1, scope='a_lv1_sub')
            a_lv1 = pixelShuffler(a_lv1, scale=2)
            a_lv1 = tf.nn.relu(a_lv1)
        else:
            a_lv1 = tf.image.resize_images(a_lv2, 2 * tf.shape(a_lv2)[1:3], method=1)
        # concat -> R
        if guidance_volume is not None:
            a_lv1 = my_residue_block(tf.concat((a_lv1, guid_lv1), axis=-1), name='a_lv1_res_com1', ngf_=ngf)
        else:
            a_lv1 = my_residue_block(a_lv1, name='a_lv1_res_com1', ngf_=ngf)

        # -------------level 0-------------
        if not nnup:
            # subpixel upsampling
            a_lv0 = conv2(a_lv1, 3, 4 * ngf, 1, scope='a_lv0_sub')
            a_lv0 = pixelShuffler(a_lv0, scale=2)
            a_lv0 = tf.nn.relu(a_lv0)
        else:
            a_lv0 = tf.image.resize_images(a_lv1, 2 * tf.shape(a_lv1)[1:3], method=1)
        # concat -> R
        if guidance_volume is not None:
            a_lv0 = my_residue_block(tf.concat((a_lv0, guid_lv0), axis=-1), name='a_lv0_res_com0', ngf_=ngf)
        else:
            a_lv0 = my_residue_block(a_lv0, name='a_lv0_res_com0', ngf_=ngf)



        attention_map = slim.conv2d(a_lv0, n, [1, 1], scope='a_out1', stride=1, activation_fn=None)  # [b, h_t, w_t, n]

        # apply softmax to attention map
        attention_map = tf.nn.softmax(attention_map, axis=-1)
        attention_map = tf.reshape(attention_map, [b * h_t * w_t, n])

        # Transfer psv to construct hr_syn
        psv_to_transfer = tf.reshape(psv_to_transfer, [b * h_t * w_t, c_t, n])  # c_t == 3 for image psv,                                                                           # c_t == c for f_psv

        f_hr_syn = []
        for i in range(c_t):
            psv_tt_ci = psv_to_transfer[:, i, :]  # [bhw, n]
            f_hr_syn_ci = tf.multiply(psv_tt_ci, attention_map)  # [bhw, n]
            f_hr_syn_ci = tf.reduce_sum(f_hr_syn_ci, axis=-1)  # [bhw]
            f_hr_syn.append(f_hr_syn_ci)
        f_hr_syn = tf.stack(f_hr_syn, axis=-1)

        f_hr_syn = tf.reshape(f_hr_syn, [b * h_t * w_t, c_t])
        f_hr_syn  = tf.reshape(f_hr_syn, [b, h_t, w_t, c_t])
        if c_t == 3:
            hr_syn = f_hr_syn
        else:
            # feature aggregation
            hr_syn = slim.conv2d(f_hr_syn, 3, [1, 1], stride=1, activation_fn=tf.nn.tanh, normalizer_fn=None,
                                 scope='color_pred')  # [b, h, w, 3]

        # reshape attention map
        attention_map = tf.reshape(attention_map, [b, h_t, w_t, n])

        if lowres_psv_to_transfer is not None:
            if lowres_pretrain:
                return init_syn, init_attention_map
            else:
                return hr_syn, attention_map, init_syn, init_attention_map
        else:
            return hr_syn, attention_map, init_attention_map

def psv_guided_att_hr_syn_net(f_lr, f_psv_to_attention, psv_to_transfer, ngf=4, vscope='net', reuse_weights=False, nnup=False):
  """Network definition for hr_syn inference.

  Args:
    f_lr: input lr features [b, h, w, c]
    f_psv_to_attention: stack of psv features [b, h, w, c, 32] to generate attention
    psv_to_transfer: which psv format transferred to LR [b, h_t, w_t, c_t, n]
    ngf: number of features for the first conv layer
    vscope: variable scope
    reuse_weights: whether to reuse weights (for weight sharing)
    nnup: whether use nn_resize + conv
  Returns:
    pred: network output at the same spatial resolution as the inputs.
  """
  def my_residue_block(midd, name, repeat=3, ngf_=64):
      midd = slim.conv2d(midd, ngf_, [3, 3], scope='%s/in' % name, stride=1)
      for j in range(repeat):
          midd_ = slim.conv2d(midd, ngf_, [3, 3], scope='%s/b%s/c1' % (name, j), stride=1)
          midd_ = slim.conv2d(midd_, ngf_, [3, 3], scope='%s/b%s/c2' % (name, j), stride=1)
          midd_ = slim.conv2d(midd_, ngf_, [3, 3], scope='%s/b%s/c3' % (name, j), stride=1)
          midd_ = tf.add(midd, midd_)
          midd = midd_
      midd = slim.conv2d(midd, ngf_, [3, 3], scope='%s/out' % name, stride=1)
      return midd

  with tf.variable_scope(vscope, reuse=reuse_weights):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], normalizer_fn=None):
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

        # del f_lr
        # del f_lr_wave
        # del f_psv_to_attention

        ################## Guidance Encoder ##################
        guidance_volume = tf.transpose(psv_to_transfer, [0, 4, 1, 2, 3])
        guidance_volume = tf.reshape(guidance_volume, [b * n, h_t, w_t, c_t])
        guid_lv0 = my_residue_block(guidance_volume, name='guid_lv0_res', ngf_=ngf)
        guid_lv1 = slim.conv2d(guid_lv0, ngf, [3, 3], scope='guid_lv1_down', stride=2)
        guid_lv1 = my_residue_block(guid_lv1, name='guid_lv1_res', ngf_=ngf)
        guid_lv2 = slim.conv2d(guid_lv1, ngf, [3, 3], scope='guid_lv2_down', stride=2)
        guid_lv2 = my_residue_block(guid_lv2, name='guid_lv2_res', ngf_=ngf)


        #################### A upsampling ####################
        tf.logging.info('attention_decoder!!!!!!')
        attention_map = tf.transpose(attention_map, [0, 3, 1, 2])
        attention_map = tf.reshape(attention_map, [b * n, h, w])
        attention_map = tf.reshape(attention_map, [b * n, h, w, 1])
        # -------------level 2-------------
        # residue blocks
        a_lv2 = my_residue_block(attention_map, name='a_lv2_res0', ngf_=ngf)
        # concat -> R
        a_lv2 = my_residue_block(tf.concat((a_lv2, guid_lv2), axis=-1), name='a_lv2_res_com2', ngf_=ngf)

        # -------------level 1-------------
        if not nnup:
            # subpixel upsampling
            a_lv1 = conv2(a_lv2, 3, 4 * ngf, 1, scope='a_lv1_sub')
            a_lv1 = pixelShuffler(a_lv1, scale=2)
            a_lv1 = tf.nn.relu(a_lv1)
        else:
            a_lv1 = tf.image.resize_images(a_lv2, 2 * tf.shape(a_lv2)[1:3], method=1)
        # concat -> R
        a_lv1 = my_residue_block(tf.concat((a_lv1, guid_lv1), axis=-1), name='a_lv1_res_com1', ngf_=ngf)

        # -------------level 1-------------
        if not nnup:
            # subpixel upsampling
            a_lv0 = conv2(a_lv1, 3, 4 * ngf, 1, scope='a_lv0_sub')
            a_lv0 = pixelShuffler(a_lv0, scale=2)
            a_lv0 = tf.nn.relu(a_lv0)
        else:
            a_lv0 = tf.image.resize_images(a_lv1, 2 * tf.shape(a_lv1)[1:3], method=1)
        # concat -> R
        a_lv0 = my_residue_block(tf.concat((a_lv0, guid_lv0), axis=-1), name='a_lv0_res_com0', ngf_=ngf)

        attention_map = slim.conv2d(a_lv0, 1, [1, 1], scope='a_out1', stride=1, activation_fn=None)  # [b * n, h_t, w_t, 1]

        attention_map = tf.reshape(attention_map, [b * n, h_t, w_t])
        attention_map = tf.reshape(attention_map, [b, n, h_t, w_t])
        attention_map = tf.transpose(attention_map, [0, 2, 3, 1])  # [b, h_t, w_t, n]

        # apply softmax to attention map
        attention_map = tf.nn.softmax(attention_map, axis=-1)
        attention_map = tf.reshape(attention_map, [b * h_t * w_t, n])

        # Transfer psv to construct hr_syn
        psv_to_transfer = tf.reshape(psv_to_transfer, [b * h_t * w_t, c_t, n])  # c_t == 3 for image psv,                                                                           # c_t == c for f_psv

        f_hr_syn = []
        for i in range(c_t):
            psv_tt_ci = psv_to_transfer[:, i, :]  # [bhw, n]
            f_hr_syn_ci = tf.multiply(psv_tt_ci, attention_map)  # [bhw, n]
            f_hr_syn_ci = tf.reduce_sum(f_hr_syn_ci, axis=-1)  # [bhw]
            f_hr_syn.append(f_hr_syn_ci)
        f_hr_syn = tf.stack(f_hr_syn, axis=-1)

        f_hr_syn = tf.reshape(f_hr_syn, [b * h_t * w_t, c_t])
        f_hr_syn  = tf.reshape(f_hr_syn, [b, h_t, w_t, c_t])
        if c_t == 3:
            hr_syn = f_hr_syn
        else:
            # feature aggregation
            hr_syn = slim.conv2d(f_hr_syn, 3, [1, 1], stride=1, activation_fn=tf.nn.tanh, normalizer_fn=None,
                                 scope='color_pred')  # [b, h, w, 3]

        # reshape attention map
        attention_map = tf.reshape(attention_map, [b, h_t, w_t, n])

        return hr_syn, attention_map, init_attention_map


def psv_guided_att_decoder(attention_lr_slice, psv_guidance_slice, ngf=4, vscope='net'): #, reuse_weights=tf.AUTO_REUSE):
  """Network definition for hr_syn inference.

  Args:
    attention_lr_slice: input lr attention with shape [b, h, w, 1]
    psv_guidance_slice: which psv format transferred to LR [b, h_t, w_t, 3]
    ngf: number of features for the first conv layer
    vscope: variable scope
    reuse_weights: whether to reuse weights (for weight sharing)
  Returns:
    pred: network output at the same spatial resolution as the psv_guidance_slice. [b, h_t, w_t, 1]
  """
  with tf.variable_scope(vscope, reuse=tf.AUTO_REUSE):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], normalizer_fn=None):
        def my_residue_block(midd, name, repeat=3, ngf_=64):
            midd = slim.conv2d(midd, ngf_, [3, 3], scope='%s/in' % name, stride=1)
            for j in range(repeat):
                midd_ = slim.conv2d(midd, ngf_, [3, 3], scope='%s/b%s/c1' % (name, j), stride=1)
                midd_ = slim.conv2d(midd_, ngf_, [3, 3], scope='%s/b%s/c2' % (name, j), stride=1)
                midd_ = slim.conv2d(midd_, ngf_, [3, 3], scope='%s/b%s/c3' % (name, j), stride=1)
                midd_ = tf.add(midd, midd_)
                midd = midd_
            midd = slim.conv2d(midd, ngf_, [3, 3], scope='%s/out' % name, stride=1)
            return midd
        b, h, w, _ = attention_lr_slice.get_shape().as_list()
        _, h_t, w_t, _ = psv_guidance_slice.get_shape().as_list()
        tf.logging.info('Spatial shape of attention_lr_slice = (%d, %d)' % (h, w))
        tf.logging.info('Spatial shape of psv_guidance_slice = (%d, %d)' % (h_t, w_t))

        ################## Guidance Encoder ##################
        guid_lv0 = my_residue_block(psv_guidance_slice, name='guid_lv0_res', ngf_=ngf)  # [b, h_t, w_t, ngf]
        guid_lv1 = slim.conv2d(guid_lv0, ngf, [3, 3], scope='guid_lv1_down', stride=2)
        guid_lv1 = my_residue_block(guid_lv1, name='guid_lv1_res', ngf_=ngf)
        guid_lv2 = slim.conv2d(guid_lv1, ngf, [3, 3], scope='guid_lv2_down', stride=2)
        guid_lv2 = my_residue_block(guid_lv2, name='guid_lv2_res', ngf_=ngf)


        #################### A upsampling ####################
        tf.logging.info('attention_decoder!!!!!!')
        # -------------level 2-------------
        # residue blocks
        a_lv2 = my_residue_block(attention_lr_slice, name='a_lv2_res0', ngf_=ngf)  # [b, h, w, ngf]
        # concat -> R
        a_lv2 = my_residue_block(tf.concat((a_lv2, guid_lv2), axis=-1), name='a_lv2_res_com2', ngf_=ngf)

        # -------------level 1-------------
        # subpixel upsampling
        a_lv1 = conv2(a_lv2, 3, 4 * ngf, 1, scope='a_lv1_sub')
        a_lv1 = pixelShuffler(a_lv1, scale=2)
        a_lv1 = tf.nn.relu(a_lv1)
        # concat -> R
        a_lv1 = my_residue_block(tf.concat((a_lv1, guid_lv1), axis=-1), name='a_lv1_res_com1', ngf_=ngf)

        # -------------level 1-------------
        # subpixel upsampling
        a_lv0 = conv2(a_lv1, 3, 4 * ngf, 1, scope='a_lv0_sub')
        a_lv0 = pixelShuffler(a_lv0, scale=2)
        a_lv0 = tf.nn.relu(a_lv0)
        # concat -> R
        a_lv0 = my_residue_block(tf.concat((a_lv0, guid_lv0), axis=-1), name='a_lv0_res_com0', ngf_=ngf)

        attention_map = slim.conv2d(a_lv0, 1, [1, 1], scope='a_out1', stride=1, activation_fn=None)  # [b, h_t, w_t, 1]

        return attention_map

# def res_fuse_net(f_lr, f_hr_syn, ngf=64, vscope='fuse_part', reuse_weights=False):
def res_fuse_net(f_lr, f_hr_syn, ngf=32, vscope='fuse_part', reuse_weights=False):
  """Network definition for lr & hr_syn fusion.

  Args:
    f_lr: feature of input lr image (bicubic-upsampled)
    f_hr_syn: feature of 'hR_syn' image from hr_syn_net
    ngf: number of features for the first conv layer
    vscope: variable scope
    reuse_weights: whether to reuse weights (for weight sharing)
  Returns:
    out: network output at the same spatial resolution as the inputs.
  """
  with tf.variable_scope(vscope, reuse=reuse_weights):
    # with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], normalizer_fn=slim.layer_norm):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], normalizer_fn=None):
        # ~~~~ lr_up and syn_hr fusing module ~~~~ #
        f_lr = slim.conv2d(f_lr, ngf, [3, 3], scope='in_f_lr', stride=1)
        f_hr_syn = slim.conv2d(f_hr_syn, ngf, [3, 3], scope='in_f_hr', stride=1)
        net = tf.concat([f_lr, f_hr_syn], axis=3)
        net = slim.conv2d(net, ngf, [3, 3], scope='conv1_1', stride=1)
        # ======residue blocks======
        # for i in range(8):
        for i in range(6):
            net_ = slim.conv2d(net, ngf, [3, 3], scope='conv2_1/b%s/c1' % i, stride=1)
            net_ = slim.conv2d(net_, ngf, [3, 3], scope='conv2_1/b%s/c2' % i, stride=1)
            net_ = tf.add(net, net_)
            net = net_
        net = slim.conv2d(net, ngf, [3, 3], scope='conv3_2', stride=1)
        net = tf.add(f_lr, net)

        net = slim.conv2d(net, ngf, [3, 3], scope='conv4_1', stride=1)

        out = slim.conv2d(
            net,
            3, [1, 1],
            stride=1,
            activation_fn=tf.nn.tanh,
            normalizer_fn=None,
            scope='sr_output')

        return out



