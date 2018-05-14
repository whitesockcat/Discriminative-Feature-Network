
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.slim.python.slim.nets import resnet_utils



import math
_HEIGHT = 513
_WIDTH = 513


def RRB(inputs, is_training = True ,depth=512):
    """Refinement Residual Block.

      Args:
        inputs: A tensor of size [batch, height, width, channels].

      Returns:
         output.
      """
    conv_1x1 = layers_lib.conv2d(inputs, depth, [1, 1], stride=1)
    conv_3x3_1 = resnet_utils.conv2d_same(conv_1x1, depth, 3, stride=1)
    batch_norm = tf.layers.batch_normalization(conv_3x3_1, training=is_training)
    relu1 = tf.nn.relu(batch_norm)
    conv_3x3_2 = resnet_utils.conv2d_same(relu1, depth, 3, stride=1)
    sum = tf.add_n([conv_1x1,conv_3x3_2])
    relu2 = tf.nn.relu(sum)
    return relu2

def CAB(inputs,inputs_up, sizediv,depth=512):
    size_inputs = tf.shape(inputs)[1:3]
    inputs_up = tf.image.resize_bilinear(inputs_up, size_inputs)
    concat = tf.concat([inputs, inputs_up], axis=3)
    # global average pooling
    global_pool = tf.reduce_mean(concat, [1, 2], name='global_average_pooling', keepdims=True)
    conv_1x1 = layers_lib.conv2d(global_pool, depth, [1, 1], stride=1)
    relu = tf.nn.relu(conv_1x1)
    conv_1x1 = layers_lib.conv2d(relu, depth, [1, 1], stride=1)
    sigmoid = tf.sigmoid(conv_1x1)
    concat1 = sigmoid
    height = math.ceil(_HEIGHT/sizediv)
    width = math.ceil(_WIDTH/sizediv)
    for i in range(height-1):
        concat1 = tf.concat([concat1,sigmoid], axis= 1)
    concat2 = concat1
    for i in range(width-1):
        concat2 = tf.concat([concat2,concat1], axis= 2)
    inputs = tf.multiply(inputs,concat2)
    sum = tf.add_n([inputs,inputs_up])
    return sum

def dfn(inputs,base_architecture = 'resnet_v2_101',output_stride = 16, is_training = True, batch_norm_decay = _BATCH_NORM_DECAY):
    if base_architecture == 'resnet_v2_50':
        base_model = resnet_v2.resnet_v2_50
    else:
        base_model = resnet_v2.resnet_v2_101
    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
      logits, end_points = base_model(inputs,
                                      num_classes=None,
                                      is_training=is_training,
                                      global_pool=False,
                                      output_stride=output_stride)

    net_4 = end_points[base_architecture + '/block4']
    net_3 = end_points[base_architecture + '/block3']
    net_2 = end_points[base_architecture + '/block2']
    net_1 = end_points[base_architecture + '/block1']

    inputs_size = tf.shape(inputs)[1:3]
    size_block1 = tf.shape(net_1)[1:3]

    net_1_rrb = RRB(net_1)
    net_2_rrb = RRB(net_2)
    net_3_rrb = RRB(net_3)
    net_4_rrb = RRB(net_4)

    net_2_rrb = tf.image.resize_bilinear(net_2_rrb, size_block1, name='upsample')
    net_3_rrb = tf.image.resize_bilinear(net_3_rrb, size_block1, name='upsample')
    net_4_rrb = tf.image.resize_bilinear(net_4_rrb, size_block1, name='upsample')

    net_1_sum = net_1_rrb
    net_2_sum = tf.add_n([net_1_sum, net_2_rrb])
    net_3_sum = tf.add_n([net_2_sum, net_3_rrb])
    net_4_sum = tf.add_n([net_3_sum, net_4_rrb])

    border_network = tf.image.resize_bilinear(net_4_sum, inputs_size, name='upsample')

    global_pool = tf.reduce_mean(net_4, [1, 2], name='global_average_pooling', keepdims=True)
    global_pool = layers_lib.conv2d(global_pool, 512, [1, 1], stride=1)

    net_1_rrb = RRB(net_1)
    net_2_rrb = RRB(net_2)
    net_3_rrb = RRB(net_3)
    net_4_rrb = RRB(net_4)

    net_4_out = RRB(CAB(net_4_rrb, global_pool, sizediv=16))
    net_3_out = RRB(CAB(net_3_rrb, net_4_out, sizediv=16))
    net_2_out = RRB(CAB(net_2_rrb, net_3_out, sizediv=16))
    net_1_out = RRB(CAB(net_1_rrb, net_2_out, sizediv=8))

    smooth_network = tf.image.resize_bilinear(net_1_out, inputs_size, name='upsample')

    return smooth_network,border_network

