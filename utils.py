# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 20:25:47 2018
@author: tyty
@e-mail: bravotty@protonmail.com
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import tflearn
import numpy as np
from skimage.measure import compare_ssim, compare_psnr
import tensorflow.contrib.slim as slim

xy = 256

def im2uint8(x):
    if x.__class__ == tf.Tensor:
        return tf.cast(tf.clip_by_value(x, 0.0, 1.0) * 255.0, tf.uint8)
    else:
        t = np.clip(x, 0.0, 1.0) * 255.0
        return t.astype(np.uint8)

def conv_torque(input):

    inputShape = list(input.shape[0:3])
    hSqrt2 = 0.5 * 2 ** 0.5
    direc = [    [  1,     0],
            [hSqrt2 ,hSqrt2],
             [  0,     1],
         [-hSqrt2 ,hSqrt2],
         [      -1,    0],
         [-hSqrt2 ,-hSqrt2],
         [      0,     -1],
         [hSqrt2 ,-hSqrt2]]
    gradientOriy = tf.zeros(inputShape,dtype=tf.float32)
    gradientOrix = tf.zeros(inputShape,dtype=tf.float32)
    AllOnes = tf.ones(inputShape,dtype=tf.float32)
    for i in range(input.shape[3]):
        # print(direc[i][1])
         gradientOriy += AllOnes*input[:,:,:,i] * direc[i][0]
         gradientOrix += AllOnes*input[:,:,:,i] * direc[i][1]
         # if i == 7:
         #     return gradientOrix

    # print(gradientOri)

    gradientOriy = tf.expand_dims(gradientOriy,axis=3)

    gradientOrix = tf.expand_dims(gradientOrix,axis=3)

    tfgradientOriy = gradientOriy
    tfgradientOrix = gradientOrix

    # filterx = np.tile(filterx,[1,1,1,10])
    # xweights_data = np.tile(xweights_data,[1,channeln,1,channeln])
    # xweights_data = np.reshape(xweights_data,[1, 2, channeln, channeln])
    # print(np.expand_dims(np.expand_dims(np.tile(np.expand_dims(np.linspace(-9.5, 9.5, 20), axis=1), [1, 20]),axis=2),axis=3))
    tffiltery20 = tf.constant(np.expand_dims(np.expand_dims(np.tile(np.expand_dims(np.linspace(9.5, -9.5, 20), axis=1), [1, 20]),axis=2),axis=3), tf.float32)
    tffilterx20 = tf.constant(np.expand_dims(np.expand_dims(np.tile(np.linspace(9.5, -9.5, 20), [20,1]), axis=2), axis=3), tf.float32)
    xresult20 = tf.nn.conv2d(tfgradientOrix, tffilterx20, strides=[1, 1, 1, 1], padding='SAME')
    yresult20 = tf.nn.conv2d(tfgradientOriy, tffiltery20, strides=[1, 1, 1, 1], padding='SAME')
    result20 = (xresult20 + yresult20)/20**2 /20**2

    tffiltery25 = tf.constant(np.expand_dims(np.expand_dims(np.tile(np.expand_dims(np.linspace(12, -12, 25), axis=1), [1, 25]),axis=2),axis=3), tf.float32)
    tffilterx25 = tf.constant(np.expand_dims(np.expand_dims(np.tile(np.linspace(12, -12, 25), [25,1]), axis=2), axis=3), tf.float32)
    xresult25 = tf.nn.conv2d(tfgradientOrix, tffilterx25, strides=[1, 1, 1, 1], padding='SAME')
    yresult25 = tf.nn.conv2d(tfgradientOriy, tffiltery25, strides=[1, 1, 1, 1], padding='SAME')
    result25 = (xresult25 + yresult25)/25**2/25**2

    tffiltery30 = tf.constant(np.expand_dims(np.expand_dims(np.tile(np.expand_dims(np.linspace(14.5, -14.5, 30), axis=1), [1, 30]),axis=2),axis=3), tf.float32)
    tffilterx30 = tf.constant(np.expand_dims(np.expand_dims(np.tile(np.linspace(14.5, -14.5, 30), [30,1]), axis=2), axis=3), tf.float32)
    xresult30 = tf.nn.conv2d(tfgradientOrix, tffilterx30, strides=[1, 1, 1, 1], padding='SAME')
    yresult30 = tf.nn.conv2d(tfgradientOriy, tffiltery30, strides=[1, 1, 1, 1], padding='SAME')
    result30 = (xresult30 + yresult30)/30**2/30**2

    tffiltery35 = tf.constant(np.expand_dims(np.expand_dims(np.tile(np.expand_dims(np.linspace(17, -17, 35), axis=1), [1, 35]),axis=2),axis=3), tf.float32)
    tffilterx35 = tf.constant(np.expand_dims(np.expand_dims(np.tile(np.linspace(17, -17, 35), [35,1]), axis=2), axis=3), tf.float32)
    xresult35 = tf.nn.conv2d(tfgradientOrix, tffilterx35, strides=[1, 1, 1, 1], padding='SAME')
    yresult35 = tf.nn.conv2d(tfgradientOriy, tffiltery35, strides=[1, 1, 1, 1], padding='SAME')
    result35 = (xresult35 + yresult35)/35**2/35**2

    tffiltery40 = tf.constant(np.expand_dims(np.expand_dims(np.tile(np.expand_dims(np.linspace(19.5, -19.5, 40), axis=1), [1, 40]),axis=2),axis=3), tf.float32)
    tffilterx40 = tf.constant(np.expand_dims(np.expand_dims(np.tile(np.linspace(19.5, -19.5, 40), [40,1]), axis=2), axis=3), tf.float32)
    xresult40 = tf.nn.conv2d(tfgradientOrix, tffilterx40, strides=[1, 1, 1, 1], padding='SAME')
    yresult40 = tf.nn.conv2d(tfgradientOriy, tffiltery40, strides=[1, 1, 1, 1], padding='SAME')
    result40 = (xresult40 + yresult40)/40**2/40**2



    # print(result41.shape)
    result = tf.concat([result20,result25,result30,result35,result40],axis=3)
    result = tf.maximum(result,0)
    Maxnpresult = tf.reduce_max(result,axis=3,keep_dims=True)
    return Maxnpresult


def ResnetBlock_att(x, dim, ksize, att,scope='rb'):
    with tf.variable_scope(scope):
        net1= slim.conv2d(x, dim, [ksize, ksize], scope='conv1_1')
        net1 = slim.conv2d(net1, dim, [ksize, ksize], activation_fn=None, scope='conv1_2')
        net1, feature, scale = channel_attention(net1, 'ch_at', 2.0)
        att = tf.reduce_mean(att,axis=3 ,keep_dims=True)
        att = tf.tile(att,[1,1,1,dim])
        net2 = att * net1
        return tf.nn.relu( net1 + net2 + x ), feature, scale

def channel_attention(input_feature, name, ratio=8):
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)

    with tf.variable_scope(name):
        channel = input_feature.get_shape()[-1]
        avg_pool = tf.reduce_mean(input_feature, axis=[1, 2], keep_dims=True)

        assert avg_pool.get_shape()[1:] == (1, 1, channel)
        avg_pool = tf.layers.dense(inputs=avg_pool,
                                   units=channel // ratio,
                                   activation=tf.nn.relu,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer,
                                   name='mlp_0',
                                   reuse=None)
        assert avg_pool.get_shape()[1:] == (1, 1, channel // ratio)
        avg_pool = tf.layers.dense(inputs=avg_pool,
                                   units=channel,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer,
                                   name='mlp_1',
                                   reuse=None)
        assert avg_pool.get_shape()[1:] == (1, 1, channel)

        max_pool = tf.reduce_max(input_feature, axis=[1, 2], keep_dims=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel)
        max_pool = tf.layers.dense(inputs=max_pool,
                                   units=channel // ratio,
                                   activation=tf.nn.relu,
                                   name='mlp_0',
                                   reuse=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel // ratio)
        max_pool = tf.layers.dense(inputs=max_pool,
                                   units=channel,
                                   name='mlp_1',
                                   reuse=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel)
        scale = tf.sigmoid(avg_pool + max_pool, 'sigmoid')

    return input_feature * scale, input_feature, scale



def fc_layer(input_, output_dim, initializer = None, activation='linear', name=None):
    if initializer == None: initializer = tf.contrib.layers.xavier_initializer()
    shape = input_.get_shape().as_list()
    with tf.variable_scope(name or "Linear", reuse=tf.AUTO_REUSE) as scope:
        if len(shape) > 2 : input_ = tf.layers.flatten(input_)
        shape = input_.get_shape().as_list()
        w = tf.get_variable("fc_w", [shape[1], output_dim], dtype=tf.float32, initializer = initializer)
        b = tf.get_variable("fc_b", [output_dim], initializer = tf.constant_initializer(0.0))
        result = tf.matmul(input_, w) + b
        if activation == 'linear':
            return result
        elif activation == 'relu':
            return tf.nn.relu(result)
        elif activation == 'sigmoid':
            return tf.nn.sigmoid(result)
        elif activation == 'tanh':
            return tf.nn.tanh(result)

def Linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  shape = input_.get_shape().as_list()
  with tf.variable_scope(scope or "Linear"):
    try:
      matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    except ValueError as err:
        msg = "NOTE: Usually, this is due to an issue with the image dimensions.  Did you correctly set '--crop' or '--input_height' or '--output_height'?"
        err.args = err.args + (msg,)
        raise
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias

def sigmoid_cross_entropy_with_logits(x, y):
    try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
    except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)



def calc_psnr(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_psnr(im1_y, im2_y)

def calc_ssim(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_ssim(im1_y, im2_y)

def cal_ssim_psnr(outs, y_s, batch_s):
    psnr = 0
    ssim = 0
    for i in range(batch_s):
        psnr += calc_psnr(y_s[i], outs[i])
        ssim += calc_ssim(outs[i], y_s[i])
    ssim /= batch_s
    psnr /= batch_s
    print ('ssim value ', ssim)
    print ('psnr value ', psnr)
    return ssim


def psnrrgb(img1, img2):
    img1 = np.clip(img1, 0, 255)
    img2 = np.clip(img2, 0, 255)
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    if(len(img1.shape) == 2):
        m, n = img1.shape
        k = 1
    elif (len(img1.shape) == 3):
        m, n, k = img1.shape

    B = 8
    diff = np.power(img1 - img2, 2)
    MAX = 2**B - 1
    MSE = np.sum(diff) / (m * n * k)
    sqrt_MSE = np.sqrt(MSE)
    PSNR = 20 * np.log10(MAX / sqrt_MSE)

    return PSNR

def align_to_four(img):
    # print ('before alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    # align to four
    a_row = int(img.shape[0] / 16) * 16
    a_col = int(img.shape[1] / 16) * 16
    img = img[0:a_row, 0:a_col]
    # print ('after alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    return img

