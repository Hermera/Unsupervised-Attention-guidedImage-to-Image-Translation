import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import (BatchNorm2d, Conv2d, Dense, Flatten, Input, DeConv2d, Lambda, \
                                LocalResponseNorm, MaxPool2d, Elementwise, InstanceNorm2d)
from tensorlayer.models import Model
from data import flags

ndf = 64

def discriminator(inputdisc, mask, transition_rate, donorm, name="discriminator"):
    with tf.variable_scope(name):
        mask = tf.cast(tf.greater_equal(mask, transition_rate), tf.float32)
        inputdisc = tf.multiply(inputdisc, mask)
        f = 4
        padw = 2
        lrelu = lambda x: tl.act.lrelu(x, 0.2)

        pad_input = tf.pad(inputdisc, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")

        o_c1 = Conv2d(
            n_filter=ndf,
            filter_size=(f, f),
            strides=(2, 2),
            padding="VALID",
            act=None,
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(0.0)
        )(pad_input)
        o_c1 = InstanceNorm2d(act=lrelu)(o_c1)

        pad_o_c1 = tf.pad(o_c1, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
        
        o_c2 = Conv2d(
            n_filter=ndf * 2,
            filter_size=(f, f),
            strides=(2, 2),
            padding="VALID",
            act=None,
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(0.0)
        )(pad_o_c1)
        o_c2 = InstanceNorm2d(act=lrelu)(o_c2)

        pad_o_c2 = tf.pad(o_c2, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
        
        o_c3 = Conv2d(
            n_filter=ndf * 4,
            filter_size=(f, f),
            strides=(2, 2),
            padding="VALID",
            act=None,
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(0.0)
        )(pad_o_c2)
        o_c3 = InstanceNorm2d(act=lrelu)(o_c3)

        pad_o_c3 = tf.pad(o_c3, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
        
        o_c4 = Conv2d(
            n_filter=ndf * 8,
            filter_size=(f, f),
            strides=(1, 1),
            padding="VALID",
            act=None,
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(0.0)
        )(pad_o_c3)
        o_c4 = InstanceNorm2d(act=lrelu)(o_c4)

        pad_o_c4 = tf.pad(o_c4, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
        
        o_c5 = Conv2d(
            n_filter=1,
            filter_size=(f, f),
            strides=(1, 1),
            padding="VALID",
            act=lrelu,
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(0.0)
        )(pad_o_c4)

        return o_c5