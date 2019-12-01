import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import (BatchNorm2d, Conv2d, Dense, Flatten, Input, DeConv2d, Lambda, \
                                LocalResponseNorm, MaxPool2d, Elementwise, InstanceNorm2d)
from tensorlayer.models import Model
from data import flags

IMG_CHANNELS = 3

ngf = 32
ndf = 64

def build_generator_9blocks(inputgen, name="generator", skip = False):
    with tf.variable_scope(name):
        f = 7
        ks = 3
        padding  = "CONSTANT"
        lrelu = lambda x: tl.act.lrelu(x, 0.2)
        inputgen = tf.pad(inputgen, [[0, 0], [ks, ks], [ks, ks], [0, 0]], padding)

        o_c1 = Conv2d(
            n_filter=ngf,
            filter_size=(f, f),
            strides=(1, 1),
            padding="VALID",
            act=None,
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(0.0)
        )(inputgen)
        o_c1 = InstanceNorm2d(act=tf.nn.relu)(o_c1)

        o_c2 = Conv2d(
            n_filter=ngf * 2,
            filter_size=(ks, ks),
            strides=(2, 2),
            padding="SAME",
            act=None,
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(0.0)
        )(o_c1)
        o_c2 = InstanceNorm2d(act=tf.nn.relu)(o_c2)

        o_c3 = Conv2d(
            n_filter=ngf * 4,
            filter_size=(ks, ks),
            strides=(2, 2),
            padding="SAME",
            act=None,
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(0.0)
        )(o_c2)
        o_c3 = InstanceNorm2d(act=tf.nn.relu)(o_c3)

        o_r1 = build_resnet_block(o_c3, ngf * 4, "r1", padding)
        o_r2 = build_resnet_block(o_r1, ngf * 4, "r2", padding)
        o_r3 = build_resnet_block(o_r2, ngf * 4, "r3", padding)
        o_r4 = build_resnet_block(o_r3, ngf * 4, "r4", padding)
        o_r5 = build_resnet_block(o_r4, ngf * 4, "r5", padding)
        o_r6 = build_resnet_block(o_r5, ngf * 4, "r6", padding)
        o_r7 = build_resnet_block(o_r6, ngf * 4, "r7", padding)
        o_r8 = build_resnet_block(o_r7, ngf * 4, "r8", padding)
        o_r9 = build_resnet_block(o_r8, ngf * 4, "r9", padding)

        o_c4 = DeConv2d(
            n_filter= ngf * 2,
            filter_size=(ks, ks),
            strides=(2, 2),
            padding="SAME",
            act=None,
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(0.0)
        )(o_r9)
        o_c4 = InstanceNorm2d(act=tf.nn.relu)(o_c4)

        o_c5 = DeConv2d(
            n_filter= ngf,
            filter_size=(ks, ks),
            strides=(2, 2),
            padding="SAME",
            act=None,
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(0.0)
        )(o_c4)
        o_c5 = InstanceNorm2d(act=tf.nn.relu)(o_c5)

        o_c6 = Conv2d(
            n_filter=IMG_CHANNELS,
            filter_size=(f, f),
            strides=(1, 1),
            padding="SAME",
            act=None,
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(0.0)
        )(o_c5)

        if skip is True:
            out_gen = tf.nn.tanh(inputgen + o_c6, name="t1")
        else:
            out_gen = tf.nn.tanh(o_c6, "t1")
        
        return out_gen

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