import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import (BatchNorm2d, Conv2d, Dense, Flatten, Input, DeConv2d, Lambda, \
                                LocalResponseNorm, MaxPool2d, Elementwise, InstanceNorm2d, PadLayer, Lambda, InputLayer, UpSampling2d, Concat)
from tensorlayer.models import Model
import pdb

IMG_CHANNELS = 3
IMG_WIDTH = 256
IMG_HEIGHT = 256

ngf = 32
ndf = 64

"""
问题1. variable_scope的问题. 需要有名字，main里需要名字来区分哪些参数是哪个网络的，再给对应网络加梯度。但tensorflow2.x没有variable_scope
如果用tf 1.x可以， tf 2.x用compat.v1.variable_scope可以.
问题2. padding, multiply, +, tf.nn.relu, tf.nn.sigmoid, tf.concat 全部需要变成tensorlayer, 无论1.x还是2.x
Concat, padding有现成的，multiply, add可以用ElementwiseLayer, tf.nn.relu和sigmoid等要用lambda layer
问题3. tf.images.resize_images改为UpSampling2d
可能还有别的bug
"""

"""
目前跟问题有关的函数：
1. tf.pad 用 tensorlayer.layers.PadLayer 替换
2. nn.relu 等用在作激活函数似乎不用替换，单独的激活函数要用Lambda包一层
3. truncated_normal_initializer 可能需要用 tf.initializers.TruncatedNormal 替换
wby's comment（所以不保证正确性:)）
"""


def build_model(skip=False):
    g_A_ae = autoenc_upsample(name="g_A_ae")
    g_B_ae = autoenc_upsample(name="g_B_ae")
    d_A = discriminator(name="d_A")
    d_B = discriminator(name="d_B")
    g_A = build_generator_9blocks("g_A", skip)
    g_B = build_generator_9blocks("g_B", skip)
    return (g_A_ae, g_B_ae, d_A, d_B, g_A, g_B)


def get_outputs(inputs, nets):

    images_a = inputs['images_a']
    images_b = inputs['images_b']
    fake_pool_a = inputs['fake_pool_a']
    fake_pool_b = inputs['fake_pool_b']
    fake_pool_a_mask = inputs['fake_pool_a_mask']
    fake_pool_b_mask = inputs['fake_pool_b_mask']
    transition_rate = inputs['transition_rate']
    donorm = inputs['donorm']

    g_A_ae, g_B_ae, d_A, d_B, g_A, g_B = nets

    mask_a = g_A_ae(images_a)
    mask_b = g_B_ae(images_b)

    mask_a = tf.concat([mask_a] * 3, axis=3)
    mask_b = tf.concat([mask_b] * 3, axis=3)

    mask_a_on_a = tf.multiply(images_a, mask_a)
    mask_b_on_b = tf.multiply(images_b, mask_b)

    prob_real_a_is_real = d_A([images_a, mask_a, transition_rate, donorm])
    prob_real_b_is_real = d_B([images_b, mask_b, transition_rate, donorm])


    fake_images_b_from_g = g_A(images_a)
    #pdb.set_trace()
    fake_images_b = tf.multiply(fake_images_b_from_g, mask_a) + tf.multiply(images_a, 1-mask_a)
    #pdb.set_trace()

    fake_images_a_from_g = g_B(images_b)

    fake_images_a = tf.multiply(fake_images_a_from_g, mask_b) + tf.multiply(images_b, 1-mask_b)

    prob_fake_a_is_real = d_A([fake_images_a, mask_b, transition_rate, donorm])
    prob_fake_b_is_real = d_B([fake_images_b, mask_a, transition_rate, donorm])

    mask_acycle = g_A_ae(fake_images_a)
    mask_bcycle = g_B_ae(fake_images_b)
    mask_bcycle = tf.concat([mask_bcycle] * 3, axis=3)
    mask_acycle = tf.concat([mask_acycle] * 3, axis=3)

    mask_acycle_on_fakeA = tf.multiply(fake_images_a, mask_acycle)
    mask_bcycle_on_fakeB = tf.multiply(fake_images_b, mask_bcycle)

    cycle_images_a_from_g = g_B(fake_images_b)
    cycle_images_b_from_g = g_A(fake_images_a)

    cycle_images_a = tf.multiply(cycle_images_a_from_g,
                                mask_bcycle) + tf.multiply(fake_images_b, 1 - mask_bcycle)

    cycle_images_b = tf.multiply(cycle_images_b_from_g,
                                mask_acycle) + tf.multiply(fake_images_a, 1 - mask_acycle)


    prob_fake_pool_a_is_real = d_A([fake_pool_a, fake_pool_a_mask, transition_rate, donorm])
    prob_fake_pool_b_is_real = d_B([fake_pool_b, fake_pool_b_mask, transition_rate, donorm])

    return {
        'prob_real_a_is_real': prob_real_a_is_real,
        'prob_real_b_is_real': prob_real_b_is_real,
        'prob_fake_a_is_real': prob_fake_a_is_real,
        'prob_fake_b_is_real': prob_fake_b_is_real,
        'prob_fake_pool_a_is_real': prob_fake_pool_a_is_real,
        'prob_fake_pool_b_is_real': prob_fake_pool_b_is_real,
        'cycle_images_a': cycle_images_a,
        'cycle_images_b': cycle_images_b,
        'fake_images_a': fake_images_a,
        'fake_images_b': fake_images_b,
        'masked_ims': tf.concat([mask_a_on_a, mask_b_on_b, mask_acycle_on_fakeA, mask_bcycle_on_fakeB], axis=0),
        'masks': tf.concat([mask_a, mask_b, mask_acycle, mask_bcycle], axis=0),
        'masked_gen_ims' : tf.concat([fake_images_b_from_g, fake_images_a_from_g, cycle_images_a_from_g, cycle_images_b_from_g], axis=0),
        'mask_tmp' : mask_a,
    }


def upsamplingDeconv(inputconv, size, name):
    size_h = size[0] / int(inputconv.get_shape()[1])
    size_w = size[1] / int(inputconv.get_shape()[2])
    size = (int(size_h), int(size_w))

    with tf.compat.v1.variable_scope(name) as vs:
        out = UpSampling2d(scale=size, method="nearest")(inputconv)
    return out


def autoenc_upsample(name):
    with tf.compat.v1.variable_scope(name):
        inputae = Input(shape=[None, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS], dtype=tf.float32)
        f = 7
        ks = 3
        padding = "REFLECT"

        pad_input = PadLayer([[0, 0], [ks, ks], [ks, ks], [0, 0]], padding)(inputae)

        o_c1 = Conv2d(
            n_filter=ngf,
            filter_size=(f, f),
            strides=(2, 2),
            act=None,
            padding="VALID",
            W_init=tf.initializers.TruncatedNormal(stddev=0.02),
            b_init=tf.constant_initializer(0.0)
        )(pad_input)

        o_c1 = InstanceNorm2d(act=tf.nn.relu)(o_c1)
        o_c2 = Conv2d(
            n_filter=ngf * 2,

            filter_size=(ks, ks),
            strides=(2, 2),
            padding="SAME",
            act=None,
            W_init=tf.initializers.TruncatedNormal(stddev=0.02),
            b_init=tf.constant_initializer(0.0)
        )(o_c1)
        o_c2 = InstanceNorm2d(act=tf.nn.relu)(o_c2)

        o_r1 = build_resnet_block_Att(o_c2, ngf * 2, "r1", padding)

        size_d1 = o_r1.get_shape().as_list()
        o_c4 = upsamplingDeconv(o_r1, size=[size_d1[1] * 2, size_d1[2] * 2], name="up1")
        o_c4 = PadLayer([[0, 0], [1, 1], [1, 1], [0, 0]], padding)(o_c4)
        o_c4_end = Conv2d(
            n_filter=ngf * 2,
            filter_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            act=None,
            W_init=tf.initializers.TruncatedNormal(stddev=0.02),
            b_init=tf.constant_initializer(0.0)
        )(o_c4)

        o_c4_end = InstanceNorm2d(act=tf.nn.relu)(o_c4_end)

        size_d2 = o_c4_end.get_shape().as_list()

        o_c5 = upsamplingDeconv(o_c4_end, size=[size_d2[1] * 2, size_d2[2] * 2], name="up2")

        o_c5 = PadLayer([[0, 0], [1, 1], [1, 1], [0, 0]], padding)(o_c5)
        o_c5_end = Conv2d(
            n_filter=ngf,
            filter_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            act=None,
            W_init=tf.initializers.TruncatedNormal(stddev=0.02),
            b_init=tf.constant_initializer(0.0)
        )(o_c5)

        o_c5_end = InstanceNorm2d(act=tf.nn.relu)(o_c5_end)
        o_c5_end = PadLayer([[0, 0], [3, 3], [3, 3], [0, 0]], padding)(o_c5_end)
        o_c6_end = Conv2d(
            n_filter=1,
            filter_size=(f, f),
            strides=(1, 1),
            padding="VALID",
            act=None,
            W_init=tf.initializers.TruncatedNormal(stddev=0.02),
            b_init=tf.constant_initializer(0.0)
        )(o_c5_end)

        output = Lambda(tf.nn.sigmoid)(o_c6_end)
        return Model(inputs=inputae, outputs=output)

def build_resnet_block(inputres, dim, name="resnet", padding="REFLECT"):
    with tf.compat.v1.variable_scope(name):
        out_res = PadLayer([[0, 0], [1, 1], [1, 1], [0, 0]], padding)(inputres)
        out_res = Conv2d(
            n_filter=dim,
            filter_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            act=None,
            W_init=tf.initializers.TruncatedNormal(stddev=0.02),
            b_init=tf.constant_initializer(0.0)
        )(out_res)
        out_res = InstanceNorm2d(act=tf.nn.relu)(out_res)
        out_res = PadLayer([[0, 0], [1, 1], [1, 1], [0, 0]], padding)(out_res)
        out_res = Conv2d(
            n_filter=dim,
            filter_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            act=None,
            W_init=tf.initializers.TruncatedNormal(stddev=0.02),
            b_init=tf.constant_initializer(0.0)
        )(out_res)
        out_res = InstanceNorm2d(act=None)(out_res)

        tmp = Elementwise(combine_fn=tf.add)([out_res, inputres])

        return Lambda(tf.nn.relu)(tmp)

def build_resnet_block_Att(inputres, dim, name="resnet", padding="REFLECT"):
    with tf.compat.v1.variable_scope(name):
        out_res = PadLayer([[0, 0], [1, 1], [1, 1], [0, 0]], padding)(inputres)

        out_res = Conv2d(
            n_filter=dim,
            filter_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            act=None,
            W_init=tf.initializers.TruncatedNormal(stddev=0.02),
            b_init=tf.constant_initializer(0.0)
        )(out_res)
        out_res = InstanceNorm2d(act=tf.nn.relu)(out_res)

        out_res = PadLayer([[0, 0], [1, 1], [1, 1], [0, 0]], padding)(out_res)

        out_res = Conv2d(
            n_filter=dim,
            filter_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            act=None,
            W_init=tf.initializers.TruncatedNormal(stddev=0.02),
            b_init=tf.constant_initializer(0.0)
        )(out_res)
        out_res = InstanceNorm2d(act=None)(out_res)

        tmp = Elementwise(combine_fn=tf.add)([out_res, inputres])
        return Lambda(tf.nn.relu)(tmp)

def build_generator_9blocks(name="generator", skip = False):
    with tf.compat.v1.variable_scope(name):
        #pdb.set_trace()
        inputgen = Input(shape=[None, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS], dtype=tf.float32)
        f = 7
        ks = 3
        padding  = "CONSTANT"
        padgen = PadLayer([[0, 0], [ks, ks], [ks, ks], [0, 0]], padding)(inputgen)

        o_c1 = Conv2d(
            n_filter=ngf,
            filter_size=(f, f),
            strides=(1, 1),
            padding="VALID",
            act=None,
            W_init=tf.initializers.TruncatedNormal(stddev=0.02),
            b_init=tf.constant_initializer(0.0)
        )(padgen)
        o_c1 = InstanceNorm2d(act=tf.nn.relu)(o_c1)

        o_c2 = Conv2d(
            n_filter=ngf * 2,
            filter_size=(ks, ks),
            strides=(2, 2),
            padding="SAME",
            act=None,
            W_init=tf.initializers.TruncatedNormal(stddev=0.02),
            b_init=tf.constant_initializer(0.0)
        )(o_c1)
        o_c2 = InstanceNorm2d(act=tf.nn.relu)(o_c2)

        o_c3 = Conv2d(
            n_filter=ngf * 4,
            filter_size=(ks, ks),
            strides=(2, 2),
            padding="SAME",
            act=None,
            W_init=tf.initializers.TruncatedNormal(stddev=0.02),
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
            W_init=tf.initializers.TruncatedNormal(stddev=0.02),
            b_init=tf.constant_initializer(0.0)
        )(o_r9)
        o_c4 = InstanceNorm2d(act=tf.nn.relu)(o_c4)

        o_c5 = DeConv2d(
            n_filter= ngf,
            filter_size=(ks, ks),
            strides=(2, 2),
            padding="SAME",
            act=None,
            W_init=tf.initializers.TruncatedNormal(stddev=0.02),
            b_init=tf.constant_initializer(0.0)
        )(o_c4)
        o_c5 = InstanceNorm2d(act=tf.nn.relu)(o_c5)

        o_c6 = Conv2d(
            n_filter=IMG_CHANNELS,
            filter_size=(f, f),
            strides=(1, 1),
            padding="SAME",
            act=None,
            W_init=tf.initializers.TruncatedNormal(stddev=0.02),
            b_init=tf.constant_initializer(0.0)
        )(o_c5)

        if skip is True:
            #out_gen = Lambda(tf.nn.tanh, name="t1")(Elementwise(combine_fn=tf.add)([inputgen, o_c6]))
            tmp = Elementwise(combine_fn=tf.add)([inputgen, o_c6])
            out_gen = Lambda(tf.nn.tanh)(tmp)
        else:
            #out_gen = Lambda(tf.nn.tanh, name="t1")(o_c6)
            out_gen = Lambda(tf.nn.tanh)(o_c6)

        return Model(inputs=inputgen, outputs=out_gen)

def my_cast(x):
    return tf.cast(x, tf.float32)

def my_cond(inps):
    # inps[0]: cond, inps[1]: x, inps[2]: y
    # cond? x: y
    return tf.multiply(inps[0], inps[1]) + tf.multiply(1 - inps[0], inps[2])

def discriminator(name="discriminator"):
    with tf.compat.v1.variable_scope(name):
        inputdisc_in = Input(shape=[None, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS], dtype=tf.float32)
        mask_in = Input(shape=[None, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS], dtype=tf.float32)
        transition_rate = Input(shape=[1], dtype=tf.float32)
        donorm = Input(shape=[1], dtype=tf.float32)

        tmp = Elementwise(combine_fn=tf.greater_equal)([mask_in, transition_rate])
        mask = Lambda(fn=my_cast)(tmp)
        inputdisc = Elementwise(combine_fn=tf.multiply)([inputdisc_in, mask])

        f = 4
        padw = 2
        lrelu = lambda x: tl.act.lrelu(x, 0.2)

        pad_input = PadLayer([[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")(inputdisc)

        o_c1 = Conv2d(
            n_filter=ndf,
            filter_size=(f, f),
            strides=(2, 2),
            padding="VALID",
            act=None,
            W_init=tf.initializers.TruncatedNormal(stddev=0.02),
            b_init=tf.constant_initializer(0.0)
        )(pad_input)
        #pdb.set_trace()
        o_c1 = Lambda(fn=my_cond)([donorm, InstanceNorm2d(act=None)(o_c1), o_c1])
        o_c1 = Lambda(fn=lrelu)(o_c1)

        pad_o_c1 = PadLayer([[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")(o_c1)

        o_c2 = Conv2d(
            n_filter=ndf * 2,
            filter_size=(f, f),
            strides=(2, 2),
            padding="VALID",
            act=None,
            W_init=tf.initializers.TruncatedNormal(stddev=0.02),
            b_init=tf.constant_initializer(0.0)
        )(pad_o_c1)
        o_c2 = Lambda(fn=my_cond)([donorm, InstanceNorm2d(act=None)(o_c2), o_c2])
        o_c2 = Lambda(fn=lrelu)(o_c2)

        pad_o_c2 = PadLayer([[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")(o_c2)

        o_c3 = Conv2d(
            n_filter=ndf * 4,
            filter_size=(f, f),
            strides=(2, 2),
            padding="VALID",
            act=None,
            W_init=tf.initializers.TruncatedNormal(stddev=0.02),
            b_init=tf.constant_initializer(0.0)
        )(pad_o_c2)
        o_c3 = Lambda(fn=my_cond)([donorm, InstanceNorm2d(act=None)(o_c3), o_c3])
        o_c3 = Lambda(fn=lrelu)(o_c3)

        pad_o_c3 = PadLayer([[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")(o_c3)

        o_c4 = Conv2d(
            n_filter=ndf * 8,
            filter_size=(f, f),
            strides=(1, 1),
            padding="VALID",
            act=None,
            W_init=tf.initializers.TruncatedNormal(stddev=0.02),
            b_init=tf.constant_initializer(0.0)
        )(pad_o_c3)
        o_c4 = Lambda(fn=my_cond)([donorm, InstanceNorm2d(act=None)(o_c4), o_c4])
        o_c4 = Lambda(fn=lrelu)(o_c4)

        pad_o_c4 = PadLayer([[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")(o_c4)

        o_c5 = Conv2d(
            n_filter=1,
            filter_size=(f, f),
            strides=(1, 1),
            padding="VALID",
            act=None,
            W_init=tf.initializers.TruncatedNormal(stddev=0.02),
            b_init=tf.constant_initializer(0.0)
        )(pad_o_c4)

        return Model(inputs=[inputdisc_in, mask_in, transition_rate, donorm], outputs=o_c5)


if __name__ == "__main__":
    width = IMG_WIDTH
    height = IMG_HEIGHT
    channels = IMG_CHANNELS

    assert tf.__version__[0] == '2'

    input_a = Input(shape=[None, width, height, channels],
        dtype=tf.float32, name="input_A")
    input_b = Input(shape=[None, width, height, channels],
        dtype=tf.float32, name="input_B")

    fake_pool_A = Input(shape=[None, width, height, channels],
        dtype=tf.float32, name="fake_pool_A")
    fake_pool_B = Input(shape=[None, width, height, channels],
        dtype=tf.float32, name="fake_pool_B")

    fake_pool_A_mask = Input(shape=[None, width, height, channels],
        dtype=tf.float32, name="fake_pool_A_mask")
    fake_pool_B_mask = Input(shape=[None, width, height, channels],
        dtype=tf.float32, name="fake_pool_B_mask")

    #global_step = tf.train.get_or_create_global_step()

    num_fake_inputs = 0

    # batch size = 1
    learning_rate = Input(shape=[1], dtype=tf.float32, name="lr")
    transition_rate = Input(shape=[1], dtype=tf.float32, name="tr")
    donorm = Input(shape=[1], dtype=tf.bool, name="donorm")


    inputs = {
        'images_a': input_a,
        'images_b': input_b,
        'fake_pool_a': fake_pool_A,
        'fake_pool_b': fake_pool_B,
        'fake_pool_a_mask': fake_pool_A_mask,
        'fake_pool_b_mask': fake_pool_B_mask,
        'transition_rate': transition_rate,
        'donorm': donorm,
    }

    build_model(skip=True)