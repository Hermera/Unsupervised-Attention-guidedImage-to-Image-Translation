import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import (BatchNorm2d, Conv2d, Dense, Flatten, Input, DeConv2d, Lambda, concat\
                                LocalResponseNorm, MaxPool2d, Elementwise, InstanceNorm2d, PadLayer)
from tensorlayer.models import Model

IMG_CHANNELS = 3
IMG_WIDTH = 256
IMG_HEIGHT = 256

ngf = 32
ndf = 64

"""
Debug了一部分，但bug太多了2333
"""

def get_outputs(inputs, skip=False):

    images_a = inputs['images_a']
    images_b = inputs['images_b']
    fake_pool_a = inputs['fake_pool_a']
    fake_pool_b = inputs['fake_pool_b']
    fake_pool_a_mask = inputs['fake_pool_a_mask']
    fake_pool_b_mask = inputs['fake_pool_b_mask']
    transition_rate = inputs['transition_rate']
    donorm = inputs['donorm']

    current_autoenc = autoenc_upsample
    current_discriminator = discriminator
    current_generator = build_generator_9blocks

    mask_a = current_autoenc(images_a, "g_A_ae")
    mask_b = current_autoenc(images_b, "g_B_ae")
    mask_a = tf.concat([mask_a] * 3, axis=3)
    mask_b = tf.concat([mask_b] * 3, axis=3)

    mask_a_on_a = tf.multiply(images_a, mask_a)
    mask_b_on_b = tf.multiply(images_b, mask_b)

    prob_real_a_is_real = current_discriminator(images_a, mask_a, transition_rate, donorm, "d_A")
    prob_real_b_is_real = current_discriminator(images_b, mask_b, transition_rate, donorm, "d_B")

    fake_images_b_from_g = current_generator(images_a, name="g_A", skip=skip)
    fake_images_b = tf.multiply(fake_images_b_from_g, mask_a) + tf.multiply(images_a, 1-mask_a)

    fake_images_a_from_g = current_generator(images_b, name="g_B", skip=skip)
    fake_images_a = tf.multiply(fake_images_a_from_g, mask_b) + tf.multiply(images_b, 1-mask_b)
    scope.reuse_variables()

    prob_fake_a_is_real = current_discriminator(fake_images_a, mask_b, transition_rate, donorm, "d_A")
    prob_fake_b_is_real = current_discriminator(fake_images_b, mask_a, transition_rate, donorm, "d_B")

    mask_acycle = current_autoenc(fake_images_a, "g_A_ae")
    mask_bcycle = current_autoenc(fake_images_b, "g_B_ae")
    mask_bcycle = tf.concat([mask_bcycle] * 3, axis=3)
    mask_acycle = tf.concat([mask_acycle] * 3, axis=3)

    mask_acycle_on_fakeA = tf.multiply(fake_images_a, mask_acycle)
    mask_bcycle_on_fakeB = tf.multiply(fake_images_b, mask_bcycle)

    cycle_images_a_from_g = current_generator(fake_images_b, name="g_B", skip=skip)
    cycle_images_b_from_g = current_generator(fake_images_a, name="g_A", skip=skip)

    cycle_images_a = tf.multiply(cycle_images_a_from_g,
                                 mask_bcycle) + tf.multiply(fake_images_b, 1 - mask_bcycle)

    cycle_images_b = tf.multiply(cycle_images_b_from_g,
                                 mask_acycle) + tf.multiply(fake_images_a, 1 - mask_acycle)

    scope.reuse_variables()

    prob_fake_pool_a_is_real = current_discriminator(fake_pool_a, fake_pool_a_mask, transition_rate, donorm, "d_A")
    prob_fake_pool_b_is_real = current_discriminator(fake_pool_b, fake_pool_b_mask, transition_rate, donorm, "d_B")

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
        'masked_ims': [mask_a_on_a, mask_b_on_b, mask_acycle_on_fakeA, mask_bcycle_on_fakeB],
        'masks': [mask_a, mask_b, mask_acycle, mask_bcycle],
        'masked_gen_ims' : [fake_images_b_from_g, fake_images_a_from_g , cycle_images_a_from_g, cycle_images_b_from_g],
        'mask_tmp' : mask_a,
    }


def upsamplingDeconv(inputconv, size, is_scale, method, align_corners, name):
    
    size_h = size[0] // int(inputconv.get_shape()[1])
    size_w = size[1] // int(inputconv.get_shape()[2])
    size = [int(size_h), int(size_w)]
    
    print("  [TL] UpSampling2dLayer %s: is_scale:%s size:%s method:%d align_corners:%s" %
          (name, is_scale, size, method, align_corners))
    try:
        out = tl.layers.UpSampling2d(scale=size)(inputconv)
    except:  # for TF 0.10
        out = tf.image.resize(inputconv, new_height=size[0], new_width=size[1], method=method)
    return out

def autoenc_upsample(inputae, name):
    f = 7
    ks = 3
    padding = "REFLECT"

    pad_input = PadLayer([[0, 0], [ks, ks], [ks, ks], [0, 0]], mode=padding)(inputae)
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
    o_c4 = upsamplingDeconv(o_r1, size=[size_d1[1] * 2, size_d1[2] * 2], is_scale=False, method=1, align_corners=False, name="up1")
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
    o_c5 = upsamplingDeconv(o_c4_end, size=[size_d2[1] * 2, size_d2[2] * 2], is_scale=False, method=1, align_corners=Flase, name="up2")
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
    o_c6_end = Conv2d(
        n_filter=1,
        filter_size=(f, f),
        strides=(1, 1),
        padding="VALID",
        act=None,
        W_init=tf.initializers.TruncatedNormal(stddev=0.02),
        b_init=tf.constant_initializer(0.0)
    )(o_c5_end)

    return tf.nn.sigmoid(o_c6_end, "sigmoid")

def build_resnet_block(inputres, dim, name="resnet", padding="REFLECT"):
    out_res = PadLayer([[0, 0], [1, 1], [1, 1], [0, 0]], mode=padding)(inputres)
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
    out_res = PadLayer([[0, 0], [1, 1], [1, 1], [0, 0]], mode=padding)(out_res)
    out_res = Conv2d(
        n_filter=dim,
        filter_size=(3, 3),
        strides=(1, 1),
        padding="VALID",
        act=None,
        W_init=tf.initializers.TruncatedNormal(stddev=0.02),
        b_init=tf.constant_initializer(0.0)
    )
    out_res = InstanceNorm2d(act=None)(out_res)

    return tf.nn.relu(out_res + inputres)

def build_resnet_block_Att(inputres, dim, name="resnet", padding="REFLECT"):
    out_res = PadLayer([[0, 0], [1, 1], [1, 1], [0, 0]], mode=padding)(inputres)

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

    out_res = PadLayer([[0, 0], [1, 1], [1, 1], [0, 0]], mode=padding)(out_res)

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

    return tf.nn.relu(out_res + inputres)

def build_generator_9blocks(inputgen, name="generator", skip = False):
    f = 7
    ks = 3
    padding  = "CONSTANT"
    inputgen = PadLayer([[0, 0], [ks, ks], [ks, ks], [0, 0]], mode=padding)(inputgen)

    o_c1 = Conv2d(
        n_filter=ngf,
        filter_size=(f, f),
        strides=(1, 1),
        padding="VALID",
        act=None,
        W_init=tf.initializers.TruncatedNormal(stddev=0.02),
        b_init=tf.constant_initializer(0.0)
    )(inputgen)
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
        out_gen = tf.nn.tanh(inputgen + o_c6, name="t1")
    else:
        out_gen = tf.nn.tanh(o_c6, "t1")

    return out_gen

def discriminator(inputdisc, mask, transition_rate, donorm, name="discriminator"):
    mask = tf.cast(tf.greater_equal(mask, transition_rate), tf.float32)
    inputdisc = tf.multiply(inputdisc, mask)
    f = 4
    padw = 2
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    pad_input = PadLayer([[0, 0], [padw, padw], [padw, padw], [0, 0]], mode="CONSTANT")(inputdisc)

    o_c1 = Conv2d(
        n_filter=ndf,
        filter_size=(f, f),
        strides=(2, 2),
        padding="VALID",
        act=None,
        W_init=tf.initializers.TruncatedNormal(stddev=0.02),
        b_init=tf.constant_initializer(0.0)
    )(pad_input)
    if donorm is True:
        o_c1 = InstanceNorm2d(act=lrelu)(o_c1)
    else:
        o_c1 = lrelu(o_c1)

    pad_o_c1 = PadLayer([[0, 0], [padw, padw], [padw, padw], [0, 0]], mode="CONSTANT")(o_c1)

    o_c2 = Conv2d(
        n_filter=ndf * 2,
        filter_size=(f, f),
        strides=(2, 2),
        padding="VALID",
        act=None,
        W_init=tf.initializers.TruncatedNormal(stddev=0.02),
        b_init=tf.constant_initializer(0.0)
    )(pad_o_c1)
    if donorm is True:
        o_c2 = InstanceNorm2d(act=lrelu)(o_c2)
    else:
        o_c2 = lrelu(o_c2)

    pad_o_c2 = PadLayer([[0, 0], [padw, padw], [padw, padw], [0, 0]], mode="CONSTANT")(o_c2)

    o_c3 = Conv2d(
        n_filter=ndf * 4,
        filter_size=(f, f),
        strides=(2, 2),
        padding="VALID",
        act=None,
        W_init=tf.initializers.TruncatedNormal(stddev=0.02),
        b_init=tf.constant_initializer(0.0)
    )(pad_o_c2)
    if donorm is True:
        o_c3 = InstanceNorm2d(act=lrelu)(o_c3)
    else:
        o_c3 = lrelu(o_c3)

    pad_o_c3 = PadLayer([[0, 0], [padw, padw], [padw, padw], [0, 0]], mode="CONSTANT")(o_c3)

    o_c4 = Conv2d(
        n_filter=ndf * 8,
        filter_size=(f, f),
        strides=(1, 1),
        padding="VALID",
        act=None,
        W_init=tf.initializers.TruncatedNormal(stddev=0.02),
        b_init=tf.constant_initializer(0.0)
    )(pad_o_c3)
    if donorm is True:
        o_c4 = InstanceNorm2d(act=lrelu)(o_c4)
    else:
        o_c4 = lrelu(o_c4)

    pad_o_c4 = PadLayer([[0, 0], [padw, padw], [padw, padw], [0, 0]], mode="CONSTANT")(o_c4)

    o_c5 = Conv2d(
        n_filter=1,
        filter_size=(f, f),
        strides=(1, 1),
        padding="VALID",
        act=lrelu,
        W_init=tf.initializers.TruncatedNormal(stddev=0.02),
        b_init=tf.constant_initializer(0.0)
    )(pad_o_c4)

    return o_c5


if __name__ == "__main__":
    width = IMG_WIDTH
    height = IMG_HEIGHT
    channels = IMG_CHANNELS


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

    outputs = get_outputs(inputs, skip=1) # all the outputs

    


    inp_list = [tensor for tensor in inputs.values()]
    oup_list = [tensor for tensor in ouputs.values()]
    net = Model(inputs=inp_list, outputs=oup_list)