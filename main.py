from datetime import datetime
import json
import numpy as np
import os
import random
import argparse
import tensorflow as tf

import cyclegan_datasets
import data_loader, losses, model

import tensorlayer as tl
from tensorlayer.files import save_ckpt, load_ckpt, exists_or_mkdir
from tensorlayer.prepro import threading_data
from tensorlayer.visualize import save_image as tl_save_image

tf.set_random_seed(1)
np.random.seed(0)

class CycleGAN:
    def __init__(self, pool_size, lambda_a,
                 lambda_b, output_root_dir, to_restore,
                 base_lr, max_step, 
                 dataset_name, checkpoint_dir, do_flipping, skip, switch, threshold_fg):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

        self._pool_size = pool_size
        self._size_before_crop = 286
        self._switch = switch
        self._threshold_fg = threshold_fg
        self._lambda_a = lambda_a
        self._lambda_b = lambda_b
        self._output_dir = os.path.join(output_root_dir, current_time +
                                        '_switch'+str(switch)+'_thres_'+str(threshold_fg))
        self._images_dir = os.path.join(self._output_dir, 'imgs')
        self._num_imgs_to_save = 20
        self._to_restore = to_restore
        self._base_lr = base_lr
        self._max_step = max_step
        self._dataset_name = dataset_name
        self._checkpoint_dir = checkpoint_dir
        self._do_flipping = do_flipping
        self._skip = skip

        self.fake_images_A = []
        self.fake_images_B = []


    def model_setup(self):
        width = model.IMG_WIDTH
        height = model.IMG_HEIGHT
        channels = model.IMG_CHANNELS

        self.input_a = tf.placeholder(
            tf.float32, [
                1, width, height, channels
            ], name="input_A")

        self.input_b = tf.placeholder(
            tf.float32, [
                1, width, model.height, channels
            ], name="input_B")

        self.fake_pool_A = tf.placeholder(
            tf.float32, [
                None, width, height, channels
            ], name="fake_pool_A")

        self.fake_pool_B = tf.placeholder(
            tf.float32, [
                None, width, height, channels
            ], name="fake_pool_B")

        self.fake_pool_A_mask = tf.placeholder(
            tf.float32, [
                None, width, height, channels
            ], name="fake_pool_A_mask")

        self.fake_pool_B_mask = tf.placeholder(
            tf.float32, [
                None, width, height, channels
            ], name="fake_pool_B_mask")

        self.global_step = tf.train.get_or_create_global_step()

        self.num_fake_inputs = 0

        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="lr")
        self.transition_rate = tf.placeholder(tf.float32, shape=[], name="tr")
        self.donorm = tf.placeholder(tf.bool, shape=[], name="donorm")

        inputs = {
            'images_a': self.input_a,
            'images_b': self.input_b,
            'fake_pool_a': self.fake_pool_A,
            'fake_pool_b': self.fake_pool_B,
            'fake_pool_a_mask': self.fake_pool_A_mask,
            'fake_pool_b_mask': self.fake_pool_B_mask,
            'transition_rate': self.transition_rate,
            'donorm': self.donorm,
        }

        outputs = model.get_outputs(inputs, skip=self._skip) # all the outputs

        self.prob_real_a_is_real = outputs['prob_real_a_is_real']
        self.prob_real_b_is_real = outputs['prob_real_b_is_real']
        self.fake_images_a = outputs['fake_images_a']
        self.fake_images_b = outputs['fake_images_b']
        self.prob_fake_a_is_real = outputs['prob_fake_a_is_real']
        self.prob_fake_b_is_real = outputs['prob_fake_b_is_real']

        self.cycle_images_a = outputs['cycle_images_a']
        self.cycle_images_b = outputs['cycle_images_b']

        self.prob_fake_pool_a_is_real = outputs['prob_fake_pool_a_is_real']
        self.prob_fake_pool_b_is_real = outputs['prob_fake_pool_b_is_real']
        self.masks = outputs['masks']
        self.masked_gen_ims = outputs['masked_gen_ims']
        self.masked_ims = outputs['masked_ims']
        self.masks_ = outputs['mask_tmp']


    def compute_losses(self):
        cycle_consistency_loss_a = \
            self._lambda_a * losses.cycle_consistency_loss(
                real_images=self.input_a, generated_images=self.cycle_images_a,
            )
        cycle_consistency_loss_b = \
            self._lambda_b * losses.cycle_consistency_loss(
                real_images=self.input_b, generated_images=self.cycle_images_b,
            )

        lsgan_loss_a = losses.lsgan_loss_generator(self.prob_fake_a_is_real)
        lsgan_loss_b = losses.lsgan_loss_generator(self.prob_fake_b_is_real)

        #generator losses
        g_loss_A = cycle_consistency_loss_a + cycle_consistency_loss_b + lsgan_loss_b 
        g_loss_B = cycle_consistency_loss_b + cycle_consistency_loss_a + lsgan_loss_a

        #discriminator losses
        d_loss_A = losses.lsgan_loss_discriminator(
            prob_real_is_real=self.prob_real_a_is_real,
            prob_fake_is_real=self.prob_fake_pool_a_is_real,
        )
        d_loss_B = losses.lsgan_loss_discriminator(
            prob_real_is_real=self.prob_real_b_is_real,
            prob_fake_is_real=self.prob_fake_pool_b_is_real,
        )

        optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)
        self.model_vars = tf.trainable_variables()

        d_A_vars = [var for var in self.model_vars if 'd_A' in var.name]
        g_A_vars = [var for var in self.model_vars if 'g_A/' in var.name]
        d_B_vars = [var for var in self.model_vars if 'd_B' in var.name]
        g_B_vars = [var for var in self.model_vars if 'g_B/' in var.name]
        g_Ae_vars = [var for var in self.model_vars if 'g_A_ae' in var.name]
        g_Be_vars = [var for var in self.model_vars if 'g_B_ae' in var.name]

        self.g_A_trainer = optimizer.minimize(g_loss_A, var_list=g_A_vars+g_Ae_vars)
        self.g_B_trainer = optimizer.minimize(g_loss_B, var_list=g_B_vars+g_Be_vars)
        self.g_A_trainer_bis = optimizer.minimize(g_loss_A, var_list=g_A_vars)
        self.g_B_trainer_bis = optimizer.minimize(g_loss_B, var_list=g_B_vars)
        self.d_A_trainer = optimizer.minimize(d_loss_A, var_list=d_A_vars)
        self.d_B_trainer = optimizer.minimize(d_loss_B, var_list=d_B_vars)

        for var in self.model_vars:
            print(var.name)

        # Summary variables for tensorboard
        self.g_A_loss_summ = tf.summary.scalar("g_A_loss", g_loss_A)
        self.g_B_loss_summ = tf.summary.scalar("g_B_loss", g_loss_B)
        self.d_A_loss_summ = tf.summary.scalar("d_A_loss", d_loss_A)
        self.d_B_loss_summ = tf.summary.scalar("d_B_loss", d_loss_B)