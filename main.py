from datetime import datetime
import copy
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
from tensorlayer.iterate import minibatches

tf.set_random_seed(1)
np.random.seed(0)

class CycleGAN:
    # TODO: This code is for tensorflow 1.x with tl 1.x
    def __init__(self, pool_size, lambda_a,
                 lambda_b, output_root_dir, to_restore,
                 base_lr, max_step, 
                 dataset_name, checkpoint_dir, do_flipping, skip, switch, threshold_fg):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

        self._pool_size = pool_size
        self._pool_upd_threshold = 0.5 # TODO
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

        # TODO: tensorlayer 1.x only

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


    def figure_writer(self, figures_to_save, names, v_html, html_mode):
        """
        A helper function to save images and updating html
        """

        for name, figure_save in zip(names, figures_to_save):
            image_name = name + str(epoch) + "_" + str(i) + ".jpg"
            if 'mask_' in name:
                figure = np.squeeze(figure_save[0])
            else:
                figure = ((np.squeeze(figure_save[0]) + 1) * 127.5).astype(np.uint8)

            tl_save_image(figure, image_path=os.path.join(self._images_dir, image_name))

            # writing html automatically...
            v_html.write(
                "<img src=\"" +
                os.path.join('imgs', image_name) + "\">"
            )
            if html_mode == 1 and 'fakeB_' in name:
                v_html.write("<br>")
        
        v_html.write("<br>")


    def save_images(self, sess, epoch, curr_tr, images_i, images_j):
        """
        Saves input and output images.
        """
        exists_or_mkdir(self._images_dir)

        if curr_tr > 0:
            donorm = False
        else:
            donorm = True

        names = ['inputA_', 'inputB_', 'fakeA_', 'fakeB_', 
                 'cycA_', 'cycB_', 'mask_a', 'mask_b']

        with open(os.path.join(self._output_dir, 'epoch_' + str(epoch) + '.html'), 'w') as v_html:
            input_iter = minibatches(images_i, images_j, batch_size=1, shuffle=True)

            for i in range(0, self._num_imgs_to_save):
                print("Saving image {}/{}...".format(i, self._num_imgs_to_save))
                image_i, image_j = next(input_iter)
                fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp, masks = sess.run(
                    [self.fake_images_a,
                     self.fake_images_b,
                     self.cycle_images_a,
                     self.cycle_images_b,
                     self.masks], 
                    feed_dict={
                        self.input_a: image_i,
                        self.input_b: image_j,
                        self.transition_rate: curr_tr,
                        self.donorm: donorm
                    }
                )

                figures_to_save = [image_i, image_j, fake_B_temp, fake_A_temp, 
                                   cyc_A_temp, cyc_B_temp, masks[0], masks[1]]

                self.figure_writer(figures_to_save, names, v_html, html_mode=0)
                

    def save_images_bis(self, sess, epoch, images_i, images_j):
        """
        Saves input and output images.
        """
        names = ['input_A_', 'mask_A_', 'masked_inputA_', 'fakeB_',
                 'input_B_', 'mask_B_', 'masked_inputB_', 'fakeA_']
        space = '&nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp ' \
                '&nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp ' \
                '&nbsp &nbsp &nbsp &nbsp &nbsp'

        exists_or_mkdir(self._images_dir)

        with open(os.path.join(self._output_dir, 'results_' + str(epoch) + '.html'), 'w') as v_html:
            v_html.write("<b>Inputs" + space + "Masks" + space + "Masked_images" + space + "Generated_images</b>")
            v_html.write("<br>")

            input_iter = minibatches(images_i, images_j, batch_size=1, shuffle=True)

            for i in range(0, self._num_imgs_to_save):
                print("Saving image {}/{}...".format(i, self._num_imgs_to_save))
                image_i, image_j = next(input_iter)

                fake_A_temp, fake_B_temp, masks, masked_ims = sess.run(
                    [self.fake_images_a,
                     self.fake_images_b,
                     self.masks,
                     self.masked_ims],
                    feed_dict={
                        self.input_a: image_i,
                        self.input_b: image_j,
                        self.transition_rate: 0.1
                    }
                )

                figures_to_save = [image_i, masks[0], masked_ims[0], fake_B_temp,
                                   image_j, masks[1], masked_ims[1], fake_A_temp]

                self.figure_writer(figures_to_save, names, v_html, html_mode=1)


    def fake_image_pool(self, num_fakes, fake, mask, fake_pool):
        """
        Maintain a pool of fake images
        """
        tmp = dict()
        tmp['im'] = fake
        tmp['mask'] = mask
        if num_fakes < self._pool_size:
            fake_pool.append(copy.deepcopy(tmp))
            return tmp
        else:
            p = random.random()
            if p > self._pool_upd_threshold:
                random_id = random.randint(0, self._pool_size - 1)
                temp = copy.deepcopy(fake_pool[random_id])
                fake_pool[random_id] = copy.deepcopy(tmp)
                return temp
            else:
                return tmp


    def train(self):
        """
        Training Function.
        We use batch size = 1 for training
        """

        # Build the network and compute losses
        self.model_setup()
        self.compute_losses()

        # Initializing the global variables
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())

        max_images = cyclegan_datasets.DATASET_TO_SIZES[self._dataset_name]
        half_training_ep = int(self._max_step / 2)
        with tf.Session() as sess:
            sess.run(init)
            
            # Restore the model to run the model from last checkpoint
            print("Loading the latest checkpoint...")

            if self._to_restore:
                load_ckpt(sess, save_dir=self._checkpoint_dir, is_latest=True)

            writer = tf.summary.FileWriter(self._output_dir)

            if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            # Training Loop
            for epoch in range(sess.run(self.global_step), self._max_step):
                print("In the epoch ", epoch)
                print("Saving the latest checkpoint...")
                # TODO: note the format of the model is for tf 1.x
                save_ckpt(sess, mode_name="AGGAN",  
                    save_dir=self._output_dir, global_step=epoch)

                # Setting lr
                curr_lr = self._base_lr
                if epoch >= half_training_ep:
                    curr_lr -= self._base_lr * (epoch - half_training_ep) / half_training_ep


                if epoch < self._switch:
                    curr_tr = 0
                    donorm = True
                    to_train_A = self.g_A_trainer
                    to_train_B = self.g_B_trainer
                else:
                    curr_tr = self._threshold_fg
                    donorm = False
                    to_train_A = self.g_A_trainer_bis
                    to_train_B = self.g_B_trainer_bis

                print("Loading data...")
                tot_inputs = data_loader.load_data(
                    self._dataset_name, self._size_before_crop,
                    False, self._do_flipping
                )
                self.inputs_img_i = tot_inputs['images_i']
                self.inputs_img_j = tot_inputs['images_j']
                assert (len(self.inputs_img_i) == len(self.inputs_img_j)
                    and max_images == len(self.inputs_img_i)

                self.save_images(sess, epoch, curr_tr, self.inputs_img_i, self.inputs_img_j)

                input_iter = minibatches(self.inputs_img_i, self.inputs_img_j, batch_size=1, shuffle=True)
                for i in range(max_images):
                    image_i, image_j = next(input_iter)
                    print("Processing batch {}/{}...".format(i, max_images))
                    # Optimizing the G_A network
                    _, fake_B_temp, smask_a, summary_string = sess.run(
                        [to_train_A,
                         self.fake_images_b,
                         self.masks[0],
                         self.g_A_loss_summ],
                        feed_dict={
                            self.input_a: image_i,
                            self.input_b: image_j,
                            self.learning_rate: curr_lr,
                            self.transition_rate: curr_tr,
                            self.donorm: donorm,
                        }
                    )
                    writer.add_summary(summary_string, epoch * max_images + i)

                    fake_B_temp1 = self.fake_image_pool(
                        self.num_fake_inputs, fake_B_temp, smask_a, self.fake_images_B
                    )

                    # Optimizing the D_B network
                    _, summary_string = sess.run(
                        [self.d_B_trainer, self.d_B_loss_summ],
                        feed_dict={
                            self.input_a: image_i,
                            self.input_b: image_j,
                            self.learning_rate: curr_lr,
                            self.fake_pool_B: fake_B_temp1['im'],
                            self.fake_pool_B_mask: fake_B_temp1['mask'],
                            self.transition_rate: curr_tr,
                            self.donorm: donorm,
                        }
                    )
                    writer.add_summary(summary_string, epoch * max_images + i)


                    # Optimizing the G_B network
                    _, fake_A_temp, smask_b, summary_string = sess.run(
                        [to_train_B,
                         self.fake_images_a,
                         self.masks[1],
                         self.g_B_loss_summ],
                        feed_dict={
                            self.input_a: image_i,
                            self.input_b: image_j,
                            self.learning_rate: curr_lr,
                            self.transition_rate: curr_tr,
                            self.donorm: donorm,
                        }
                    )
                    writer.add_summary(summary_string, epoch * max_images + i)

                    fake_A_temp1 = self.fake_image_pool(
                        self.num_fake_inputs, fake_A_temp, smask_b ,self.fake_images_A
                    )

                    # Optimizing the D_A network
                    _, mask_tmp__, summary_string = sess.run(
                        [self.d_A_trainer,self.masks_, self.d_A_loss_summ],
                        feed_dict={
                            self.input_a: image_i,
                            self.input_b: image_j,
                            self.learning_rate: curr_lr,
                            self.fake_pool_A: fake_A_temp1['im'],
                            self.fake_pool_A_mask: fake_A_temp1['mask'],
                            self.transition_rate: curr_tr,
                            self.donorm: donorm,
                        }
                    )

                    self.num_fake_inputs += 1

                    writer.add_summary(summary_string, epoch * max_images + i)
                    writer.flush()
                    
                sess.run(tf.assign(self.global_step, epoch + 1))

            coord.request_stop()
            coord.join(threads)
            writer.add_graph(sess.graph)


    def test(self):
        """
        Testing Function.
        """
        print("Testing the results")
        print("Loading data...")

        tot_inputs = data_loader.load_data(
            self._dataset_name, self._size_before_crop,
            False, self._do_flipping, num_img=self._num_imgs_to_save
        )
        self.inputs_img_i = tot_inputs['images_i']
        self.inputs_img_j = tot_inputs['images_j']
        assert len(self.inputs_img_i) == len(self.inputs_img_j)

        self.model_setup()
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            print("Loading the latest checkpoint...")
            load_ckpt(sess, save_dir=self._checkpoint_dir, is_latest=True)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            self._num_imgs_to_save = cyclegan_datasets.DATASET_TO_SIZES[self._dataset_name]

            self.save_images_bis(sess, sess.run(self.global_step),
                self.inputs_img_i, self.inputs_img_j)

            coord.request_stop()
            coord.join(threads)