"""
TODO: add parse_argument function and add the main routine (main function)
"""
import copy
import json
import numpy as np
import os
import random
import argparse


import tensorflow as tf
assert tf.__version__[0] == '2'


import cyclegan_datasets
import data_loader
import model
from test_loss import cycle_consistency_loss, lsgan_loss_discriminator, lsgan_loss_generator


import tensorlayer as tl
from tensorlayer.layers import Input
from tensorlayer.files import exists_or_mkdir
from tensorlayer.prepro import threading_data
from tensorlayer.visualize import save_image as tl_save_image
from tensorlayer.iterate import minibatches
from tensorlayer.files import load_and_assign_npz_dict, save_npz_dict
from tensorlayer.models import Model


class CycleGAN(object):
    def __init__(self, pool_size, lambda_a,
                 lambda_b, output_root_dir, to_restore, checkpoint_name, 
                 base_lr, max_step, 
                 dataset_name, checkpoint_dir, do_flipping, skip, switch, threshold_fg):
        self._pool_size = pool_size
        self._pool_upd_threshold = 0.5
        self._size_before_crop = 286
        self._switch = switch
        self._threshold_fg = threshold_fg
        self._lambda_a = lambda_a
        self._lambda_b = lambda_b
        self._output_dir = os.path.join(output_root_dir,
                                        'switch'+str(switch)+'_thres_'+str(threshold_fg))
        self._images_dir = os.path.join(self._output_dir, 'imgs')
        self._num_imgs_to_save = 20
        self._to_restore = to_restore
        self._checkpoint_name = checkpoint_name
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


        self.input_a = Input(shape=[None, width, height, channels], 
            dtype=tf.float32, name="input_A")
        self.input_b = Input(shape=[None, width, height, channels],
            dtype=tf.float32, name="input_B")
        
        self.fake_pool_A = Input(shape=[None, width, height, channels],
            dtype=tf.float32, name="fake_pool_A")
        self.fake_pool_B = Input(shape=[None, width, height, channels],
            dtype=tf.float32, name="fake_pool_B")

        self.fake_pool_A_mask = Input(shape=[None, width, height, channels],
            dtype=tf.float32, name="fake_pool_A_mask")
        self.fake_pool_B_mask = Input(shape=[None, width, height, channels],
            dtype=tf.float32, name="fake_pool_B_mask")

        self.num_fake_inputs = 0

        # batch size = 1
        self.learning_rate = Input(shape=[1], dtype=tf.float32, name="lr")
        self.transition_rate = Input(shape=[1], dtype=tf.float32, name="tr")
        self.donorm = Input(shape=[1], dtype=tf.bool, name="donorm")

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

        inp_list = [tensor for tensor in inputs.values()]
        oup_list = [tensor for tensor in outputs.values()]

        net = Model(inputs=inp_list, outputs=oup_list + inp_list)

        for var in net.trainable_weights:
            print(var.name)

        self.d_A_vars = [var for var in net.trainable_weights if 'd_A' in var.name]
        self.g_A_vars = [var for var in net.trainable_weights if 'g_A/' in var.name]
        self.d_B_vars = [var for var in net.trainable_weights if 'd_B' in var.name]
        self.g_B_vars = [var for var in net.trainable_weights if 'g_B/' in var.name]
        self.g_Ae_vars = [var for var in net.trainable_weights if 'g_A_ae' in var.name]
        self.g_Be_vars = [var for var in net.trainable_weights if 'g_B_ae' in var.name]


        self.g_A_trainer = tf.optimizers.Adam(self.learning_rate, beta_1=0.5)
        self.g_B_trainer = tf.optimizers.Adam(self.learning_rate, beta_1=0.5)
        self.g_A_trainer_bis = tf.optimizers.Adam(self.learning_rate, beta_1=0.5)
        self.g_B_trainer_bis = tf.optimizers.Adam(self.learning_rate, beta_1=0.5)
        self.d_A_trainer = tf.optimizers.Adam(self.learning_rate, beta_1=0.5)
        self.d_B_trainer = tf.optimizers.Adam(self.learning_rate, beta_1=0.5)

        return net


    def input_converter(self):
        return [self.input_a, self.input_b, self.fake_pool_A, self.fake_pool_B, 
            self.fake_pool_A_mask, self.fake_pool_B_mask, self.transition_rate, self.donorm]

    def output_converter(self, outputs):
        self.prob_real_a_is_real = outputs[0]
        self.prob_real_b_is_real = outputs[1]
        self.fake_images_a = outputs[2]
        self.fake_images_b = outputs[3]
        self.prob_fake_a_is_real = outputs[4]
        self.prob_fake_b_is_real = outputs[5]

        self.cycle_images_a = outputs[6]
        self.cycle_images_b = outputs[7]

        self.prob_fake_pool_a_is_real = outputs[8]
        self.prob_fake_pool_b_is_real = outputs[9]
        self.masks = outputs[10]
        self.masked_gen_ims = outputs[11]
        self.masked_ims = outputs[12]
        self.masks_ = outputs[13]
        self.input_a = outputs[14]
        self.input_b = outputs[15]
        self.fake_pool_A = outputs[16]
        self.fake_pool_B = outputs[17]
        self.fake_pool_A_mask = outputs[18]
        self.fake_pool_B_mask = outputs[19]
        self.transition_rate = outputs[20]
        self.donorm = outputs[21]


    def compute_losses(self):
        cycle_consistency_loss_a = \
            self._lambda_a * cycle_consistency_loss(
                real_images=self.input_a, generated_images=self.cycle_images_a,
            )
        cycle_consistency_loss_b = \
            self._lambda_b * cycle_consistency_loss(
                real_images=self.input_b, generated_images=self.cycle_images_b,
            )

        lsgan_loss_a = lsgan_loss_generator(self.prob_fake_a_is_real)
        lsgan_loss_b = lsgan_loss_generator(self.prob_fake_b_is_real)

        #generator losses
        self.g_A_loss = cycle_consistency_loss_a + cycle_consistency_loss_b + lsgan_loss_b 
        self.g_B_loss = cycle_consistency_loss_b + cycle_consistency_loss_a + lsgan_loss_a

        #discriminator losses
        self.d_A_loss = lsgan_loss_discriminator(
            prob_real_is_real=self.prob_real_a_is_real,
            prob_fake_is_real=self.prob_fake_pool_a_is_real,
        )
        self.d_B_loss = lsgan_loss_discriminator(
            prob_real_is_real=self.prob_real_b_is_real,
            prob_fake_is_real=self.prob_fake_pool_b_is_real,
        )



    def figure_writer(self, figures_to_save, names, v_html, epoch, i, html_mode):
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


    def save_images(self, net, epoch, curr_tr, images_i, images_j):
        """
        Saves input and output images.
        """
        net.eval()

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
                self.image_a, self.image_b = next(input_iter)
                tmp_imgA = self.get_fake_image_pool(
                    self.num_fake_inputs, self.fake_images_A
                )
                self.fake_pool_A_mask = tmp_imgA["mask"]
                self.fake_pool_A = tmp_imgA["im"]

                tmp_imgB = self.get_fake_image_pool(
                    self.num_fake_inputs, self.fake_images_B
                )
                self.fake_pool_B_mask = tmp_imgB["mask"]
                self.fake_pool_B = tmp_imgB["im"]

                self.transition_rate = np.array([curr_tr], dtype=np.float32)
                self.donorm = np.array([donorm], dtype=np.bool)

                self.output_converter(net(self.input_converter()))

                figures_to_save = [self.image_a, self.image_b, self.fake_images_b, self.fake_images_a, 
                                   self.cycle_images_a, self.cycle_images_b, self.masks[0], self.masks[1]]

                self.figure_writer(figures_to_save, names, v_html, epoch, i, html_mode=0)
                

    def save_images_bis(self, net, epoch, images_i, images_j):
        """
        Saves input and output images.
        """
        names = ['input_A_', 'mask_A_', 'masked_inputA_', 'fakeB_',
                 'input_B_', 'mask_B_', 'masked_inputB_', 'fakeA_']
        space = '&nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp ' \
                '&nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp ' \
                '&nbsp &nbsp &nbsp &nbsp &nbsp'

        net.eval()

        exists_or_mkdir(self._images_dir)

        with open(os.path.join(self._output_dir, 'results_' + str(epoch) + '.html'), 'w') as v_html:
            v_html.write("<b>Inputs" + space + "Masks" + space + "Masked_images" + space + "Generated_images</b>")
            v_html.write("<br>")

            input_iter = minibatches(images_i, images_j, batch_size=1, shuffle=True)

            for i in range(0, self._num_imgs_to_save):
                print("Saving image {}/{}...".format(i, self._num_imgs_to_save))

                self.image_a, self.image_b = next(input_iter)
                tmp_imgA = self.get_fake_image_pool(
                    self.num_fake_inputs, self.fake_images_A
                )
                self.fake_pool_A_mask = tmp_imgA["mask"]
                self.fake_pool_A = tmp_imgA["im"]

                tmp_imgB = self.get_fake_image_pool(
                    self.num_fake_inputs, self.fake_images_B
                )
                self.fake_pool_B_mask = tmp_imgB["mask"]
                self.fake_pool_B = tmp_imgB["im"]

                self.transition_rate = np.array([0.1], dtype=np.float32)
                self.donorm = np.array([True], dtype=np.bool)

                self.output_converter(net(self.input_converter()))

                figures_to_save = [self.image_a, self.masks[0], self.masked_ims[0], self.fake_images_b,
                                   self.image_b, self.masks[1], self.masked_ims[1], self.fake_images_a]

                self.figure_writer(figures_to_save, names, v_html, epoch, i, html_mode=1)


    def upd_fake_image_pool(self, num_fakes, fake, mask, fake_pool):
        """
        Maintain a pool of fake images
        """
        tmp = dict()
        tmp['im'] = fake.numpy()
        tmp['mask'] = mask.numpy()
        if num_fakes < self._pool_size:
            fake_pool.append(copy.deepcopy(tmp))
            if num_fakes == 1:
                fake_pool[0] = copy.deepcopy(tmp)
        else:
            random_id = random.randint(0, self._pool_size - 1)
            fake_pool[random_id] = copy.deepcopy(tmp)


    def get_fake_image_pool(self, num_fakes, fake_pool):
        num_fakes = min(num_fakes, self._pool_size)
        random_id = num_fakes - 1
        if random.random() > self._pool_upd_threshold:
            random_id = random.randint(0, num_fakes - 1)
        
        temp = copy.deepcopy(fake_pool[random_id])
        return temp


    def train(self):
        """
        Training Function.
        We use batch size = 1 for training
        """

        # Build the network and compute losses
        net = self.model_setup()


        max_images = cyclegan_datasets.DATASET_TO_SIZES[self._dataset_name]
        half_training_ep = int(self._max_step / 2)
        

        # Restore the model to run the model from last checkpoint
        print("Loading the latest checkpoint...")

        if self._to_restore:
            checkpoint_name = os.path.join(self._checkpoint_dir, self._checkpoint_name)
            load_and_assign_npz_dict(checkpoint_name, network=net)
            self.global_step = int(checkpoint_name[-2:])
        else:
            self.global_step = 0

        exists_or_mkdir(self._output_dir)

        self.upd_fake_image_pool(self.num_fake_inputs, self.fake_pool_A,
            self.fake_pool_A_mask, self.fake_images_A)
        self.upd_fake_image_pool(self.num_fake_inputs, self.fake_pool_B,
            self.fake_pool_B_mask, self.fake_images_B)
        self.num_fake_inputs += 1
        

        # Training Loop
        for epoch in range(self.global_step, self._max_step):
            print("In the epoch ", epoch)
            print("Saving the latest checkpoint...")
            save_npz_dict(net.all_weights, os.path.join(self._output_dir, "AGGAN_%2d" % epoch))


            # Setting lr
            curr_lr = self._base_lr
            if epoch >= half_training_ep:
                curr_lr -= self._base_lr * (epoch - half_training_ep) / half_training_ep
                self.g_A_trainer.learning_rate = curr_lr
                self.g_B_trainer.learning_rate = curr_lr
                self.g_A_trainer_bis.learning_rate = curr_lr
                self.g_B_trainer_bis.learning_rate = curr_lr
                self.d_A_trainer.learning_rate = curr_lr
                self.d_B_trainer.learning_rate = curr_lr

            if epoch < self._switch:
                curr_tr = 0
                donorm = True
                to_train_A = self.g_A_trainer
                to_train_B = self.g_B_trainer
                to_train_A_vars = self.g_A_vars + self.g_Ae_vars
                to_train_B_vars = self.g_B_vars + self.g_Be_vars
            else:
                curr_tr = self._threshold_fg
                donorm = False
                to_train_A = self.g_A_trainer_bis
                to_train_B = self.g_B_trainer_bis
                to_train_A_vars = self.g_A_vars
                to_train_B_vars = self.g_B_vars


            print("Loading data...")
            tot_inputs = data_loader.load_data(
                self._dataset_name, self._size_before_crop,
                False, self._do_flipping
            )
            self.inputs_img_i = tot_inputs['images_i']
            self.inputs_img_j = tot_inputs['images_j']
            assert (len(self.inputs_img_i) == len(self.inputs_img_j) and max_images == len(self.inputs_img_i))

            self.save_images(net, epoch, curr_tr, self.inputs_img_i, self.inputs_img_j)

            net.train()

            input_iter = minibatches(self.inputs_img_i, self.inputs_img_j, batch_size=1, shuffle=True)
            for i in range(max_images):
                print("Processing batch {}/{} in {}th epoch".format(i, max_images, epoch))

                self.image_a, self.image_b = next(input_iter)
                tmp_imgA = self.get_fake_image_pool(
                    self.num_fake_inputs, self.fake_images_A
                )
                self.fake_pool_A_mask = tmp_imgA["mask"]
                self.fake_pool_A = tmp_imgA["im"]

                tmp_imgB = self.get_fake_image_pool(
                    self.num_fake_inputs, self.fake_images_B
                )
                self.fake_pool_B_mask = tmp_imgB["mask"]
                self.fake_pool_B = tmp_imgB["im"]

                self.transition_rate = np.array([curr_tr], dtype=np.float32)
                self.donorm = np.array([donorm], dtype=np.bool)

                with tf.GradientTape(persistent=True) as tape:
                    self.output_converter(net(self.input_converter()))
                    self.upd_fake_image_pool(
                        self.num_fake_inputs, self.fake_images_b, self.masks[0], self.fake_images_B
                    )
                    self.upd_fake_image_pool(
                        self.num_fake_inputs, self.fake_images_a, self.masks[1], self.fake_images_A
                    )
                    self.compute_losses()

                grad = tape.gradient(self.d_B_loss, self.d_B_vars)
                self.d_B_trainer.apply_gradients(zip(grad, self.d_B_vars))

                grad = tape.gradient(self.d_A_loss, self.d_A_vars)
                self.d_A_trainer.apply_gradients(zip(grad, self.d_A_vars))

                grad = tape.gradient(self.g_A_loss, to_train_A_vars)
                to_train_A.apply_gradients(zip(grad, to_train_A_vars))

                grad = tape.gradient(self.g_B_loss, to_train_B_vars)
                to_train_B.apply_gradients(zip(grad, to_train_B_vars))

                tot_loss = self.g_A_loss + self.g_B_loss + self.d_A_loss + self.d_B_loss

                print("[training_info] g_A_loss = {}, g_B_loss = {}, d_A_loss = {}, d_B_loss = {}, \
                    tot_loss = {}, lr={}, curr_tr={}".format(self.g_A_loss, self.g_B_loss, self.d_A_loss, \
                    self.d_B_loss, tot_loss, curr_lr, curr_tr))

                self.num_fake_inputs += 1
                
            self.global_step = epoch + 1



    def test(self):
        """
        Testing Function.
        """
        print("Testing the results")
        print("Loading data...")

        tot_inputs = data_loader.load_data(
            self._dataset_name, self._size_before_crop,
            False, self._do_flipping
        )
        self.inputs_img_i = tot_inputs['images_i']
        self.inputs_img_j = tot_inputs['images_j']
        assert len(self.inputs_img_i) == len(self.inputs_img_j)

        net = self.model_setup()

        self.upd_fake_image_pool(self.num_fake_inputs, self.fake_pool_A,
            self.fake_pool_A_mask, self.fake_images_A)
        self.upd_fake_image_pool(self.num_fake_inputs, self.fake_pool_B,
            self.fake_pool_B_mask, self.fake_images_B)
        self.num_fake_inputs += 1

        print("Loading the latest checkpoint...")
        checkpoint_name = os.path.join(self._checkpoint_dir, self._checkpoint_name)
        load_and_assign_npz_dict(checkpoint_name, network=net)
        self.global_step = int(self._checkpoint_name[-2:])

        self._num_imgs_to_save = cyclegan_datasets.DATASET_TO_SIZES[self._dataset_name]

        self.save_images_bis(net, self.global_step, self.inputs_img_i, self.inputs_img_j)




def parse_args():
    desc = "Tensorlayer implementation of cycleGan using attention"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--to_train', type=int, default=True, 
        help='Train mode or not.')
        
    parser.add_argument('--log_dir',
        type=str, default=None, help='The folder where the data is logged to.')

    parser.add_argument('--config_filename', type=str, default='train', 
        help='The name of the configuration file.')

    parser.add_argument('--checkpoint_dir', type=str, default='', 
        help='The directory of the train/test split.')

    parser.add_argument('--skip', type=bool, default=False,
        help='Whether to add skip connection between input and output.')

    parser.add_argument('--switch', type=int, default=30,
        help='In what epoch the FG starts to be fed to the discriminator')

    parser.add_argument('--threshold', type=float, default=0.1,
        help='The threshold proportion to select the FG')

    parser.add_argument('--checkpoint_name', type=str, default='',
        help='The suffix name of the latest checkpoint.')


    return parser.parse_args()

def main():
    
    args = parse_args()
    if args is None:
        exit()

    to_train = args.to_train
    log_dir = args.log_dir
    config_filename = args.config_filename
    checkpoint_dir = args.checkpoint_dir
    skip = args.skip
    switch = args.switch
    threshold_fg = args.threshold

    exists_or_mkdir(log_dir)
    
    with open(config_filename) as config_file:
        config = json.load(config_file)


    lambda_a = float(config['_LAMBDA_A']) if '_LAMBDA_A' in config else 10.0
    lambda_b = float(config['_LAMBDA_B']) if '_LAMBDA_B' in config else 10.0
    pool_size = int(config['pool_size']) if 'pool_size' in config else 50

    to_restore = (to_train == 2)
    base_lr = float(config['base_lr']) if 'base_lr' in config else 0.0002
    max_step = int(config['max_step']) if 'max_step' in config else 200
    dataset_name = str(config['dataset_name'])
    do_flipping = bool(config['do_flipping'])
    checkpoint_name = args.checkpoint_name

    if checkpoint_name == '' and to_train != 1:
        print("Error: please provide the latest checkpoint name.")
        exit()

    cyclegan_model = CycleGAN(pool_size, lambda_a, lambda_b, log_dir,
                              to_restore, checkpoint_name, base_lr, max_step, 
                              dataset_name, checkpoint_dir, do_flipping, skip,
                              switch, threshold_fg)

    if to_train > 0:
        cyclegan_model.train()
    else:
        cyclegan_model.test()


if __name__ == '__main__':
    main()
