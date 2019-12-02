import tensorflow as tf
from tensorlayer.cost import mean_squared_error, absolute_difference_error

def cycle_consistency_losses(real_imgs, gen_imgs):
    return absolute_difference_error(gen_imgs, real_imgs, is_mean=True)

def lsgan_loss_generator(prob_fake_is_real):
    return mean_squared_error(prob_fake_is_real, 1, is_mean=True)

def lsgan_loss_discriminator(prob_real_is_real, prob_fake_is_real):
    return mean_squared_error(prob_real_is_real, 1, is_mean=True) +
            mean_squared_error(prob_fake_is_real, 0, is_mean=True) * 0.5
