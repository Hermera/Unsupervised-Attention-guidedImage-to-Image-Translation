# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorlayer as tl
import numpy as np

import cyclegan_datasets
import model

def minibatches_unsupervised(inputs=None, batch_size=None, allow_dynamic_batch_size=False, shuffle=False):
    """
    ######## Modified on the prototype of tensorlayer.iterate.minibatches ########
    ########                 unsupervised --> no label                    ########
    Generate a generator that input a group of example in numpy.array, 
    return the examples by the given batch size.

    Parameters
    ----------
    inputs : numpy.array
        The input features, every row is a example.
    batch_size : int
        The batch size.
    allow_dynamic_batch_size: boolean
        Allow the use of the last data batch in case the number of examples is not a multiple of batch_size, this may result in unexpected behaviour if other functions expect a fixed-sized batch-size.
    shuffle : boolean
        Indicating whether to use a shuffling queue, shuffle the dataset before return.
    """
    
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    # for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
    # chulei: handling the case where the number of samples is not a multiple of batch_size, avoiding wasting samples
    for start_idx in range(0, len(inputs), batch_size):
        end_idx = start_idx + batch_size
        if end_idx > len(inputs):
            if allow_dynamic_batch_size:
                end_idx = len(inputs)
            else:
                break
        if shuffle:
            excerpt = indices[start_idx:end_idx]
        else:
            excerpt = slice(start_idx, end_idx)
        if isinstance(inputs, list) and (shuffle ==True):
            # zsdonghao: for list indexing when shuffle==True
            yield [inputs[i] for i in excerpt]
        else:
            yield inputs[excerpt]

"""
def _load_samples(csv_name, image_type):
    
    filename_queue = tf.train.string_input_producer(
        [csv_name])

    reader = tf.TextLineReader()
    _, csv_filename = reader.read(filename_queue)

    record_defaults = [tf.constant([], dtype=tf.string),
                       tf.constant([], dtype=tf.string)]

    filename_i, filename_j = tf.decode_csv(
        csv_name, record_defaults=record_defaults)  #csv_filename

    file_contents_i = tf.io.read_file(filename_i)
    file_contents_j = tf.io.read_file(filename_j)
    if image_type == '.jpg':
        image_decoded_A = tf.io.decode_jpeg(
            file_contents_i, channels=model.IMG_CHANNELS)
        image_decoded_B = tf.io.decode_jpeg(
            file_contents_j, channels=model.IMG_CHANNELS)
    elif image_type == '.png':
        image_decoded_A = tf.io.decode_png(
            file_contents_i, channels=model.IMG_CHANNELS, dtype=tf.uint8)
        image_decoded_B = tf.io.decode_png(
            file_contents_j, channels=model.IMG_CHANNELS, dtype=tf.uint8)

    return image_decoded_A, image_decoded_B
"""

def load_data(dataset_name, image_size_before_crop,
              do_shuffle=False, do_flipping=False):
    
    if dataset_name not in cyclegan_datasets.DATASET_TO_SIZES:
        raise ValueError('split name %s was not recognized.'
                         % dataset_name)
    
    image_i, image_j, _, _ = tl.files.load_cyclegan_dataset(dataset_name, path='input')
    num_rows = cyclegan_datasets.DATASET_TO_SIZES[dataset_name]
    
    if len(image_i) < num_rows:
        for i in range(len(image_i),num_rows):
            image_i.append(image_i[i%len(image_i)])
    if len(image_j) < num_rows:
        for j in range(len(image_j),num_rows):
            image_j.append(image_j[j%len(image_j)])
    
    inputs = {
        'image_i': image_i,
        'image_j': image_j
    }

    def resize_wrapper(im_list, image_size_before_crop):
        im_l=[]
        for im in im_list:
            im_l.append(tl.prepro.imresize(
                    im, [image_size_before_crop, image_size_before_crop]))
        return im_l
    """
    def flip_wrapper(im_list):
        im_l=[]
        for im in im_list:
            im_l.append(tl.prepro.flip_axis(im, axis=1, is_random=True))
        return im_l
    """
    
    # Preprocessing:
    inputs['image_i'] = resize_wrapper(inputs['image_i'], 
          image_size_before_crop)
    inputs['image_j'] = resize_wrapper(inputs['image_j'], 
          image_size_before_crop)
    
    
    if do_flipping is True:
        inputs['image_i'] = tl.prepro.flip_axis(inputs['image_i'], axis=1, is_random=True)
        inputs['image_j'] = tl.prepro.flip_axis(inputs['image_j'], axis=1, is_random=True)

    inputs['image_i'] = tl.prepro.crop_multi(inputs['image_i'], 
          model.IMG_WIDTH, model.IMG_HEIGHT, is_random=True)
    inputs['image_j'] = tl.prepro.crop_multi(inputs['image_j'], 
          model.IMG_WIDTH, model.IMG_HEIGHT, is_random=True)

    inputs['image_i']=np.array(inputs['image_i'])
    inputs['image_j']=np.array(inputs['image_j'])
    
    inputs['image_i']=(inputs['image_i']/127.5)-1
    inputs['image_j']=(inputs['image_j']/127.5)-1

    """
    # Batch    
    if do_shuffle is True:
        inputs['images_i'], inputs['images_j'] = minibatches_unsupervised(
                [inputs['image_i'], inputs['image_j']],1,shuffle=True)
    else:
        inputs['images_i'], inputs['images_j'] = minibatches_unsupervised(
                [inputs['image_i'], inputs['image_j']],1)
    """
    
    return inputs
"""
def load_data(dataset_name, image_size_before_crop,
              do_shuffle=False, do_flipping=False):
    

    :param dataset_name: The name of the dataset.
    :param image_size_before_crop: Resize to this size before random cropping.
    :param do_shuffle: Shuffle switch.
    :param do_flipping: Flip switch.
    :return:

    
    if dataset_name not in cyclegan_datasets.DATASET_TO_SIZES:
        raise ValueError('split name %s was not recognized.'
                         % dataset_name)

    csv_name = cyclegan_datasets.PATH_TO_CSV[dataset_name]

    image_i, image_j = _load_samples(
        csv_name, cyclegan_datasets.DATASET_TO_IMAGETYPE[dataset_name])
    inputs = {
        'image_i': image_i,
        'image_j': image_j
    }


    def resize_wrapper(im_list, image_size_before_crop):
        im_l=[]
        for im in im_list:
            im_l.append(tl.prepro.imresize(
                    im, [image_size_before_crop, image_size_before_crop]))
        return im_l

    def flip_wrapper(im_list):
        im_l=[]
        for im in im_list:
            im_l.append(tl.prepro.flip_axis(im, axis=1, is_random=True))
        return im_l

    # Preprocessing:
    inputs['image_i'] = resize_wrapper(inputs['image_i'], 
          image_size_before_crop)
    inputs['image_j'] = resize_wrapper(inputs['image_j'], 
          image_size_before_crop)
    
    
    if do_flipping is True:
        inputs['image_i'] = flip_wrapper(inputs['image_i'])
        inputs['image_i'] = flip_wrapper(inputs['image_i'])


    inputs['image_i'] = tl.prepro.crop_multi(inputs['image_i'], 
          model.IMG_WIDTH, model.IMG_HEIGHT, is_random=True)
    inputs['image_j'] = tl.prepro.crop_multi(inputs['image_j'], 
          model.IMG_WIDTH, model.IMG_HEIGHT, is_random=True)


    inputs['image_i'] = tf.subtract(tf.math.divide(inputs['image_i'], 127.5), 1)
    inputs['image_j'] = tf.subtract(tf.math.divide(inputs['image_j'], 127.5), 1)

    # Batch    
    if do_shuffle is True:
        inputs['images_i'], inputs['images_j'] = minibatches_unsupervised(
                [inputs['image_i'], inputs['image_j']],1,shuffle=True)
    else:
        inputs['images_i'], inputs['images_j'] = minibatches_unsupervised(
                [inputs['image_i'], inputs['image_j']],1)

    return inputs
"""