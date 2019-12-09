# -*- coding: utf-8 -*-
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
    
    return inputs