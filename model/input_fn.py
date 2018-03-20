"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf
import re
import numpy as np
import os

NUM_CAMS = 4


def _parse_function(filename, label, size):
    """Obtain the image from the filename (for both training and validation).

    The following operations are applied:
        - Decode the image from jpeg format
        - Convert to float and to range [0, 1]
    """
#     print(filename)
#     print("filename is ^")
    image_output = None
    for i in range(NUM_CAMS): 
        fileBase = 'data/JDP/CS230_Visuomotor_Learning/parser/TeleOpVRSession_2018-03-07_14-38-06_Camera' + str(i + 1) + '/'
        fileBase = tf.constant(fileBase)
#         split_filenames = tf.string_split([filename], 'CAM1')
#         for idx in split_filenames.indicies():
#             filenames_arr.append(split_filenames[idx])
#         print('printing split filenames shape')
#         print(split_filenames.shape)
        filename_part_one = tf.substr(filename, 0, 40)
        filename_part_two = tf.substr(filename, 41, -1)
#         new_path = fileBase + split_filenames.values[0] + tf.constant('CAM' + str(i + 1)) + split_filenames.values[1]
        new_path = fileBase + filename_part_one + tf.constant(str(i + 1)) + filename_part_two
#         new_path = split_filenames.values[0]
#         print('what is this?')
#         print(final_filename)
        # make path point to correct camera folder
#         new_path = tf.py_func(fix_string, [str(i + 1), filename], tf.string) 
#         call fix_string with params i and filename on tf.py_func
        
        image_string = tf.read_file(new_path)
        # Don't use tf.image.decode_image, or the output shape will be undefined
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)

        # This will convert to float values in [0, 1]
        image = tf.image.convert_image_dtype(image_decoded, tf.float32)

        resized_image = tf.image.resize_images(image, [size, size])
        
#         (size, size, 3)
        if(image_output != None):
            image_output = tf.concat([image_output, resized_image], axis=2)
        else:
            image_output = resized_image
    
        

    return image_output, label


def train_preprocess(image, label, use_random_flip):
    """Image preprocessing for training.

    Apply the following operations:
        - Horizontally flip the image with probability 1/2
        - Apply random brightness and saturation
    """
    print(image.get_shape())
    C1,C2,C3,C4 = tf.split(image,[3,3,3,3],2)
    image_output = None
    for i, img in enumerate([C1,C2,C3,C4]):
        if use_random_flip:
            img = tf.image.random_flip_left_right(img)

        img = tf.image.random_brightness(img, max_delta=32.0 / 255.0)
        img = tf.image.random_saturation(img, lower=0.5, upper=1.5)

#         Make sure the image is still in [0, 1]
        img = tf.clip_by_value(img, 0.0, 1.0)
        if(image_output != None):
            image_output = tf.concat([image_output, img], axis=2)
        else:
            image_output = img
    
    return image_output, label


def input_fn(is_training, filenames, labels, params):
   
    """Input function for the SIGNS dataset.

    The filenames have format "{label}_IMG_{id}.jpg".
    For instance: "data_dir/2_IMG_4584.jpg".

    Args:
        is_training: (bool) whether to use the train or test pipeline.
                     At training, we shuffle the data and have multiple epochs
        filenames: (list) filenames of the images, as ["data_dir/{label}_IMG_{id}.jpg"...]
        labels: (list) corresponding list of labels
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    num_samples = len(filenames)
    assert len(filenames) == labels.shape[0], "Filenames and labels should have same length"

    # Create a Dataset serving batches of images and labels
    # We don't repeat for multiple epochs because we always train and evaluate for one epoch
    parse_fn = lambda f, l: _parse_function(f, l, params.image_size)
    train_fn = lambda f, l: train_preprocess(f, l, params.use_random_flip)
    print(labels)
    if is_training:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
            .shuffle(num_samples)  # whole dataset into the buffer ensures good shuffling
            .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
            .map(train_fn, num_parallel_calls=params.num_parallel_calls)
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )
    else:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
            .map(parse_fn)
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )

    # Create reinitializable iterator from dataset
    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()
    iterator_init_op = iterator.initializer

    inputs = {'images': images, 'labels': labels, 'iterator_init_op': iterator_init_op}
    return inputs
