'''
Creates an input function consuming TFrecords. The TFrecords should be stored in a
single directory as described in 

https://github.com/tensorflow/models/blob/f87a58cd96d45de73c9a8330a06b2ab56749a7fa/
research/inception/inception/data/build_image_data.py

'''

import os
import tensorflow as tf


mode_dict = {'test':  'validation-*',
             'train': 'train-*'
             }


def get_filenames(path, mode):
    return  [os.path.join(path,fn) for fn in os.listdir(path)
                if fn[0] == mode_dict[mode]]

def classes(path):
    return [line for line in open(os.path.join(path, 'cats.txt'), 'r')]

def num_classes(path):
    return len(classes(path))

def make_input_fn(path, mode):

    nc = num_classes(path)

    def load_and_augment(single_record):
        features = tf.parse_single_example(
                      single_record,
                      features={
                        'image/encoded': tf.FixedLenFeature([], tf.string),
                        'image/class/label': tf.FixedLenFeature([], tf.int64),
                      })
        image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_images(image, [64, 64])
        #augmentation
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        label = tf.one_hot(features['image/class/label'], nc)
        return image, label 

    def input_fn(batch_size):
        n_epochs=10
        n_parallel=12

        files = tf.data.Dataset.list_files(os.path.join(path,mode_dict[mode]), shuffle=True, seed=0)
        # block_length: number of consecutive elements to produce from each input element before 
        #               cycling to another input element
        dataset = files.apply(tf.contrib.data.parallel_interleave(
                tf.data.TFRecordDataset, cycle_length=n_parallel, block_length=1))  
        
        dataset = dataset.apply(
                tf.contrib.data.shuffle_and_repeat(buffer_size=2000, count=10, seed=0))

        ### Cache in memory
        dataset = dataset.cache()
       
        dataset = dataset.apply(
                tf.contrib.data.map_and_batch(load_and_augment, batch_size))
        
        dataset = dataset.prefetch(buffer_size=batch_size) 
        return dataset
    return input_fn
