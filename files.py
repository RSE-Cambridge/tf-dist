import os, glob
import tensorflow as tf
from tensorflow.data import Dataset

def classes(path):
    return [line for line in open(os.path.join(path, '..', 'cats.txt'), 'r')]

def num_classes(path):
    return len(classes(path))

def make_input_fn(path):
    cs = classes(path)
    nc = len(cs)

    def gen():
        for ic, c in enumerate(cs):
            c = c.strip()
            for filename in os.listdir(os.path.join(path, c)):
                yield os.path.join(path, c, filename), ic

    def load_and_augment(filename, ic):
        image_string = tf.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_images(image, [64, 64])

        #augmentation
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.clip_by_value(image, 0.0, 1.0)

        ic = tf.one_hot(ic, nc)
        return image, ic

    def input_fn(batch_size):
        dataset = tf.data.Dataset.from_generator(gen, (tf.string, tf.uint8))
        dataset = dataset.map(load_and_augment, num_parallel_calls=8)

        return dataset.shuffle(2000).batch(batch_size).prefetch(None)
    return input_fn
