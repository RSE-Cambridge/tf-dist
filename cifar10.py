import numpy as np
import tensorflow as tf
from tensorflow.logging import info, debug

(train_images, train_labels), (test_images, test_labels) = \
      tf.keras.datasets.cifar10.load_data()

img_width, img_height, img_channels = 32, 32, 3
label_dimensions = 10

train_images = np.asarray(train_images, dtype=np.float32) / 255
test_images = np.asarray(test_images, dtype=np.float32) / 255

train_images_mean = np.mean(train_images, axis=0)
train_images -= train_images_mean
test_images -= train_images_mean

train_images = train_images.reshape((-1, img_width, img_height, img_channels))
test_images = test_images.reshape((-1, img_width, img_height, img_channels))

debug("shape train_images %s" % (train_images.shape,))
debug("shape train_labels %s" % (train_labels.shape,))
debug("shape test_images %s" % (test_images.shape,))
debug("shape test_labels %s" % (test_labels.shape,))

train_labels  = tf.keras.utils.to_categorical(train_labels, label_dimensions)
test_labels = tf.keras.utils.to_categorical(test_labels, label_dimensions)

train_labels = train_labels.astype(np.float32)
test_labels = test_labels.astype(np.float32)

train_labels = np.asarray(train_labels).astype('int').reshape((-1, label_dimensions))
test_labels = np.asarray(test_labels).astype('int').reshape((-1, label_dimensions))

def make_input_fn(images, labels):
    def input_fn(batch_size):
        debug("input_fn images shape %s"%(images.shape,))
        debug("input_fn labels shape %s"%(labels.shape,))
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))

        SHUFFLE_SIZE = 5000
        dataset = dataset.shuffle(SHUFFLE_SIZE).repeat().batch(batch_size)
        dataset = dataset.prefetch(None)

        return dataset
    return input_fn

test_input_fn = make_input_fn(test_images, test_labels)
train_input_fn = make_input_fn(train_images, train_labels)

num_classes = label_dimensions
