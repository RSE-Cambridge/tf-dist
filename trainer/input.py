import tensorflow as tf
from tensorflow.logging import info, debug

def input_fn(images, labels, batch_size):
    debug("input_fn images shape %s"%(images.shape,))
    debug("input_fn labels shape %s"%(labels.shape,))
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    SHUFFLE_SIZE = 5000
    dataset = dataset.shuffle(SHUFFLE_SIZE).repeat().batch(batch_size)
    dataset = dataset.prefetch(None)

    return dataset

