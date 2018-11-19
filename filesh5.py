import h5py
import tensorflow as tf

def num_classes(filename):
    with h5py.File(filename, 'r') as f:
        return len(f['categories'])

def make_input_fn(filename, s):
    f = h5py.File(filename, 'r')
    nc = len(f['categories'])

    images = np.asarray(f[s]['data'], dtype=np.float32)

    labels = tf.keras.utils.to_categorical(f[s]['labels'], nc)
    labels = labels.astype(np.float32)
    labels = labels.astype('int').reshape((-1, nc))

    def augment(filename, ic):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.clip_by_value(image, 0.0, 1.0)

        ic = tf.one_hot(ic, nc)
        return image, ic

    def input_fn(batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.map(augment, num_parallel_calls=8)
        return dataset.shuffle(2000*nc).batch(batch_size).prefetch(None)
    return input_fn
