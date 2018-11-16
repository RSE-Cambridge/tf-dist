import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.logging import info, debug


metrics = [
        tf.keras.metrics.categorical_accuracy,
        tf.keras.metrics.categorical_crossentropy,
        tf.keras.metrics.top_k_categorical_accuracy
        ]

def model_fn(img_width, img_height, img_channels, classes, learning_rate):
  input = Input((img_width, img_height, img_channels))

  from tensorflow.keras.applications.resnet50 import ResNet50
  model = ResNet50(weights=None, input_tensor=input, classes=classes)

  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

  model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
  
  return model

def model_fn_new(img_width, img_height, img_channels, num_classes, learning_rate):
    # Subtracting pixel mean improves accuracy

# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------
    n = 3

# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
    version = 1

# Computed depth from supplied model parameter n
    if version == 1:
        depth = n * 6 + 2
    elif version == 2:
        depth = n * 9 + 2

    from resnet import resnet_v1, resnet_v2

    if version == 2:
        model = resnet_v2(input_shape=(img_width, img_height, img_channels), depth=depth)
    else:
        model = resnet_v1(input_shape=(img_width, img_height, img_channels), depth=depth)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)

    info('ResNet%dv%d' % (depth, version))
    return model
