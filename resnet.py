import tensorflow as tf


BATCH_NORM_MOMENTUM = 0.997
BATCH_NORM_EPSILON = 1e-5
DATA_FORMAT = 'channels_first'  # 'channels_first' or 'channels_last'


def resnet(images, is_training, num_classes):
    """
    This is an implementation of classical ResNet-50.
    It is take from here:
    https://github.com/tensorflow/models/blob/master/official/resnet/

    Arguments:
        images: a float tensor with shape [batch_size, image_height, image_width, 3],
            a batch of RGB images with pixel values in the range [0, 1].
        is_training: a boolean.
        num_classes: an integer.
    Returns:
        a float tensor with shape [batch_size, num_classes].
    """

    with tf.name_scope('standardize_input'):
        means = (1.0 / 255.0) * tf.constants([123.68, 116.78, 103.94], dtype=tf.float32)
        x = images - means

    with tf.variable_scope('ResNet-50'):

        if DATA_FORMAT == 'channels_first':
            x = tf.transpose(x, [0, 3, 1, 2])

        x = conv2d_same(x, 64, kernel_size=7, stride=2)
        x = batch_norm_relu(x, is_training)
        x = tf.layers.max_pooling2d(
            inputs=x, pool_size=3, strides=2,
            padding='same', data_format=DATA_FORMAT
        )

        num_units_per_block = [3, 4, 6, 3]
        strides = [1, 2, 2, 2]

        for i, num_units, stride in enumerate(zip(num_units_per_block, strides)):
            num_filters = 64 * (2**i)  # [64, 128, 256, 512]
            x = block(x, is_training, num_filters, num_units, stride)

        # global average pooling
        axes = [2, 3] if DATA_FORMAT == 'channels_first' else [1, 2]
        x = tf.reduce_mean(x, axes)

    logits = tf.layers.dense(
        x, units=num_classes, name='classifier',
        kernel_initializer=tf.random_normal_initializer(stddev=0.01)
    )
    return logits


def block(x, is_training, num_filters, num_units, stride):
    x = bottleneck(x, is_training, num_filters, stride, use_projection=True)
    for _ in range(1, num_units):
        x = bottleneck(x, is_training, num_filters)
    return x


def bottleneck(x, is_training, num_filters, stride=1, use_projection=False):

    shortcut = x

    if use_projection:
        shortcut = conv2d_same(shortcut, 4 * num_filters, kernel_size=1, stride=stride)
        shortcut = batch_norm_relu(shortcut, is_training, use_relu=False)

    x = conv2d_same(x, num_filters, kernel_size=1)
    x = batch_norm_relu(x, is_training)

    x = conv2d_same(x, num_filters, kernel_size=3, stride=stride, rate=rate)
    x = batch_norm_relu(x, is_training)

    x = conv2d_same(x, 4 * num_filters, kernel_size=1)
    x = batch_norm_relu(x, is_training, use_relu=False)

    x += shortcut
    x = tf.nn.relu(x)
    return x


def batch_norm_relu(x, is_training, use_relu=True):
    x = tf.layers.batch_normalization(
        inputs=x, axis=1 if DATA_FORMAT == 'channels_first' else 3,
        momentum=BATCH_NORM_MOMENTUM, epsilon=BATCH_NORM_EPSILON,
        center=True, scale=True, training=is_training, fused=True
    )
    if use_relu:
        x = tf.nn.relu(x)
    return x


def conv2d_same(x, num_filters, kernel_size=3, stride=1, rate=1):
    if stride == 1:
        return tf.layers.conv2d(
            inputs=x, filters=num_filters,
            kernel_size=(kernel_size, kernel_size),
            strides=(stride, stride),
            dilation_rate=(rate, rate),
            padding='same', use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
            data_format=DATA_FORMAT
        )
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        if DATA_FORMAT == 'channels_first':
            paddings = [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]]
        else:
            paddings = [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]]

        return tf.layers.conv2d(
            inputs=tf.pad(x, paddings), filters=num_filters,
            kernel_size=(kernel_size, kernel_size),
            strides=(stride, stride),
            dilation_rate=(rate, rate),
            padding='valid', use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
            data_format=DATA_FORMAT
        )
