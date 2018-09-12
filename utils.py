import tensorflow as tf
import tensorflow.contrib.slim as slim


def block(x, num_units, out_channels=None, downsample=True, scope='stage'):
    with tf.variable_scope(scope):

        with tf.variable_scope('unit_1'):
            x, y = basic_unit_with_downsampling(
                x, out_channels=out_channels,
                downsample=downsample
            )

        for j in range(2, num_units + 1):
            with tf.variable_scope('unit_%d' % j):
                x, y = concat_shuffle_split(x, y)
                x = basic_unit(x)
        x = tf.concat([x, y], axis=3)

    return x


def concat_shuffle_split(x, y):
    with tf.name_scope('concat_shuffle_split'):
        shape = tf.shape(x)
        batch_size = shape[0]
        height, width = shape[1], shape[2]
        depth = x.shape[3].value

        z = tf.stack([x, y], axis=3)  # shape [batch_size, height, width, 2, depth]
        z = tf.transpose(z, [0, 1, 2, 4, 3])
        z = tf.reshape(z, [batch_size, height, width, 2*depth])
        x, y = tf.split(z, num_or_size_splits=2, axis=3)
        return x, y


def basic_unit(x):
    in_channels = x.shape[3].value
    x = slim.conv2d(x, in_channels, (1, 1), stride=1, scope='conv1x1_before')
    x = slim.separable_conv2d(x, None, (3, 3), stride=1, depth_multiplier=1, activation_fn=None, scope='depthwise')
    x = slim.conv2d(x, in_channels, (1, 1), stride=1, scope='conv1x1_after')
    return x


def basic_unit_with_downsampling(x, out_channels=None, downsample=True):
    in_channels = x.shape[3].value
    out_channels = 2 * in_channels if out_channels is None else out_channels
    stride = 2 if downsample else 1  # paradoxically, it sometimes doesn't downsample

    y = slim.conv2d(x, in_channels, (1, 1), stride=1, scope='conv1x1_before')
    y = slim.separable_conv2d(y, None, (3, 3), stride=stride, depth_multiplier=1, activation_fn=None, scope='depthwise')
    y = slim.conv2d(y, out_channels // 2, (1, 1), stride=1, scope='conv1x1_after')

    with tf.variable_scope('second_branch'):
        x = slim.separable_conv2d(x, None, (3, 3), stride=stride, depth_multiplier=1, activation_fn=None, scope='depthwise')
        x = slim.conv2d(x, out_channels // 2, (1, 1), stride=1, scope='conv1x1_after')
        return x, y
