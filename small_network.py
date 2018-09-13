import tensorflow as tf
from utils import block, batch_norm_relu


def shufflenet(images, is_training, num_classes=1000, depth_multiplier='1.0'):
    """
    This is an implementation of ShuffleNet v2:
    https://arxiv.org/abs/1807.11164

    Arguments:
        images: a float tensor with shape [batch_size, image_height, image_width, 3],
            a batch of RGB images with pixel values in the range [0, 1].
        is_training: a boolean.
        num_classes: an integer.
        depth_multiplier: a string, possible values are '0.5', '1.0', '1.5', and '2.0'.
    Returns:
        a float tensor with shape [batch_size, num_classes].
    """
    possibilities = {'0.5': 48, '1.0': 120, '1.5': 176, '2.0': 224}
    # 116 for '1.0' in the original paper
    initial_depth = possibilities[depth_multiplier]

    with tf.name_scope('standardize_input'):
        x = (2.0 * images) - 1.0

    with tf.variable_scope('ShuffleNetV2'):

        x = tf.layers.conv2d(x, 24, (3, 3), strides=(2, 2), use_bias=False, padding='same', name='Conv1')
        x = batch_norm_relu(x, is_training, name='bn')
        x = tf.layers.max_pooling2d(x, (3, 3), strides=(2, 2), padding='same', name='MaxPool')

        x = block(x, is_training, num_units=4, out_channels=initial_depth, scope='Stage2')
        x = block(x, is_training, num_units=8, scope='Stage3')
        x = block(x, is_training, num_units=4, scope='Stage4')

        final_channels = 1024 if depth_multiplier != '2.0' else 2048
        x = tf.layers.conv2d(x, final_channels, (1, 1), strides=(1, 1), use_bias=False, padding='same', name='Conv5')

    # global average pooling
    x = tf.reduce_mean(x, axis=[1, 2])

    logits = tf.layers.dense(
        x, num_classes, name='classifier',
        kernel_initializer=tf.contrib.layers.xavier_initializer()
    )
    return logits
