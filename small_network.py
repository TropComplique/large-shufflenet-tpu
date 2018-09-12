import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import block


BATCH_NORM_MOMENTUM = 0.9
BATCH_NORM_EPSILON = 1e-3


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
    possibilities = {'0.5': 48, '1.0': 116, '1.5': 176, '2.0': 224}
    initial_depth = possibilities[depth_multiplier]

    def batch_norm(x):
        x = tf.layers.batch_normalization(
            x, axis=3, center=True, scale=True,
            training=is_training,
            momentum=BATCH_NORM_MOMENTUM,
            epsilon=BATCH_NORM_EPSILON,
            fused=True, name='batch_norm'
        )
        return x

    with tf.name_scope('standardize_input'):
        x = (2.0 * images) - 1.0

    with tf.variable_scope('ShuffleNetV2'):
        params = {
            'padding': 'SAME', 'activation_fn': tf.nn.relu,
            'normalizer_fn': batch_norm, 'data_format': 'NHWC',
            'weights_initializer': tf.contrib.layers.xavier_initializer()
        }
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], **params):

            x = slim.conv2d(x, 24, (3, 3), stride=2, scope='Conv1')
            x = slim.max_pool2d(x, (3, 3), stride=2, padding='SAME', scope='MaxPool')

            x = block(x, num_units=4, out_channels=initial_depth, scope='Stage2')
            x = block(x, num_units=8, scope='Stage3')
            x = block(x, num_units=4, scope='Stage4')

            final_channels = 1024 if depth_multiplier != '2.0' else 2048
            x = slim.conv2d(x, final_channels, (1, 1), stride=1, scope='Conv5')

    # global average pooling
    x = tf.reduce_mean(x, axis=[1, 2])

    logits = slim.fully_connected(
        x, num_classes, activation_fn=None, scope='classifier',
        weights_initializer=tf.contrib.layers.xavier_initializer()
    )
    return logits
