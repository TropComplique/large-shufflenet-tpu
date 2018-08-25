import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import depthwise_conv, block


BATCH_NORM_MOMENTUM = 0.9
BATCH_NORM_EPSILON = 1e-5


def large_shufflenet(images, is_training, num_classes=1000):
    """
    This is an implementation of ShuffleNet v2-50:
    https://arxiv.org/abs/1807.11164
    It is a more fast/accurate alternative to ResNet-50.

    Arguments:
        images: a float tensor with shape [batch_size, image_height, image_width, 3],
            a batch of RGB images with pixel values in the range [0, 1].
        is_training: a boolean.
        num_classes: an integer.
    Returns:
        a float tensor with shape [batch_size, num_classes].
    """
    initial_depth = 244

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

    with tf.variable_scope('ShuffleNetV2-50'):
        params = {
            'padding': 'SAME', 'activation_fn': tf.nn.relu,
            'normalizer_fn': batch_norm, 'data_format': 'NHWC',
            'weights_initializer': tf.contrib.layers.xavier_initializer()
        }
        with slim.arg_scope([slim.conv2d, depthwise_conv], **params):

            x = slim.conv2d(x, 64, (3, 3), stride=2, scope='Conv1')
            x = slim.max_pool2d(x, (3, 3), stride=2, padding='SAME', scope='MaxPool')

            x = block(x, num_units=3, out_channels=initial_depth, downsample=False, scope='Stage2')
            x = block(x, num_units=4, scope='Stage3')
            x = block(x, num_units=6, scope='Stage4')
            x = block(x, num_units=3, scope='Stage5')

            x = slim.conv2d(x, 2048, (1, 1), stride=1, scope='Conv6')

            # number of conv layers in the network:
            # 1 + (3 + 4 + 6 + 3)*3 + 1 = 50

    # global average pooling
    x = tf.reduce_mean(x, axis=[1, 2])

    logits = slim.fully_connected(
        x, num_classes, activation_fn=None, scope='classifier',
        weights_initializer=tf.contrib.layers.xavier_initializer()
    )
    return logits
