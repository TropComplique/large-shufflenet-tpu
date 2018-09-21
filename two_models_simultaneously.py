import tensorflow as tf
from tensorflow.contrib import summary
from tensorflow.contrib.tpu.python.tpu import bfloat16
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
from small_network import shufflenet


MOMENTUM = 0.9
USE_NESTEROV = True
MOVING_AVERAGE_DECAY = 0.995
LABEL_SMOOTHING = 0

DEPTH_MULTIPLIER1 = '1.5'
DEPTH_MULTIPLIER2 = '2.0'


def model_fn(features, labels, mode, params):

    # inference will happen in another way
    assert mode != tf.estimator.ModeKeys.PREDICT

    network1 = lambda images, is_training: shufflenet(
        images, is_training, num_classes=params['num_classes'],
        depth_multiplier=DEPTH_MULTIPLIER1
    )
    network2 = lambda images, is_training: shufflenet(
        images, is_training, num_classes=params['num_classes'],
        depth_multiplier=DEPTH_MULTIPLIER2
    )

    # tensor `features` is a half precision tensor with shape [height, width, 3, batch_size],
    # it represents RGB images with values in [0, 1]

    images = features
    images = tf.transpose(images, [3, 0, 1, 2])  # HWCN to NHWC
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    if params['use_bfloat16']:
        with bfloat16.bfloat16_scope():
            with tf.variable_scope('network1'):
                logits1 = network1(images, is_training)
            with tf.variable_scope('network2'):
                logits2 = network2(images, is_training)
        # to full precision
        logits1 = tf.to_float(logits1)
        logits2 = tf.to_float(logits2)
    else:
        with tf.variable_scope('network1'):
            logits1 = network1(images, is_training)
        with tf.variable_scope('network2'):
            logits2 = network2(images, is_training)

    with tf.name_scope('weight_decay'):
        regularization_loss1 = get_regularization_loss(params['weight_decay'], scope='network1')
        regularization_loss2 = get_regularization_loss(params['weight_decay'], scope='network2')

    with tf.name_scope('cross_entropy'):
        one_hot_labels = tf.one_hot(labels, params['num_classes'])
        cross_entropy1 = tf.losses.softmax_cross_entropy(
            logits=logits1, onehot_labels=one_hot_labels,
            label_smoothing=LABEL_SMOOTHING
        )
        cross_entropy2 = tf.losses.softmax_cross_entropy(
            logits=logits2, onehot_labels=one_hot_labels,
            label_smoothing=LABEL_SMOOTHING
        )

    total_loss1 = cross_entropy1 + regularization_loss1
    total_loss2 = cross_entropy2 + regularization_loss2
    total_loss = total_loss1 + total_loss2

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode, loss=total_loss,
            eval_metrics=(metric_fn, [labels, logits1, logits2])
        )

    assert mode == tf.estimator.ModeKeys.TRAIN
    with tf.variable_scope('learning_rate_schedule'):
        global_step = tf.train.get_global_step()
        learning_rate = get_learning_rate(global_step, params)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops), tf.variable_scope('optimizer'):
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM, use_nesterov=USE_NESTEROV)
        optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)
        train_op = optimizer.minimize(total_loss, global_step)

    with tf.control_dependencies([train_op]), tf.name_scope('ema'):
        ema = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DECAY, num_updates=global_step)
        train_op = ema.apply(tf.trainable_variables())

    with tf.name_scope('train_accuracy_calculation'):
        predictions1 = tf.argmax(logits1, axis=1, output_type=tf.int32)
        predictions2 = tf.argmax(logits2, axis=1, output_type=tf.int32)
        train_accuracy1 = tf.reduce_mean(tf.to_float(tf.equal(labels, predictions1)), axis=0)
        train_accuracy2 = tf.reduce_mean(tf.to_float(tf.equal(labels, predictions2)), axis=0)

    tensors_to_summarize = [
        tf.reshape(global_step, [1]),
        tf.reshape(total_loss1, [1]),
        tf.reshape(total_loss2, [1]),
        tf.reshape(cross_entropy1, [1]),
        tf.reshape(cross_entropy2, [1]),
        tf.reshape(regularization_loss1, [1]),
        tf.reshape(regularization_loss2, [1]),
        tf.reshape(learning_rate, [1]),
        tf.reshape(train_accuracy1, [1]),
        tf.reshape(train_accuracy2, [1])
    ]

    def host_call_fn(
            global_step, total_loss1, total_loss2,
            cross_entropy1, cross_entropy2,
            regularization_loss1, regularization_loss2,
            learning_rate, train_accuracy1, train_accuracy2):

        global_step = global_step[0]
        with summary.create_file_writer(params['model_dir'], max_queue=params['iterations_per_loop']).as_default():
            with summary.always_record_summaries():
                summary.scalar('entire_loss1', total_loss1[0], step=global_step)
                summary.scalar('entire_loss2', total_loss2[0], step=global_step)
                summary.scalar('cross_entropy_loss1', cross_entropy1[0], step=global_step)
                summary.scalar('cross_entropy_loss2', cross_entropy2[0], step=global_step)
                summary.scalar('regularization_loss1', regularization_loss1[0], step=global_step)
                summary.scalar('regularization_loss2', regularization_loss2[0], step=global_step)
                summary.scalar('learning_rate', learning_rate[0], step=global_step)
                summary.scalar('train_accuracy1', train_accuracy1[0], step=global_step)
                summary.scalar('train_accuracy2', train_accuracy2[0], step=global_step)
                return summary.all_summary_ops()

    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode, loss=total_loss, train_op=train_op,
        host_call=(host_call_fn, tensors_to_summarize)
    )


def metric_fn(labels, logits1, logits2):
    """
    Arguments:
        labels: an int tensor with shape [batch_size].
        logits1: a float tensor with shape [batch_size, num_classes].
        logits2: a float tensor with shape [batch_size, num_classes].
    """
    predictions1 = tf.argmax(logits1, axis=1, output_type=tf.int32)
    predictions2 = tf.argmax(logits2, axis=1, output_type=tf.int32)
    accuracy1 = tf.metrics.accuracy(labels, predictions1)
    accuracy2 = tf.metrics.accuracy(labels, predictions2)
    in_top5_1 = tf.to_float(tf.nn.in_top_k(predictions=logits1, targets=labels, k=5))
    in_top5_2 = tf.to_float(tf.nn.in_top_k(predictions=logits2, targets=labels, k=5))
    top5_accuracy1 = tf.metrics.mean(in_top5_1)
    top5_accuracy2 = tf.metrics.mean(in_top5_2)
    return {
        'accuracy1': accuracy1, 'top5_accuracy1': top5_accuracy1,
        'accuracy2': accuracy2, 'top5_accuracy2': top5_accuracy2
    }


def get_learning_rate(global_step, params):

    # in case there is a change in the batch size
    scaler = tf.to_float(params['global_batch_size'] / 1024)

    warm_up_steps = tf.to_int64(params['warm_up_steps'])
    warm_up_lr = tf.to_float(params['warm_up_lr'])  # initial learning rate
    first_normal_lr = tf.to_float(params['initial_learning_rate'])

    # linearly increase learning rate (warm up)
    ratio = tf.to_float(global_step) / tf.to_float(warm_up_steps)
    warm_up_learning_rate = warm_up_lr + ratio * (first_normal_lr - warm_up_lr)

    # normal learning rate schedule
    learning_rate = tf.train.polynomial_decay(
        params['initial_learning_rate'], global_step - warm_up_steps,
        params['decay_steps'], params['end_learning_rate'],
        power=1.0
    )  # linear decay

    is_normal = global_step >= warm_up_steps
    learning_rate = tf.cond(is_normal, lambda: learning_rate, lambda: warm_up_learning_rate)
    return scaler * learning_rate


def get_regularization_loss(weight_decay, scope):
    weights = [
        v for v in tf.trainable_variables()
        if 'weights' in v.name and 'depthwise_weights' not in v.name and scope in v.name
    ]  # we don't penalize depthwise layers
    values = [tf.multiply(weight_decay, tf.nn.l2_loss(w)) for w in weights]
    return tf.add_n(values)


class RestoreMovingAverageHook(tf.train.SessionRunHook):
    def __init__(self, model_dir):
        super(RestoreMovingAverageHook, self).__init__()
        self.model_dir = model_dir

    def begin(self):
        ema = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DECAY)
        variables_to_restore = ema.variables_to_restore()
        self.load_ema = tf.contrib.framework.assign_from_checkpoint_fn(
            tf.train.latest_checkpoint(self.model_dir), variables_to_restore
        )

    def after_create_session(self, sess, coord):
        tf.logging.info('Loading EMA weights...')
        self.load_ema(sess)
