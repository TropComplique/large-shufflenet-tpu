import tensorflow as tf
import os
from model import model_fn, RestoreMovingAverageHook
from input_pipeline import Pipeline
tf.logging.set_verbosity('INFO')


"""
The purpose of this script is to train a network.
Evaluation will happen periodically.

To use it just run:
python train.py
"""

# 1281144/1024 = 1251.12
# so 1 epoch = 1251 steps

TPU = ''
TPU_ZONE = ''
GCP_PROJECT = ''

BATCH_SIZE = 1024
VALIDATION_BATCH_SIZE = 1024  # some images will be excluded
NUM_EPOCHS = 90
TRAIN_DATASET_SIZE = 1281144

steps_per_eval = 1251
iterations_per_loop = 1251
NUM_STEPS = NUM_EPOCHS * (TRAIN_DATASET_SIZE // BATCH_SIZE)  # 112590


PARAMS = {
    'train_dataset_path': '/mnt/datasets/imagenet/train_shards/',
    'val_dataset_path': '/mnt/datasets/imagenet/val_shards/',

    # linear learning rate schedule
    'initial_learning_rate': 0.5,
    'decay_steps': NUM_STEPS,
    'end_learning_rate': 1e-6,

    'model_dir': 'models/run00/',
    'num_classes': 1000,
    'depth_multiplier': '1.0',
    'weight_decay': 4e-5,

    'batch_size': BATCH_SIZE,

    'warm_up_steps': 4,
    'warm_up_lr': 1e-6,

    'lr_boundaries': [4, 21, 35, 43],
    'lr_values': [1.0, 0.1, 0.01, 0.001],

    'train_steps': NUM_STEPS,

    'iterations_per_loop': 1251
}


def get_input_fn(is_training, image_size=None):

    dataset_path = PARAMS['train_dataset_path'] if is_training else PARAMS['val_dataset_path']
    filenames = os.listdir(dataset_path)
    filenames = [n for n in filenames if n.endswith('.tfrecords')]
    filenames = [os.path.join(dataset_path, n) for n in sorted(filenames)]

    def input_fn():
        pipeline = Pipeline(
            filenames, is_training, image_size=image_size,
            batch_size=BATCH_SIZE if is_training else VALIDATION_BATCH_SIZE
        )
        return pipeline.dataset

    return input_fn


tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
    TPU, zone=TPU_ZONE,
    project=GCP_PROJECT
)
tpu_config = tf.contrib.tpu.TPUConfig(
    iterations_per_loop=params['iterations_per_loop'],
    per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
)
config = tf.contrib.tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    model_dir=MODEL_DIR, save_checkpoints_steps=FLAGS.iterations_per_loop,
    keep_checkpoint_max=5, tpu_config=tpu_config,
    save_checkpoints_steps=max(600, FLAGS.iterations_per_loop),
    log_step_count_steps=1000,
)


estimator = tf.contrib.tpu.TPUEstimator(
    model_fn=model_fn, model_dir=params['model_dir'],
    params=params, config=config,
    train_batch_size=BATCH_SIZE,
    eval_batch_size=VALIDATION_BATCH_SIZE
)


imagenet_train_small = get_input_fn(is_training=True, image_size=128)
imagenet_train_medium = get_input_fn(is_training=True, image_size=224)
imagenet_train_large = get_input_fn(is_training=True, image_size=288)
imagenet_eval = get_input_fn(is_training=False)  # for validation we use size 224


resnet_classifier.train(input_fn=imagenet_train_small, max_steps=18 * 1251)
resnet_classifier.train(input_fn=imagenet_train_medium, max_steps=41 * 1251)
resnet_classifier.train(input_fn=imagenet_train_large, max_steps=min(50 * 1251, FLAGS.train_steps))


steps_per_epoch = FLAGS.num_train_images // FLAGS.train_batch_size
eval_steps = FLAGS.num_eval_images // FLAGS.eval_batch_size

current_step = estimator._load_global_step_from_checkpoint_dir(FLAGS.model_dir)
steps_per_epoch = FLAGS.num_train_images // FLAGS.train_batch_size
while current_step < FLAGS.train_steps:
    # Train for up to steps_per_eval number of steps.
    # At the end of training, a checkpoint will be written to --model_dir.
    next_checkpoint = min(current_step + FLAGS.steps_per_eval,
                          FLAGS.train_steps)
    resnet_classifier.train(
        input_fn=imagenet_train.input_fn, max_steps=next_checkpoint)
    current_step = next_checkpoint

    eval_results = resnet_classifier.evaluate(
        input_fn=imagenet_eval.input_fn,
        steps=FLAGS.num_eval_images // FLAGS.eval_batch_size)
