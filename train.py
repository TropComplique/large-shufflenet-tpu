import tensorflow as tf
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
VAL_DATASET_SIZE = 49999
NUM_STEPS_PER_EPOCH = TRAIN_DATASET_SIZE // BATCH_SIZE
NUM_STEPS = NUM_EPOCHS * NUM_STEPS_PER_EPOCH  # 112590

# Controls how often evaluation is performed.
STEPS_PER_EVAL = 1251  # evaluate after every epoch

# Number of steps to run on TPU before outfeeding metrics to the CPU.
ITERATIONS_PER_LOOP = 1251  # number before returning to CPU

PARAMS = {
    'train_file_pattern': 'gs://imagenetdata/train_shards/shard-*',
    'val_file_pattern': 'gs://imagenetdata/val_shards/shard-*',
    'model_dir': 'gs://imagenetdata/models/run00/',

    'num_classes': 1000,
    'depth_multiplier': '1.0',
    'weight_decay': 4e-5,

    'global_batch_size': BATCH_SIZE,

    'warm_up_steps': 4 * NUM_STEPS_PER_EPOCH,
    'warm_up_lr': 1e-6,

    'lr_boundaries': [n * NUM_STEPS_PER_EPOCH for n in [21, 35, 43]],
    'lr_values': [1.0, 0.1, 0.01, 0.001],

    # linear learning rate schedule
    # 'initial_learning_rate': 0.5,
    # 'decay_steps': NUM_STEPS,
    # 'end_learning_rate': 1e-6,

    'iterations_per_loop': ITERATIONS_PER_LOOP
}


def get_input_fn(is_training, image_size=None):

    file_pattern = PARAMS['train_file_pattern'] if is_training else PARAMS['val_file_pattern']

    def input_fn():
        pipeline = Pipeline(
            file_pattern, is_training, image_size=image_size,
            batch_size=BATCH_SIZE if is_training else VALIDATION_BATCH_SIZE
        )
        return pipeline.dataset

    return input_fn


tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
    TPU, zone=TPU_ZONE,
    project=GCP_PROJECT
)
tpu_config = tf.contrib.tpu.TPUConfig(
    iterations_per_loop=ITERATIONS_PER_LOOP,
    per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
)
config = tf.contrib.tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    model_dir=MODEL_DIR, save_checkpoints_steps=ITERATIONS_PER_LOOP,
    keep_checkpoint_max=5, tpu_config=tpu_config,
    log_step_count_steps=500
)


estimator = tf.contrib.tpu.TPUEstimator(
    model_fn=model_fn, model_dir=PARAMS['model_dir'],
    params=params, config=config,
    train_batch_size=BATCH_SIZE,
    eval_batch_size=VALIDATION_BATCH_SIZE
)


train_small_input_fn = get_input_fn(is_training=True, image_size=128)
train_medium_input_fn = get_input_fn(is_training=True, image_size=224)
train_large_input_fn = get_input_fn(is_training=True, image_size=288)
eval_input_fn = get_input_fn(is_training=False)  # for validation we use size 224


def load_global_step_from_checkpoint_dir(checkpoint_dir):
    try:
        checkpoint_reader = tf.train.NewCheckpointReader(
            tf.train.latest_checkpoint(checkpoint_dir)
        )
        return checkpoint_reader.get_tensor(tf.GraphKeys.GLOBAL_STEP)
    except:
        return 0



tf.logging.info(
    'Training for %d steps (%d epochs in total). Current step %d.',
    NUM_STEPS, NUM_EPOCHS, current_step
)


def train_and_eval(input_fn, end_step):

    # load last checkpoint and start from there
    current_step = load_global_step_from_checkpoint_dir(output_dir)

    while current_step < end_step:

        next_checkpoint = min(current_step + STEPS_PER_EVAL, end_step)
        estimator.train(input_fn=input_fn, max_steps=next_checkpoint)
        current_step = next_checkpoint

        tf.logging.info('Starting to evaluate.')
        eval_results = estimator.evaluate(
            input_fn=eval_input_fn,
            steps=VAL_DATASET_SIZE // VALIDATION_BATCH_SIZE
        )
        tf.logging.info('Eval results at step %d: %s', next_checkpoint, eval_results)


train_and_eval(train_small_input_fn, 18 * NUM_STEPS_PER_EPOCH)
train_and_eval(train_medium_input_fn, 41 * NUM_STEPS_PER_EPOCH)
train_and_eval(train_large_input_fn, NUM_EPOCHS * NUM_STEPS_PER_EPOCH)
