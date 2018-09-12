import tensorflow as tf


SHUFFLE_BUFFER_SIZE = 2048
NUM_PARALLEL_CALLS = 16
NUM_CORES = 8
RESIZE_METHOD = tf.image.ResizeMethod.BILINEAR
EVALUATION_IMAGE_SIZE = 224  # this will be used for validation
MIN_DIMENSION = 256  # when evaluating, resize to this size before doing central crop


class Pipeline:
    def __init__(self, file_pattern, is_training, batch_size, image_size=None):
        """
        Arguments:
            filenames: a string.
            is_training: a boolean.
            batch_size: an integer.
            image_size: an integer or None, it will be used for training only.
        """
        if not is_training:
            assert image_size is None

        self.is_training = is_training
        self.image_size = image_size

        dataset = tf.data.Dataset.list_files(file_pattern, shuffle=is_training)
        if is_training:
            dataset = dataset.repeat()

        def fetch_dataset(filename):
            buffer_size = 8 * 1024 * 1024
            dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
            return dataset

        dataset = dataset.apply(tf.contrib.data.parallel_interleave(
            fetch_dataset, cycle_length=NUM_PARALLEL_CALLS, sloppy=True
        ))

        if is_training:
            dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)

        dataset = dataset.apply(tf.contrib.data.map_and_batch(
            self.parse_and_preprocess, batch_size=batch_size,
            num_parallel_batches=NUM_CORES, drop_remainder=True
        ))

        # for some reason this is needed for TPU
        dataset = dataset.map(
            lambda images, labels: (tf.transpose(images, [1, 2, 3, 0]), labels),
            num_parallel_calls=NUM_CORES
        )  # from NHWC to HWCN

        dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

        self.dataset = dataset

    def parse_and_preprocess(self, example_proto):
        """What this function does:
        1. Parses one record from a tfrecords file and decodes it.
        2. Possibly augments it.

        Returns:
            image: a float16 (half precision) tensor with shape [height, width, 3],
                a RGB image with pixel values in the range [0, 1].
            label: an int tensor with shape [].
        """
        features = {
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'ymin': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'xmin': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'ymax': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'xmax': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True)
        }
        parsed_features = tf.parse_single_example(example_proto, features)

        # get an image as a string, it will be decoded later
        image_as_string = parsed_features['image']

        # get a label
        label = tf.to_int32(parsed_features['label'])

        if self.is_training:

            # get groundtruth boxes, they must be in from-zero-to-one format,
            # also, it assumed that ymin < ymax and xmin < xmax
            boxes = tf.stack([
                parsed_features['ymin'], parsed_features['xmin'],
                parsed_features['ymax'], parsed_features['xmax']
            ], axis=1)
            boxes = tf.to_float(boxes)  # shape [num_boxes, 4]
            # they are only used for data augmentation

            image = get_random_crop(image_as_string, boxes)
            image = tf.image.random_flip_left_right(image)
            image = tf.image.resize_images(
                image, [self.image_size, self.image_size],
                method=RESIZE_METHOD
            )  # has type float32

            image = (1.0 / 255.0) * image  # to [0, 1] range
            image = random_color_manipulations(image, probability=0.75, grayscale_probability=0.05)
            image.set_shape([self.image_size, self.image_size, 3])
        else:
            image = tf.image.decode_jpeg(image_as_string, channels=3)  # has uint8 type
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)  # convert to [0, 1] range
            image = resize_keeping_aspect_ratio(image, MIN_DIMENSION)
            image = central_crop(image, crop_height=EVALUATION_IMAGE_SIZE, crop_width=EVALUATION_IMAGE_SIZE)
            image.set_shape([EVALUATION_IMAGE_SIZE, EVALUATION_IMAGE_SIZE, 3])

        # we do mixed precision training
        image = tf.cast(image, dtype=tf.bfloat16)

        # in the format required by tf.estimator, they will be batched later
        features, labels = image, label
        return features, labels


def resize_keeping_aspect_ratio(image, min_dimension):
    """
    Arguments:
        image: a float tensor with shape [height, width, 3].
        min_dimension: an int tensor with shape [].
    Returns:
        a float tensor with shape [new_height, new_width, 3],
            where min_dimension = min(new_height, new_width).
    """
    image_shape = tf.shape(image)
    height = tf.to_float(image_shape[0])
    width = tf.to_float(image_shape[1])

    original_min_dim = tf.minimum(height, width)
    scale_factor = tf.to_float(min_dimension) / original_min_dim
    new_height = tf.round(height * scale_factor)
    new_width = tf.round(width * scale_factor)

    new_size = [tf.to_int32(new_height), tf.to_int32(new_width)]
    image = tf.image.resize_images(image, new_size, method=RESIZE_METHOD)
    return image


def central_crop(image, crop_height, crop_width):
    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    # min(height, width) = 256,
    # crop_height = crop_width = 224

    amount_to_be_cropped_h = height - crop_height
    amount_to_be_cropped_w = width - crop_width
    crop_top = amount_to_be_cropped_h // 2
    crop_left = amount_to_be_cropped_w // 2
    # so it will have equal margins on left-right and top-bottom

    return tf.slice(image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])


def get_random_crop(image_as_string, boxes):

    distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.image.extract_jpeg_shape(image_as_string),
        bounding_boxes=tf.expand_dims(boxes, axis=0),
        min_object_covered=0.25,
        aspect_ratio_range=[0.75, 1.33],
        area_range=[0.08, 1.0],
        max_attempts=100,
        use_image_if_no_bounding_boxes=True
    )
    begin, size, _ = distorted_bounding_box
    offset_y, offset_x, _ = tf.unstack(begin)
    target_height, target_width, _ = tf.unstack(size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])

    crop = tf.image.decode_and_crop_jpeg(image_as_string, crop_window, channels=3)
    return crop  # has uint8 type


def random_color_manipulations(image, probability=0.9, grayscale_probability=0.1):

    def manipulate(image):
        br_delta = tf.random_uniform([], -32.0/255.0, 32.0/255.0)
        cb_factor = tf.random_uniform([], -0.1, 0.1)
        cr_factor = tf.random_uniform([], -0.1, 0.1)
        channels = tf.split(axis=2, num_or_size_splits=3, value=image)
        red_offset = 1.402 * cr_factor + br_delta
        green_offset = -0.344136 * cb_factor - 0.714136 * cr_factor + br_delta
        blue_offset = 1.772 * cb_factor + br_delta
        channels[0] += red_offset
        channels[1] += green_offset
        channels[2] += blue_offset
        image = tf.concat(axis=2, values=channels)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image

    def to_grayscale(image):
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.grayscale_to_rgb(image)
        return image

    with tf.name_scope('random_color_manipulations'):
        do_it = tf.less(tf.random_uniform([]), probability)
        image = tf.cond(do_it, lambda: manipulate(image), lambda: image)

    with tf.name_scope('to_grayscale'):
        do_it = tf.less(tf.random_uniform([]), grayscale_probability)
        image = tf.cond(do_it, lambda: to_grayscale(image), lambda: image)

    return image
