"""Preprocess input data.
This code referred to preprocessor.py

A preprocessing function receives a set of inputs,
e.g. an image and bounding boxes,
performs an operation on them, and returns them.

The preprocess function receives a tensor_dict which is a dictionary that maps
different field names to their tensors. For example,
tensor_dict[fields.InputDataFields.image] holds the image tensor.
The image is a rank 4 tensor: [1, height, width, channels] with
dtype=tf.float32. The groundtruth_boxes is a rank 2 tensor: [N, 4] or [N, 5] where
in each row there is a box with [ymin xmin ymax xmax] or [cy, cx, h, w, ang].
Boxes are in normalized coordinates meaning their coordinate values range in [0, 1]

Important Note: In tensor_dict, images is a rank 4 tensor, but preprocessing
functions receive a rank 3 tensor for processing the image. Thus, inside the
preprocess function we squeeze the image to become a rank 3 tensor and then
we pass it to the functions. At the end of the preprocess we expand the image
back to rank 4.
"""

# Author: Jamyoung Koo
# Date created: 2018.02.21
# Date last modified: 2018.05.14
# Python Version: 3.5


import math
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import cv2

from core import standard_fields as fields


RANGE_OF_SENSOR = {
    'K2': [[105, 171, 206], [808, 715, 748]],
    'K3': [[984, 1405, 2772], [4826, 4236, 6581]],
    'K3A': [[695, 1159, 1469], [6219, 6524, 5257]]
}


def swap_height_width_of_rboxes(rboxes):
    """Swap height with width of rboxes

    Args:
      rboxes: rank 2 float32 tensor containing the bounding boxes -> [N, 5].
             Boxes are in normalized form meaning their coordinates vary between [0, 1].
             Each row is in the form of [cy, cx, h, w, ang].

    Returns:
      rboxes swapped height with width
    """
    with tf.name_scope('SwapHeightWidthOfRBoxes', values=[rboxes]):
        cy, cx, h, w, ang = tf.split(value=rboxes, num_or_size_splits=5, axis=1)
        return tf.concat([cy, cx, w, h, ang], 1)


def subtract_angle_of_rboxes(rboxes, angle):
    """Swap height with width of rboxes

    Args:
      rboxes: rank 2 float32 tensor containing the bounding boxes -> [N, 5].
             Boxes are in normalized form meaning their coordinates vary between [0, 1].
             Each row is in the form of [cy, cx, h, w, ang].
      angle: subtracting a angle.

    Returns:
      rboxes subtracted a angle.
    """
    with tf.name_scope('SubtractAngleOfRboxes', values=[rboxes]):
        cy, cx, h, w, _ang = tf.split(value=rboxes, num_or_size_splits=5, axis=1)
        sub_ang = _ang - angle
        sub_ang = tf.where(sub_ang > (math.pi / 2), sub_ang - math.pi, sub_ang)
        sub_ang = tf.where(sub_ang < (-math.pi / 2), sub_ang + math.pi, sub_ang)
        return tf.concat([cy, cx, h, w, sub_ang], 1)


def normalize_tiff(image, sensor='K3'):
    """Normalizes pixel values in the tiff image.
    Moves the pixel values from the RANGE_OF_SENSOR range to a the [0, 255] range.

    Args:
      image: rank 3 float32 tensor containing 1 image -> [height, width, channels].
      sensor: a string of sensor name

    Returns:
      image: image which is the same shape as input image.
    """
    with tf.name_scope('NormalizeTiff', values=[image]):
        original_range = tf.constant([[0, 0, 0], [255, 255, 255]], dtype=tf.float32)
        K3 = tf.constant(RANGE_OF_SENSOR['K3'], dtype=tf.float32)
        K3A = tf.constant(RANGE_OF_SENSOR['K3A'], dtype=tf.float32)
        K2 = tf.constant(RANGE_OF_SENSOR['K2'], dtype=tf.float32)

        original_range = tf.cond(tf.equal(sensor, 'K2'), lambda: K2, lambda: original_range)
        original_range = tf.cond(tf.equal(sensor, 'K3'), lambda: K3, lambda: original_range)
        original_range = tf.cond(tf.equal(sensor, 'K3A'), lambda: K3A, lambda: original_range)

        target_minval = tf.constant([0, 0, 0], dtype=tf.float32)
        target_maxval = tf.constant([255, 255, 255], dtype=tf.float32)

        image = tf.to_float(image)
        image = tf.subtract(image, original_range[0])
        image = tf.multiply(image, (target_maxval - target_minval) / (original_range[1] - original_range[0]))
        image = tf.add(image, target_minval)
        image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)
        return image


def cutoff_max_gts(image, rboxes, classes, max_gts=100):
    """Cut off randomly ground truths to be max_gts of the number of gts

    Args:
      image: rank 3 float32 tensor containing 1 image -> [height, width, channels].
      rboxes: rank 2 float32 tensor containing the bounding boxes -> [N, 5].
             Boxes are in normalized form meaning their coordinates vary between [0, 1].
             Each row is in the form of [cy, cx, h, w, ang].
      classes: rank 1 int32 tensor of shape [num_rboxes] containing the object classes.
      max_gts: maximum ground truths

    Returns:
      rboxes subtracted a angle.
    """

    def _cutoff_gts(image, rboxes, classes):
        half_max_gts = max_gts // 2

        rboxes *= [image.shape[0], image.shape[1], image.shape[0], image.shape[1], 1]

        sorted_indices = rboxes[:, 3].argsort()[::-1]
        sel_rboxes = rboxes[sorted_indices[:half_max_gts]]
        sel_classes = classes[sorted_indices[:half_max_gts]]

        _rboxes = rboxes[sorted_indices[half_max_gts:]]
        _classes = classes[sorted_indices[half_max_gts:]]
        _rboxes, _classes = shuffle(_rboxes, _classes)

        for rbox in _rboxes[half_max_gts:]:
            points = cv2.boxPoints(((rbox[1], rbox[0]), (rbox[3], rbox[2]), math.degrees(rbox[4])))
            cv2.fillConvexPoly(image, points.astype(dtype=np.int32), 0)

        sel_rboxes = np.vstack((sel_rboxes, _rboxes[:half_max_gts]))
        sel_classes = np.hstack((sel_classes, _classes[:half_max_gts]))

        sel_rboxes /= [image.shape[0], image.shape[1], image.shape[0], image.shape[1], 1]

        return image, sel_rboxes, sel_classes

    with tf.name_scope('CutoffMaxGTs', values=[rboxes, classes]):
        num_gts = tf.shape(rboxes)[0]
        do_cutoff = tf.greater(num_gts, max_gts)
        iamge_shape = image.shape
        image, rboxes, classes = tf.cond(do_cutoff,
                                         lambda: tf.py_func(_cutoff_gts, [image, rboxes, classes],
                                                            [tf.float32, tf.float32, tf.int64]),
                                         lambda: [image, rboxes, classes])
        image.set_shape(iamge_shape)
        rboxes.set_shape([max_gts, 5])
        classes.set_shape(max_gts)

        return image, rboxes, classes


def get_default_func_arg_map():
    """Returns the default mapping from a preprocessor input function to its args.

    Returns:
      A map from preprocessing functions to the arguments they receive.
    """
    prep_func_arg_map = {
        swap_height_width_of_rboxes: (fields.InputDataFields.groundtruth_rboxes,),
        subtract_angle_of_rboxes: (fields.InputDataFields.groundtruth_rboxes,),
        normalize_tiff: (fields.InputDataFields.image, fields.InputDataFields.sensor),
        cutoff_max_gts: (fields.InputDataFields.image, fields.InputDataFields.groundtruth_rboxes,
                         fields.InputDataFields.groundtruth_classes),
    }

    return prep_func_arg_map


def preprocess(tensor_dict, preprocess_options, func_arg_map=None):
    """Preprocess images and bounding boxes.

    Various types of preprocessing (to be implemented) based on the preprocess_options dictionary

    Args:
      tensor_dict: dictionary that contains images, boxes, and can contain other
                   things as well.
                   images-> rank 4 float32 tensor contains
                            1 image -> [1, height, width, 3].
                            with pixel values varying between [0, 1]
                   boxes-> rank 2 float32 tensor containing
                           the bounding boxes -> [N, 5].
                           Boxes are in normalized form meaning
                           their coordinates vary between [0, 1].
                           Each row is in the form
                           of [cy, cx, h, w, ang].
      preprocess_options: It is a list of tuples, where each tuple contains a
                          function and a dictionary that contains arguments and
                          their values.
      func_arg_map: mapping from preprocessing functions to arguments that they
                    expect to receive and return.

    Returns:
      tensor_dict: which contains the preprocessed images, rboxes, etc.

    Raises:
      ValueError: (a) If the functions passed to Preprocess are not in func_arg_map.
                  (b) If the arguments that a function needs do not exist in tensor_dict.
                  (c) If image in tensor_dict is not rank 4
    """
    if func_arg_map is None:
        func_arg_map = get_default_func_arg_map()

    # changes the images to image (rank 4 to rank 3) since the functions
    # receive rank 3 tensor for image
    if fields.InputDataFields.image in tensor_dict:
        images = tensor_dict[fields.InputDataFields.image]
        if len(images.get_shape()) != 4:
            raise ValueError('images in tensor_dict should be rank 4')
        image = tf.squeeze(images, squeeze_dims=[0])
        tensor_dict[fields.InputDataFields.image] = image

    # Preprocess inputs based on preprocess_options
    for option in preprocess_options:
        func, params = option
        if func not in func_arg_map:
            raise ValueError('The function %s does not exist in func_arg_map' % (func.__name__))
        arg_names = func_arg_map[func]
        for a in arg_names:
            if a is not None and a not in tensor_dict:
                raise ValueError('The function %s requires argument %s' % (func.__name__, a))

        def get_arg(key):
            return tensor_dict[key] if key is not None else None

        args = [get_arg(a) for a in arg_names]
        results = func(*args, **params)
        if not isinstance(results, (list, tuple)):
            results = (results,)

        # Removes None args since the return values will not contain those.
        arg_names = [arg_name for arg_name in arg_names if arg_name is not None]
        for res, arg_name in zip(results, arg_names):
            tensor_dict[arg_name] = res

    # changes the image to images (rank 3 to rank 4) to be compatible to what
    # we received in the first place
    if fields.InputDataFields.image in tensor_dict:
        image = tensor_dict[fields.InputDataFields.image]
        images = tf.expand_dims(image, 0)
        tensor_dict[fields.InputDataFields.image] = images

    return tensor_dict
