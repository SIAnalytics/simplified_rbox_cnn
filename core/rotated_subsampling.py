import math
import functools
import tensorflow as tf


def rotated_subsampling(image, boxes, box_ind, crop_size):
    """Spatial Subsampling for Rotated Bounding Box

    Implements a image-like tensor rotated subsampling as described in [2].

    Parameters
    ----------
    image : float
        [num_batch, H, W, channels]
    boxes: List[float}
        The output of the localisation network
        [num_boxes, 5].
        The second channel includes rotated bounding box information for each instance.
        The order of second channel information : (y, x, h, w, a)... same notation with [2].
        Each information should be already normalized in [0, 1] scales. e.g) (y/H, x/W, h/H, w/W, a)
        Unit of a is radian.
        ex) [[0.4, 0.6, 0.3, 0.6, pi/4], ...]
    box_ind: List[int]
        A Tensor of type int32.
        A 1-D tensor of shape [num_boxes] with int32 values in [0, num_batch).
        The value of box_ind[i] specifies the image that the i-th box refers to.
    original_size : List[int]
        [H, W]
    crop_size : List[int]
        [new_H, new_W]

    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  ROTATED REGION BASED CNN FOR SHIP DETECTION
            Zikun Liu, Jingao Hu, Lubin Weng, Yiping Yang
    """

    def _rotated_subsampling(boxes_and_ind, image, crop_size):
        with tf.variable_scope('rotated_subsampling'):
            boxes = boxes_and_ind[0]
            box_ind = boxes_and_ind[1]
            image_shape = tf.cast(tf.shape(image), dtype=tf.float32)
            original_size = (image_shape[1], image_shape[2])

            image_reindex = tf.gather(image, box_ind, axis=0)

            size_max = tf.cast(tf.maximum(tf.sqrt(2.) * original_size[0], tf.sqrt(2.) * original_size[1]), tf.int32)
            transformed_images = tf.image.resize_image_with_crop_or_pad(image_reindex, size_max, size_max)

            translation_matrix = tf.concat([
                tf.ones((tf.shape(boxes)[0], 1), tf.float32),
                tf.zeros((tf.shape(boxes)[0], 1), tf.float32),
                boxes[:, 1, None] * original_size[1] - tf.cast(original_size[1] / 2, tf.float32),
                tf.zeros((tf.shape(boxes)[0], 1), tf.float32),
                tf.ones((tf.shape(boxes)[0], 1), tf.float32),
                boxes[:, 0, None] * original_size[0] - tf.cast(original_size[0] / 2, tf.float32),
                tf.zeros((tf.shape(boxes)[0], 2), tf.float32),
            ],
                axis=1)

            rotation_matrix = tf.contrib.image.angles_to_projective_transforms(boxes[:, 4],
                                                                               tf.cast(size_max, tf.float32),
                                                                               tf.cast(size_max, tf.float32)
                                                                               )

            composed_matrix = tf.contrib.image.compose_transforms(translation_matrix, rotation_matrix)

            transformed_images = tf.contrib.image.transform(transformed_images, composed_matrix,
                                                            interpolation='BILINEAR')

            height_ratio = (boxes[:, 2] * original_size[0]) / tf.cast(size_max, tf.float32)
            width_ratio = (boxes[:, 3] * original_size[1]) / tf.cast(size_max, tf.float32)
            transformed_images = tf.image.crop_and_resize(transformed_images,
                                                          tf.stack([0.5 - height_ratio / 2, 0.5 - width_ratio / 2,
                                                                    0.5 + height_ratio / 2, 0.5 + width_ratio / 2],
                                                                   axis=1),
                                                          tf.range(tf.shape(boxes)[0]),
                                                          crop_size)

        return transformed_images

    def _rotated_subsampling_map(boxes, box_ind, image, crop_size):
        boxes = tf.expand_dims(boxes, axis=1)
        box_ind = tf.expand_dims(box_ind, axis=1)
        transformed_images = tf.map_fn(
            functools.partial(_rotated_subsampling, image=image, crop_size=crop_size), elems=[boxes, box_ind],
            dtype=tf.float32, parallel_iterations=300)
        return tf.squeeze(transformed_images)

    max_boxes = 300
    n_boxes = tf.shape(boxes)[0]
    n_depth = image.shape.as_list()[-1]
    do_map_fn = tf.greater(n_boxes, max_boxes)
    transformed_images = tf.cond(do_map_fn, lambda: _rotated_subsampling_map(boxes, box_ind, image, crop_size),
                                 lambda: _rotated_subsampling([boxes, box_ind], image, crop_size))
    transformed_images = tf.reshape(transformed_images, (n_boxes, crop_size[0], crop_size[1], n_depth))
    return transformed_images


def rotated_subsampling_diagonal(image, boxes, box_ind, crop_size):
    """Spatial Subsampling for Rotated Bounding Box

    Implements a image-like tensor rotated subsampling as described in [2].

    Parameters
    ----------
    image : float
        [num_batch, H, W, channels]
    boxes: List[float}
        The output of the localisation network
        [num_boxes, 5].
        The second channel includes rotated bounding box information for each instance.
        The order of second channel information : (y, x, h, w, a)... same notation with [2].
        Each information should be already normalized in [0, 1] scales. e.g) (y/H, x/W, h/H, w/W, a)
        Unit of a is radian.
        ex) [[0.4, 0.6, 0.3, 0.6, pi/4], ...]
    box_ind: List[int]
        A Tensor of type int32.
        A 1-D tensor of shape [num_boxes] with int32 values in [0, num_batch).
        The value of box_ind[i] specifies the image that the i-th box refers to.
    original_size : List[int]
        [H, W]
    crop_size : List[int]
        [new_H, new_W]

    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  ROTATED REGION BASED CNN FOR SHIP DETECTION
            Zikun Liu, Jingao Hu, Lubin Weng, Yiping Yang
    """

    with tf.variable_scope('rotated_subsampling'):
        image_shape = tf.cast(tf.shape(image), dtype=tf.float32)
        original_size = (image_shape[1], image_shape[2])

        image_reindex = tf.gather(image, box_ind, axis=0)

        size_max = tf.cast(tf.maximum(tf.sqrt(2.) * original_size[0], tf.sqrt(2.) * original_size[1]), tf.int32)
        transformed_images = tf.image.resize_image_with_crop_or_pad(image_reindex, size_max, size_max)

        translation_matrix = tf.concat([
            tf.ones((tf.shape(boxes)[0], 1), tf.float32),
            tf.zeros((tf.shape(boxes)[0], 1), tf.float32),
            boxes[:, 1, None] * original_size[1] - tf.cast(original_size[1] / 2, tf.float32),
            tf.zeros((tf.shape(boxes)[0], 1), tf.float32),
            tf.ones((tf.shape(boxes)[0], 1), tf.float32),
            boxes[:, 0, None] * original_size[0] - tf.cast(original_size[0] / 2, tf.float32),
            tf.zeros((tf.shape(boxes)[0], 2), tf.float32)],
            axis=1)

        digonal_angle = boxes[:, 4] + math.pi / 4
        rotation_matrix = tf.contrib.image.angles_to_projective_transforms(digonal_angle,
                                                                           tf.cast(size_max, tf.float32),
                                                                           tf.cast(size_max, tf.float32)
                                                                           )
        composed_matrix = tf.contrib.image.compose_transforms(translation_matrix, rotation_matrix)

        transformed_images = tf.contrib.image.transform(transformed_images, composed_matrix, interpolation='BILINEAR')

        height = boxes[:, 2] * original_size[0]
        width = boxes[:, 3] * original_size[1]

        digoanl_height = (height / 2) / (tf.sin(math.pi / 4)) + tf.cos(math.pi / 4) * width
        digoanl_height_ratio = digoanl_height / tf.cast(size_max, tf.float32)
        transformed_images = tf.image.crop_and_resize(transformed_images,
                                                      tf.stack([0.5 - digoanl_height_ratio / 2,
                                                                0.5 - digoanl_height_ratio / 2,
                                                                0.5 + digoanl_height_ratio / 2,
                                                                0.5 + digoanl_height_ratio / 2],
                                                               axis=1),
                                                      tf.range(tf.shape(boxes)[0]),
                                                      crop_size)

    return transformed_images
