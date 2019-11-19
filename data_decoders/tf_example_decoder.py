# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tensorflow Example proto decoder for object detection.

A decoder to decode string tensors containing serialized tensorflow.Example
protos for object detection.
"""
import tensorflow as tf
from tensorflow.contrib.slim.python.slim.data.tfexample_decoder import ItemHandler
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops

from core import data_decoder
from core import standard_fields as fields

slim_example_decoder = tf.contrib.slim.tfexample_decoder


class Image(ItemHandler):
    """An ItemHandler that decodes a parsed Tensor as an image."""

    def __init__(self,
                 channels=3,
                 dtype=dtypes.uint8,
                 repeated=False):
        """Initializes the image.

        Args:
          channels: the number of channels in the image.
          dtype: images will be decoded at this bit depth. Different formats
            support different bit depths. See tf.image.decode_image, tf.decode_raw,
          repeated: if False, decodes a single image. If True, decodes a
            variable number of image strings from a 1D tensor of strings.
        """

        self._image_key = 'image/encoded'
        self._format_key = 'image/format'
        self._height_key = 'image/height'
        self._width_key = 'image/width'
        self._channels = channels
        self._dtype = dtype
        self._repeated = repeated
        super(Image, self).__init__([self._image_key, self._format_key, self._height_key, self._width_key])

    def tensors_to_item(self, keys_to_tensors):
        """See base class."""
        image_buffer = keys_to_tensors[self._image_key]
        image_format = keys_to_tensors[self._format_key]
        image_height = keys_to_tensors[self._height_key]
        image_width = keys_to_tensors[self._width_key]

        if self._repeated:
            return functional_ops.map_fn(lambda x: self._decode(x, image_format, image_height, image_width),
                                         image_buffer, dtype=self._dtype)
        else:
            return self._decode(image_buffer, image_format, image_height, image_width)

    def _decode(self, image_buffer, image_format, image_height, image_width):
        """Decodes the image buffer.

        Args:
          image_buffer: The tensor representing the encoded image tensor.
          image_format: The image format for the image in `image_buffer`. If image
            format is `raw`, all images are expected to be in this format, otherwise
            this op can decode a mix of `jpg` and `png` formats.

        Returns:
          A tensor that represents decoded image of self._shape, or
          (?, ?, self._channels) if self._shape is not specified.
        """

        def decode_image():
            """Decodes a png or jpg based on the headers."""
            return image_ops.decode_image(image_buffer, self._channels)

        def decode_raw():
            """Decodes a raw image."""
            return parsing_ops.decode_raw(image_buffer, out_type=self._dtype)

        pred_fn_pairs = {
            math_ops.logical_or(
                math_ops.equal(image_format, 'raw'),
                math_ops.equal(image_format, 'RAW')): decode_raw,
        }

        if self._dtype == dtypes.uint8:
            image = control_flow_ops.case(pred_fn_pairs, default=decode_image, exclusive=True)
        else:
            image = decode_raw()

        image = array_ops.reshape(image, tf.stack([image_height, image_width, 3]))

        return image


class RotatedBoundingBox(ItemHandler):
    """An ItemHandler that concatenates a set of parsed Tensors to Rotated Bounding Boxes.
    """

    def __init__(self, keys=None, prefix=None):
        """Initialize the rotated bounding box handler.
        Args:
          keys: A list of four key names representing the cy, cx, h, w, ang
          prefix: An optional prefix for each of the rotated bounding box keys.
            If provided, `prefix` is appended to each key in `keys`.
        Raises:
          ValueError: if keys is not `None` and also not a list of exactly 5 keys
        """
        if keys is None:
            keys = ['cy', 'cx', 'h', 'w', 'ang']
        elif len(keys) != 5:
            raise ValueError('Rotated boundingBox expects 5 keys but got {}'.format(len(keys)))
        self._prefix = prefix
        self._keys = keys
        self._full_keys = [prefix + k for k in keys]
        super(RotatedBoundingBox, self).__init__(self._full_keys)

    def tensors_to_item(self, keys_to_tensors):
        """Maps the given dictionary of tensors to a contatenated list of bboxes.
        Args:
          keys_to_tensors: a mapping of TF-Example keys to parsed tensors.
        Returns:
          [num_boxes, 5] tensor of bounding box coordinates,
            i.e. 1 rotated bounding box per row, in order [cy, cx, h, w, ang].
        """
        sides = []
        for key in self._full_keys:
            side = keys_to_tensors[key]
            if isinstance(side, sparse_tensor.SparseTensor):
                side = side.values
            side = array_ops.expand_dims(side, 0)
            sides.append(side)

        rotated_bounding_box = array_ops.concat(sides, 0)
        return array_ops.transpose(rotated_bounding_box)


class TfExampleDecoder(data_decoder.DataDecoder):
    """Tensorflow Example proto decoder."""

    def __init__(self, dtype='uint8'):
        """Constructor sets keys_to_features and items_to_handlers.

         Args:
          image_shape: image shape for raw data format.
        """

        if dtype == 'float32':
            self._dtype = tf.float32
        elif dtype == 'uint16':
            self._dtype = tf.uint16
        else:
            self._dtype = tf.uint8

        self.keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
            'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/key/sha256': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/source_id': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/sensor': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/height': tf.FixedLenFeature((), tf.int64, 1),
            'image/width': tf.FixedLenFeature((), tf.int64, 1),
            'image/gsd': tf.FixedLenFeature((), tf.float32, 1),

            # Object boxes.
            'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),

            # Object rotated boxes.
            'image/object/rbbox/cy': tf.VarLenFeature(tf.float32),
            'image/object/rbbox/cx': tf.VarLenFeature(tf.float32),
            'image/object/rbbox/h': tf.VarLenFeature(tf.float32),
            'image/object/rbbox/w': tf.VarLenFeature(tf.float32),
            'image/object/rbbox/ang': tf.VarLenFeature(tf.float32),

            # Object classes.
            'image/object/class/label': tf.VarLenFeature(tf.int64),
            'image/object/area': tf.VarLenFeature(tf.float32),
            'image/object/is_crowd': tf.VarLenFeature(tf.int64),
            'image/object/difficult': tf.VarLenFeature(tf.int64),

            # Instance masks and classes.
            'image/segmentation/object': tf.VarLenFeature(tf.int64),
            'image/segmentation/object/class': tf.VarLenFeature(tf.int64)
        }
        self.items_to_handlers = {
            fields.InputDataFields.image: Image(dtype=self._dtype),
            fields.InputDataFields.source_id: (
                slim_example_decoder.Tensor('image/source_id')),
            fields.InputDataFields.sensor: (
                slim_example_decoder.Tensor('image/sensor')),
            fields.InputDataFields.key: (
                slim_example_decoder.Tensor('image/key/sha256')),
            fields.InputDataFields.filename: (
                slim_example_decoder.Tensor('image/filename')),
            fields.InputDataFields.gsd: (
                slim_example_decoder.Tensor('image/gsd')),
            # Object boxes.
            fields.InputDataFields.groundtruth_boxes: (
                slim_example_decoder.BoundingBox(
                    ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/')),
            # Object rotated boxes.
            fields.InputDataFields.groundtruth_rboxes: (
                RotatedBoundingBox(
                    ['cy', 'cx', 'h', 'w', 'ang'], 'image/object/rbbox/')),
            # Object classes.
            fields.InputDataFields.groundtruth_classes: (
                slim_example_decoder.Tensor('image/object/class/label')),
            fields.InputDataFields.groundtruth_area: slim_example_decoder.Tensor(
                'image/object/area'),
            fields.InputDataFields.groundtruth_is_crowd: (
                slim_example_decoder.Tensor('image/object/is_crowd')),
            fields.InputDataFields.groundtruth_difficult: (
                slim_example_decoder.Tensor('image/object/difficult')),
            # Instance masks and classes.
            fields.InputDataFields.groundtruth_instance_masks: (
                slim_example_decoder.ItemHandlerCallback(
                    ['image/segmentation/object', 'image/height', 'image/width'],
                    self._reshape_instance_masks)),
            fields.InputDataFields.groundtruth_instance_classes: (
                slim_example_decoder.Tensor('image/segmentation/object/class')),
        }

    def decode(self, tf_example_string_tensor):
        """Decodes serialized tensorflow example and returns a tensor dictionary.

        Args:
          tf_example_string_tensor: a string tensor holding a serialized tensorflow
            example proto.

        Returns:
          A dictionary of the following tensors.
          fields.InputDataFields.image - 3D uint8 tensor of shape [None, None, 3]
            containing image.
          fields.InputDataFields.source_id - string tensor containing original
            image id.
          fields.InputDataFields.key - string tensor with unique sha256 hash key.
          fields.InputDataFields.filename - string tensor with original dataset
            filename.
          fields.InputDataFields.groundtruth_boxes - 2D float32 tensor of shape
            [None, 4] containing box corners.
          fields.InputDataFields.groundtruth_classes - 1D int64 tensor of shape
            [None] containing classes for the boxes.
          fields.InputDataFields.groundtruth_area - 1D float32 tensor of shape
            [None] containing containing object mask area in pixel squared.
          fields.InputDataFields.groundtruth_is_crowd - 1D bool tensor of shape
            [None] indicating if the boxes enclose a crowd.
          fields.InputDataFields.groundtruth_difficult - 1D bool tensor of shape
            [None] indicating if the boxes represent `difficult` instances.
          fields.InputDataFields.groundtruth_instance_masks - 3D int64 tensor of
            shape [None, None, None] containing instance masks.
          fields.InputDataFields.groundtruth_instance_classes - 1D int64 tensor
            of shape [None] containing classes for the instance masks.
          fields.InputDataFields.groundtruth_rboxes - 2D float32 tensor of shape
            [None, 5] containing boxes.
        """

        serialized_example = tf.reshape(tf_example_string_tensor, shape=[])
        decoder = slim_example_decoder.TFExampleDecoder(self.keys_to_features,
                                                        self.items_to_handlers)
        keys = decoder.list_items()
        tensors = decoder.decode(serialized_example, items=keys)
        tensor_dict = dict(zip(keys, tensors))
        is_crowd = fields.InputDataFields.groundtruth_is_crowd
        tensor_dict[is_crowd] = tf.cast(tensor_dict[is_crowd], dtype=tf.bool)
        tensor_dict[fields.InputDataFields.image].set_shape([None, None, 3])
        return tensor_dict

    def _reshape_instance_masks(self, keys_to_tensors):
        """Reshape instance segmentation masks.

        The instance segmentation masks are reshaped to [num_instances, height,
        width] and cast to boolean type to save memory.

        Args:
          keys_to_tensors: a dictionary from keys to tensors.

        Returns:
          A 3-D boolean tensor of shape [num_instances, height, width].
        """
        masks = keys_to_tensors['image/segmentation/object']
        if isinstance(masks, tf.SparseTensor):
            masks = tf.sparse_tensor_to_dense(masks)
        height = keys_to_tensors['image/height']
        width = keys_to_tensors['image/width']
        to_shape = tf.cast(tf.stack([-1, height, width]), tf.int32)

        return tf.cast(tf.reshape(masks, to_shape), tf.bool)
