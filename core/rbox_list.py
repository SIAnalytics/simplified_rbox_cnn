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

"""Rotated Bounding Box List definition.
This code referred to 'box_list.py'

RBoxList represents a list of rotated bounding boxes as tensorflow
tensors, where each bounding box is represented as a row of 5 numbers,
[cy, cx, h, w, ang].  It is assumed that all bounding boxes
within a given list correspond to a single image.  See also
rbox_list_ops.py for common box related operations (such as area, iou, etc).

Optionally, users can add additional related fields (such as weights).
We assume the following things to be true about fields:
* they correspond to boxes in the rbox_list along the 0th dimension
* they have inferrable rank at graph construction time
* all dimensions except for possibly the 0th can be inferred
  (i.e., not None) at graph construction time.

Some other notes:
  * Following tensorflow conventions, we use height, width ordering,
  and correspondingly, y,x (or cy, cx, h, w, ang) ordering
  * Tensors are always provided as (flat) [N, 5] tensors.
"""

import tensorflow as tf

from core.box_list import BoxList


class RBoxList(BoxList):
    """RBox collection."""

    def __init__(self, boxes):
        """Constructs rbox collection.

        Notice that the super init(Boxlist) don't call due to a shape of boxes

        Args:
          boxes: a tensor of shape [N, 5] representing box corners

        Raises:
          ValueError: if invalid dimensions for bbox data or if bbox data is not in float32 format.
        """
        if len(boxes.get_shape()) != 2 or boxes.get_shape()[-1] != 5:
            raise ValueError('Invalid dimensions for rbox data.')
        if boxes.dtype != tf.float32:
            raise ValueError('Invalid tensor type: should be tf.float32')
        self.data = {'boxes': boxes}

    def set(self, boxes):
        """Convenience function for setting rbox coordinates.

        Args:
          boxes: a tensor of shape [N, 5] representing rbox.

        Raises:
          ValueError: if invalid dimensions for rbbox data
        """
        if len(boxes.get_shape()) != 2 or boxes.get_shape()[-1] != 5:
            raise ValueError('Invalid dimensions for rbox data.')
        self.data['boxes'] = boxes

    def get_corners(self, separate_xy=False):
        """Computes corners of the boxes.

        Args:
          boxes: a tensor of shape [N, 5] representing rbox.
          separate_xy: Whether to separate x and y of rbox corners

        Returns:
          a tensor with shape [N, 8] or tensors of x and y  with shape [N, 4] representing rbox corners
        """
        [cy, cx, h, w, ang] = tf.split(self.get(), 5, axis=1)
        h = h / 2
        w = w / 2
        cos = tf.cos(ang)
        sin = tf.sin(ang)

        lt_x = cx - w * cos + h * sin
        lt_y = cy - w * sin - h * cos
        rt_x = cx + w * cos + h * sin
        rt_y = cy + w * sin - h * cos
        lb_x = cx - w * cos - h * sin
        lb_y = cy - w * sin + h * cos
        rb_x = cx + w * cos - h * sin
        rb_y = cy + w * sin + h * cos

        if separate_xy:
            x = tf.squeeze(tf.stack([lt_x, rt_x, rb_x, lb_x], 1), [2])
            y = tf.squeeze(tf.stack([lt_y, rt_y, rb_y, lb_y], 1), [2])
            return x, y
        else:
            return tf.squeeze(tf.stack([lt_y, lt_x, rt_y, rt_x, rb_y, rb_x, lb_y, lb_x], 1), [2])
