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

"""Rotated Faster RCNN box coder.
This code referred to 'faster_rcnn_box_coder.py'

Rotated Faster RCNN box coder follows the coding schema described below:
  ref: ROTATED REGION BASED CNN FOR SHIP DETECTION(http://www.escience.cn/system/file?fileId=90265)

  tx = (cos(ang_a)*(x - xa) + sin(ang_a)*(x - xa)) / wa
  ty = (-sin(ang_a)(y - ya) + cos(ang_a)*(y - ya)) / ha
  th = log(h / ha)
  tw = log(w / wa)
  ta = (ang - ang_a) / (pi/2)

  where x, y, w, h, ang, denote the box's center coordinates, width, height and angle
  respectively. Similarly, xa, ya, wa, ha, ang_a denote the anchor's center
  coordinates, width, height and angle. tx, ty, tw, th, ta denote the anchor-encoded
  center, width, height and angle respectively.

  See http://arxiv.org/abs/1506.01497 for details.
"""
import math

import tensorflow as tf

from core import box_coder
from core import rbox_list

EPSILON = 1e-8


class FasterRcnnRBoxCoder(box_coder.BoxCoder):
    """Faster RCNN rbox coder."""

    def __init__(self, scale_factors=None):
        """Constructor for FasterRcnnBoxCoder.

        Args:
          scale_factors: List of 5 positive scalars to scale ty, tx, th, tw, ta.
            If set to None, does not perform scaling. For Faster RCNN,
            the open-source implementation recommends using [10.0, 10.0, 5.0, 5.0].
        """
        if scale_factors:
            assert len(scale_factors) == 5
            for scalar in scale_factors:
                assert scalar > 0
        self._scale_factors = scale_factors

    @property
    def code_size(self):
        return 5

    def _encode(self, boxes, anchors):
        """Encode a rbox collection with respect to anchor collection.

        Args:
          boxes: RBoxList holding N boxes to be encoded.
          anchors: RBoxList of anchors.

        Returns:
          a tensor representing N anchor-encoded rboxes of the format
          [ty, tx, th, tw, ta].
        """
        # Convert anchors to the center coordinate representation.
        ycenter_a, xcenter_a, ha, wa, ang_a = tf.unstack(tf.transpose(anchors.get()))
        ycenter, xcenter, h, w, ang = tf.unstack(tf.transpose(boxes.get()))

        # Avoid NaN in division and log below.
        ha += EPSILON
        wa += EPSILON
        h += EPSILON
        w += EPSILON

        tx = (tf.cos(ang_a) * (xcenter - xcenter_a) + tf.sin(ang_a) * (ycenter - ycenter_a)) / wa
        ty = (-tf.sin(ang_a) * (xcenter - xcenter_a) + tf.cos(ang_a) * (ycenter - ycenter_a)) / ha
        tw = tf.log(w / wa)
        th = tf.log(h / ha)
        ta = (ang-ang_a) / (math.pi / 2)

        # Scales location targets as used in paper for joint training.
        if self._scale_factors:
            ty *= self._scale_factors[0]
            tx *= self._scale_factors[1]
            th *= self._scale_factors[2]
            tw *= self._scale_factors[3]
            ta *= self._scale_factors[4]

        return tf.transpose(tf.stack([ty, tx, th, tw, ta]))

    def _decode(self, rel_codes, anchors):
        """Decode relative codes to rboxes.

        Args:
          rel_codes: a tensor representing N anchor-encoded rboxes.
          anchors: BoxList of anchors.

        Returns:
          boxes: BoxList holding N bounding rboxes.
        """
        ycenter_a, xcenter_a, ha, wa, ang_a = tf.unstack(tf.transpose(anchors.get()))
        ty, tx, th, tw, ta = tf.unstack(tf.transpose(rel_codes))

        if self._scale_factors:
            ty /= self._scale_factors[0]
            tx /= self._scale_factors[1]
            th /= self._scale_factors[2]
            tw /= self._scale_factors[3]
            ta /= self._scale_factors[4]

        ycenter = tx * wa * tf.sin(ang_a) + ty * ha * tf.cos(ang_a) + ycenter_a
        xcenter = tx * wa * tf.cos(ang_a) + ty * ha * -tf.sin(ang_a) + xcenter_a
        w = tf.exp(tw) * wa
        h = tf.exp(th) * ha
        ang = ta * (math.pi / 2) + ang_a

        return rbox_list.RBoxList(tf.transpose(tf.stack([ycenter, xcenter, h, w, ang])))
