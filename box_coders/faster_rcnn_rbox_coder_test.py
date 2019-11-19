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

"""Tests for object_detection.box_coder.faster_rcnn_rbox_coder."""

import tensorflow as tf

from box_coders import faster_rcnn_rbox_coder
from core import rbox_list


class FasterRcnnRBoxCoderTest(tf.test.TestCase):
    def test_get_correct_relative_codes_after_encoding(self):
        boxes = [[10.0, 10.0, 20.0, 15.0, 0.1], [0.2, 0.1, 0.5, 0.4, 0.2]]
        anchors = [[15.0, 12.0, 30.0, 18.0, 0.15], [0.1, 0.0, 0.7, 0.9, 0.1]]
        expected_rel_codes = [[-0.154833, -0.151374, -0.405465, -0.182322, -0.031831],
                              [0.127882, 0.121649, -0.336472, -0.810930, 0.063662]]

        boxes = rbox_list.RBoxList(tf.constant(boxes))
        anchors = rbox_list.RBoxList(tf.constant(anchors))
        coder = faster_rcnn_rbox_coder.FasterRcnnRBoxCoder()
        rel_codes = coder.encode(boxes, anchors)
        with self.test_session() as sess:
            rel_codes_out, = sess.run([rel_codes])
            self.assertAllClose(rel_codes_out, expected_rel_codes)

    def test_get_correct_relative_codes_after_encoding_with_scaling(self):
        boxes = [[10.0, 10.0, 20.0, 15.0, 0.1], [0.2, 0.1, 0.5, 0.4, 0.2]]
        anchors = [[15.0, 12.0, 30.0, 18.0, 0.15], [0.1, 0.0, 0.7, 0.9, 0.1]]
        scale_factors = [2, 3, 4, 5, 6]
        expected_rel_codes = [[-0.309665, -0.454122, -1.621860, -0.911607, -0.190986],
                              [0.2557630, 0.364945, -1.345888, -4.054651, 0.381972]]

        boxes = rbox_list.RBoxList(tf.constant(boxes))
        anchors = rbox_list.RBoxList(tf.constant(anchors))
        coder = faster_rcnn_rbox_coder.FasterRcnnRBoxCoder(scale_factors=scale_factors)
        rel_codes = coder.encode(boxes, anchors)
        with self.test_session() as sess:
            rel_codes_out, = sess.run([rel_codes])
            self.assertAllClose(rel_codes_out, expected_rel_codes)

    def test_get_correct_boxes_after_decoding(self):
        anchors = [[15.0, 12.0, 30.0, 18.0, 0.15], [0.1, 0.0, 0.7, 0.9, 0.1]]
        rel_codes = [[-0.154833, -0.151374, -0.405465, -0.182322, -0.05],
                     [0.127882, 0.121649, -0.336472, -0.810930, 0.1]]
        expected_boxes = [[10.0, 10.0, 20.0, 15.0,  0.07146],
                          [0.2, 0.1, 0.5, 0.4, 0.25708]]

        anchors = rbox_list.RBoxList(tf.constant(anchors))
        coder = faster_rcnn_rbox_coder.FasterRcnnRBoxCoder()
        boxes = coder.decode(rel_codes, anchors)
        with self.test_session() as sess:
            boxes_out, = sess.run([boxes.get()])
            self.assertAllClose(boxes_out, expected_boxes)

    def test_get_correct_boxes_after_decoding_with_scaling(self):
        anchors = [[15.0, 12.0, 30.0, 18.0, 0.15], [0.1, 0.0, 0.7, 0.9, 0.1]]
        rel_codes = [[-0.309665, -0.454122, -1.621860, -0.911607, -0.3],
                     [0.2557630, 0.364945, -1.345888, -4.054651, 0.6]]
        scale_factors = [2, 3, 4, 5, 6]
        expected_boxes = [[10.0, 10.0, 20.0, 15.0, 0.07146],
                          [0.2, 0.1, 0.5, 0.4, 0.25708]]

        anchors = rbox_list.RBoxList(tf.constant(anchors))
        coder = faster_rcnn_rbox_coder.FasterRcnnRBoxCoder(scale_factors=scale_factors)
        boxes = coder.decode(rel_codes, anchors)
        with self.test_session() as sess:
            boxes_out, = sess.run([boxes.get()])
            self.assertAllClose(boxes_out, expected_boxes)


if __name__ == '__main__':
    tf.test.main()
