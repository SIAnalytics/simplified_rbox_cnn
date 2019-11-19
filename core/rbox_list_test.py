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

"""Tests for object_detection.core.rbox_list."""

import tensorflow as tf

from core import rbox_list


class RBoxListTest(tf.test.TestCase):
    """Tests for RBoxList class."""

    def test_num_boxes(self):
        data = tf.constant([[0, 0, 1, 1, 0.1], [1, 1, 2, 3, 0.2], [3, 4, 5, 5, 0.3]], tf.float32)
        expected_num_boxes = 3

        rboxes = rbox_list.RBoxList(data)
        with self.test_session() as sess:
            num_boxes_output = sess.run(rboxes.num_boxes())
            self.assertEquals(num_boxes_output, expected_num_boxes)

    def test_create_box_list_with_dynamic_shape(self):
        data = tf.constant([[0, 0, 1, 1, 0.1], [1, 1, 2, 3, 0.2], [3, 4, 5, 5, 0.3]], tf.float32)
        indices = tf.reshape(tf.where(tf.greater([1, 0, 1], 0)), [-1])
        data = tf.gather(data, indices)
        assert data.get_shape().as_list() == [None, 5]
        expected_num_boxes = 2

        rboxes = rbox_list.RBoxList(data)
        with self.test_session() as sess:
            num_boxes_output = sess.run(rboxes.num_boxes())
            self.assertEquals(num_boxes_output, expected_num_boxes)

    def test_box_list_invalid_inputs(self):
        data0 = tf.constant([[[0, 0, 1, 1], [3, 4, 5, 5]]], tf.float32)
        data1 = tf.constant([[0, 0, 1], [1, 1, 2], [3, 4, 5]], tf.float32)
        data2 = tf.constant([[0, 0, 1], [1, 1, 2], [3, 4, 5]], tf.int32)

        with self.assertRaises(ValueError):
            _ = rbox_list.RBoxList(data0)
        with self.assertRaises(ValueError):
            _ = rbox_list.RBoxList(data1)
        with self.assertRaises(ValueError):
            _ = rbox_list.RBoxList(data2)

    def test_num_boxes_static(self):
        box_corners = [[10.0, 10.0, 20.0, 15.0, 0.1], [0.2, 0.1, 0.5, 0.4, 0.2]]
        rboxes = rbox_list.RBoxList(tf.constant(box_corners))
        self.assertEquals(rboxes.num_boxes_static(), 2)
        self.assertEquals(type(rboxes.num_boxes_static()), int)

    def test_num_boxes_static_for_uninferrable_shape(self):
        placeholder = tf.placeholder(tf.float32, shape=[None, 5])
        rboxes = rbox_list.RBoxList(placeholder)
        self.assertEquals(rboxes.num_boxes_static(), None)

    def test_as_tensor_dict(self):
        rboxlist = rbox_list.RBoxList(tf.constant([[0.1, 0.1, 0.4, 0.4, 0.1], [0.1, 0.1, 0.5, 0.5, 0.2]], tf.float32))
        rboxlist.add_field('classes', tf.constant([0, 1]))
        rboxlist.add_field('scores', tf.constant([0.75, 0.2]))
        tensor_dict = rboxlist.as_tensor_dict()

        expected_boxes = [[0.1, 0.1, 0.4, 0.4, 0.1], [0.1, 0.1, 0.5, 0.5, 0.2]]
        expected_classes = [0, 1]
        expected_scores = [0.75, 0.2]

        with self.test_session() as sess:
            tensor_dict_out = sess.run(tensor_dict)
            self.assertAllEqual(3, len(tensor_dict_out))
            self.assertAllClose(expected_boxes, tensor_dict_out['boxes'])
            self.assertAllEqual(expected_classes, tensor_dict_out['classes'])
            self.assertAllClose(expected_scores, tensor_dict_out['scores'])

    def test_as_tensor_dict_with_features(self):
        rboxlist = rbox_list.RBoxList(tf.constant([[0.1, 0.1, 0.4, 0.4, 0.1], [0.1, 0.1, 0.5, 0.5, 0.2]], tf.float32))
        rboxlist.add_field('classes', tf.constant([0, 1]))
        rboxlist.add_field('scores', tf.constant([0.75, 0.2]))
        tensor_dict = rboxlist.as_tensor_dict(['boxes', 'classes', 'scores'])

        expected_boxes = [[0.1, 0.1, 0.4, 0.4, 0.1], [0.1, 0.1, 0.5, 0.5, 0.2]]
        expected_classes = [0, 1]
        expected_scores = [0.75, 0.2]

        with self.test_session() as sess:
            tensor_dict_out = sess.run(tensor_dict)
            self.assertAllEqual(3, len(tensor_dict_out))
            self.assertAllClose(expected_boxes, tensor_dict_out['boxes'])
            self.assertAllEqual(expected_classes, tensor_dict_out['classes'])
            self.assertAllClose(expected_scores, tensor_dict_out['scores'])

    def test_as_tensor_dict_missing_field(self):
        rboxlist = rbox_list.RBoxList(tf.constant([[0.1, 0.1, 0.4, 0.4, 0.1], [0.1, 0.1, 0.5, 0.5, 0.2]], tf.float32))
        rboxlist.add_field('classes', tf.constant([0, 1]))
        rboxlist.add_field('scores', tf.constant([0.75, 0.2]))
        with self.assertRaises(ValueError):
            rboxlist.as_tensor_dict(['foo', 'bar'])

    def test_get_corners(self):
        data = tf.constant([[0, 0, 1, 1, 0.0], [1, 1, 2, 3, 0.2], [3, 4, 5, 5, 0.3]], tf.float32)
        expected_corners = [[-0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5],
                            [-0.278071, -0.271431, 0.317937, 2.66877, 2.27807, 2.27143, 1.68206, -0.668769],
                            [-0.127142, 2.35046, 1.35046, 7.12714, 6.12714, 5.64954, 4.64954, 0.872858]]
        rboxes = rbox_list.RBoxList(data)
        with self.test_session() as sess:
            corners_output = sess.run(rboxes.get_corners())
            self.assertAllClose(expected_corners, corners_output)


if __name__ == '__main__':
    tf.test.main()
