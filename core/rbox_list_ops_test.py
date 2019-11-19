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

"""Tests for object_detection.core.rbox_list_ops."""
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import errors

from core import rbox_list
from core import rbox_list_ops


class RBoxListOpsTest(tf.test.TestCase):
    """Tests for common rotated bounding box operations."""

    def test_area(self):
        rbox = tf.constant([[0.0, 0.0, 10.0, 20.0, 0.1], [1.0, 2.0, 3.0, 4.0, 0.2]])
        exp_output = [200.0, 12.0]
        boxes = rbox_list.RBoxList(rbox)
        areas = rbox_list_ops.area(boxes)
        with self.test_session() as sess:
            areas_output = sess.run(areas)
            self.assertAllClose(areas_output, exp_output)

    def test_height_width(self):
        rbox = tf.constant([[0.0, 0.0, 10.0, 20.0, 0.1], [1.0, 2.0, 3.0, 4.0, 0.2]])
        exp_output_heights = [10., 3.]
        exp_output_widths = [20., 4.]
        boxes = rbox_list.RBoxList(rbox)
        heights, widths = rbox_list_ops.height_width(boxes)
        with self.test_session() as sess:
            output_heights, output_widths = sess.run([heights, widths])
            self.assertAllClose(output_heights, exp_output_heights)
            self.assertAllClose(output_widths, exp_output_widths)

    def test_scale(self):
        rbox = tf.constant([[0, 0, 100, 200, 0.1], [50, 120, 100, 140, 0.2]], dtype=tf.float32)
        boxes = rbox_list.RBoxList(rbox)
        boxes.add_field('extra_data', tf.constant([[1], [2]]))

        y_scale = tf.constant(1.0 / 100)
        x_scale = tf.constant(1.0 / 200)
        scaled_boxes = rbox_list_ops.scale(boxes, y_scale, x_scale)
        exp_output = [[0, 0, 1, 1, 0.1], [0.5, 0.6, 1.0, 0.7, 0.2]]
        with self.test_session() as sess:
            scaled_corners_out = sess.run(scaled_boxes.get())
            self.assertAllClose(scaled_corners_out, exp_output)
            extra_data_out = sess.run(scaled_boxes.get_field('extra_data'))
            self.assertAllEqual(extra_data_out, [[1], [2]])

    def test_clip_to_window_filter_boxes_which_fall_outside_the_window(self):
        pass

    def test_clip_to_window_without_filtering_boxes_which_fall_outside_the_window(self):
        pass

    def test_prune_outside_window_filters_boxes_which_fall_outside_the_window(self):
        window = tf.constant([0, 0, 10, 15], tf.float32)
        rboxes = tf.constant([[5.0, 5.0, 4.0, 4.0, 0.0],
                              [-1.0, -2.0, 4.0, 5.0, 0.0],
                              [6.0, 6.0, 2.0, 9.0, 0.0],
                              [5.0, 7.5, 10.0, 15.0, 0.0],
                              [-10.0, -10.0, -9.0, -9.0, 0.0],
                              [0.0, 0.0, 300.0, 600.0, 0.0]])
        boxes = rbox_list.RBoxList(rboxes)
        boxes.add_field('extra_data', tf.constant([[1], [2], [3], [4], [5], [6]]))
        exp_output = [[5.0, 5.0, 4.0, 4.0, 0.0],
                      [6.0, 6.0, 2.0, 9.0, 0.0],
                      [5.0, 7.5, 10.0, 15.0, 0.0]]
        pruned, keep_indices = rbox_list_ops.prune_outside_window(boxes, window)
        with self.test_session() as sess:
            pruned_output = sess.run(pruned.get())
            self.assertAllClose(pruned_output, exp_output)
            keep_indices_out = sess.run(keep_indices)
            self.assertAllEqual(keep_indices_out, [0, 2, 3])
            extra_data_out = sess.run(pruned.get_field('extra_data'))
            self.assertAllEqual(extra_data_out, [[1], [3], [4]])

    def test_prune_completely_outside_window(self):
        pass

    def test_intersection_shapely(self):
        rbox1 = tf.constant([[4.0, 3.0, 7.0, 5.0, 0.1],
                             [5.0, 6.0, 10.0, 7.0, 0.2]])
        rbox2 = tf.constant([[3.0, 4.0, 6.0, 8.0, 0.1],
                             [14.0, 14.0, 15.0, 15.0, 0.2],
                             [0.0, 0.0, 20.0, 20.0, 0.3]])
        exp_output = [[27.025812, 0., 35.],
                      [30.889588, 7.2258, 63.043903]]
        boxes1 = rbox_list.RBoxList(rbox1)
        boxes2 = rbox_list.RBoxList(rbox2)
        intersect = rbox_list_ops.intersection(boxes1, boxes2, ver='shapely')
        with self.test_session() as sess:
            intersect_output = sess.run(intersect)
            self.assertAllClose(intersect_output, exp_output)

    def test_intersection_opencv(self):
        rbox1 = tf.constant([[4.0, 3.0, 7.0, 5.0, 0.1],
                             [5.0, 6.0, 10.0, 7.0, 0.2]])
        rbox2 = tf.constant([[3.0, 4.0, 6.0, 8.0, 0.1],
                             [14.0, 14.0, 15.0, 15.0, 0.2],
                             [0.0, 0.0, 20.0, 20.0, 0.3]])
        exp_output = [[27.025812, 0., 35.],
                      [30.889588, 7.2258, 63.043903]]
        boxes1 = rbox_list.RBoxList(rbox1)
        boxes2 = rbox_list.RBoxList(rbox2)

        intersect = rbox_list_ops.intersection(boxes1, boxes2, ver='opencv')
        with self.test_session() as sess:
            intersect_output = sess.run(intersect)
            self.assertAllClose(intersect_output, exp_output)

    def test_intersection_cython(self):
        rbox1 = tf.constant([[4.0, 3.0, 7.0, 5.0, 0.1],
                             [5.0, 6.0, 10.0, 7.0, 0.2]])
        rbox2 = tf.constant([[3.0, 4.0, 6.0, 8.0, 0.1],
                             [14.0, 14.0, 15.0, 15.0, 0.2],
                             [0.0, 0.0, 20.0, 20.0, 0.3]])
        exp_output = [[27.025812, 0., 35.],
                      [30.889588, 7.2258, 63.043903]]
        boxes1 = rbox_list.RBoxList(rbox1)
        boxes2 = rbox_list.RBoxList(rbox2)

        intersect = rbox_list_ops.intersection(boxes1, boxes2, ver='cython')
        with self.test_session() as sess:
            intersect_output = sess.run(intersect)
            self.assertAllClose(intersect_output, exp_output)

    def test_intersection_tf(self):
        rbox1 = tf.constant([[4.0, 3.0, 7.0, 5.0, 0.1],
                             [5.0, 6.0, 10.0, 7.0, 0.2]])
        rbox2 = tf.constant([[3.0, 4.0, 6.0, 8.0, 0.1],
                             [14.0, 14.0, 15.0, 15.0, 0.2],
                             [0.0, 0.0, 20.0, 20.0, 0.3],
                             [0.0, 0.0, 20.0, 20.0, -1.57],
                             [5.0, 6.0, 10.0, 7.0, 0.2]])
        exp_output = [[27.025812, 0., 35., -1., 20.040615],
                      [30.889588, 7.2258, 63.043903, -1., 70.0]]
        boxes1 = rbox_list.RBoxList(rbox1)
        boxes2 = rbox_list.RBoxList(rbox2)

        intersect = rbox_list_ops.intersection(boxes1, boxes2, ver='tf')
        with self.test_session() as sess:
            intersect_output = sess.run(intersect)
            self.assertAllClose(intersect_output, exp_output)

    def test_matched_intersection(self):
        rbox1 = tf.constant([[4.0, 3.0, 7.0, 5.0, 0.1], [5.0, 6.0, 10.0, 7.0, 0.2]])
        rbox2 = tf.constant([[3.0, 4.0, 6.0, 8.0, 0.1], [14.0, 14.0, 15.0, 15.0, 0.2]])
        exp_output = [27.025812, 7.2258]
        boxes1 = rbox_list.RBoxList(rbox1)
        boxes2 = rbox_list.RBoxList(rbox2)
        intersect = rbox_list_ops.matched_intersection(boxes1, boxes2)
        with self.test_session() as sess:
            intersect_output = sess.run(intersect)
            self.assertAllClose(intersect_output, exp_output)

    def test_iou(self):
        rbox1 = tf.constant([[4.0, 3.0, 7.0, 5.0, 0.1],
                             [5.0, 6.0, 10.0, 7.0, 0.2]])
        rbox2 = tf.constant([[3.0, 4.0, 6.0, 8.0, 0.1],
                             [14.0, 14.0, 15.0, 15.0, 0.2],
                             [0.0, 0.0, 20.0, 20.0, 0.3],
                             [0.0, 0.0, 20.0, 20.0, -1.57]])
        exp_output = [[27.025812 / 55.974188, 0. / 260, 35. / 400, -1],
                      [30.889588 / 87.110412, 7.2258 / 287.7742, 63.043903 / 406.956097, -1]]
        boxes1 = rbox_list.RBoxList(rbox1)
        boxes2 = rbox_list.RBoxList(rbox2)
        iou = rbox_list_ops.iou(boxes1, boxes2)
        with self.test_session() as sess:
            iou_output = sess.run(iou)
            self.assertAllClose(iou_output, exp_output)

    def test_matched_iou(self):
        rbox1 = tf.constant([[4.0, 3.0, 7.0, 5.0, 0.1], [5.0, 6.0, 10.0, 7.0, 0.2]])
        rbox2 = tf.constant([[3.0, 4.0, 6.0, 8.0, 0.1], [14.0, 14.0, 15.0, 15.0, 0.2]])
        exp_output = [27.025812 / 55.974188, 7.2258 / 287.7742]
        boxes1 = rbox_list.RBoxList(rbox1)
        boxes2 = rbox_list.RBoxList(rbox2)
        iou = rbox_list_ops.matched_iou(boxes1, boxes2)
        with self.test_session() as sess:
            iou_output = sess.run(iou)
            self.assertAllClose(iou_output, exp_output)

    def test_iouworks_on_empty_inputs(self):
        rbox1 = tf.constant([[4.0, 3.0, 7.0, 5.0, 0.1],
                             [5.0, 6.0, 10.0, 7.0, 0.2]])
        rbox2 = tf.constant([[3.0, 4.0, 6.0, 8.0, 0.1],
                             [14.0, 14.0, 15.0, 15.0, 0.2],
                             [0.0, 0.0, 20.0, 20.0, 0.3]])

        boxes1 = rbox_list.RBoxList(rbox1)
        boxes2 = rbox_list.RBoxList(rbox2)
        boxes_empty = rbox_list.RBoxList(tf.zeros((0, 5)))
        iou_empty_1 = rbox_list_ops.iou(boxes1, boxes_empty)
        iou_empty_2 = rbox_list_ops.iou(boxes_empty, boxes2)
        iou_empty_3 = rbox_list_ops.iou(boxes_empty, boxes_empty)
        with self.test_session() as sess:
            iou_output_1, iou_output_2, iou_output_3 = sess.run([iou_empty_1, iou_empty_2, iou_empty_3])
            self.assertAllEqual(iou_output_1.shape, (2, 0))
            self.assertAllEqual(iou_output_2.shape, (0, 3))
            self.assertAllEqual(iou_output_3.shape, (0, 0))

    def test_ioa(self):
        rbox1 = tf.constant([[4.0, 3.0, 7.0, 5.0, 0.1],
                             [5.0, 6.0, 10.0, 7.0, 0.2]])
        rbox2 = tf.constant([[3.0, 4.0, 6.0, 8.0, 0.1],
                             [14.0, 14.0, 15.0, 15.0, 0.2],
                             [0.0, 0.0, 20.0, 20.0, 0.3]])
        exp_output_1 = [[27.025812 / 48, 0. / 225, 35. / 400],
                        [30.889588 / 48, 7.2258 / 225, 63.043903 / 400]]
        exp_output_2 = [[27.025812 / 35, 30.889588 / 70],
                        [0. / 35, 7.2258 / 70],
                        [35. / 35, 63.043903 / 70]]

        boxes1 = rbox_list.RBoxList(rbox1)
        boxes2 = rbox_list.RBoxList(rbox2)
        ioa_1 = rbox_list_ops.ioa(boxes1, boxes2)
        ioa_2 = rbox_list_ops.ioa(boxes2, boxes1)
        with self.test_session() as sess:
            ioa_output_1, ioa_output_2 = sess.run([ioa_1, ioa_2])
            self.assertAllClose(ioa_output_1, exp_output_1)
            self.assertAllClose(ioa_output_2, exp_output_2)

    def test_prune_non_overlapping_boxes(self):
        rbox1 = tf.constant([[4.0, 3.0, 7.0, 5.0, 0.1],
                             [5.0, 6.0, 10.0, 7.0, 0.2]])
        rbox2 = tf.constant([[3.0, 4.0, 6.0, 8.0, 0.1],
                             [14.0, 14.0, 15.0, 15.0, 0.2],
                             [0.0, 0.0, 20.0, 20.0, 0.3]])
        boxes1 = rbox_list.RBoxList(rbox1)
        boxes2 = rbox_list.RBoxList(rbox2)
        minoverlap = 0.5

        exp_output_1 = boxes1
        exp_output_2 = rbox_list.RBoxList(rbox2[:1])
        output_1, keep_indices_1 = rbox_list_ops.prune_non_overlapping_boxes(
            boxes1, boxes2, min_overlap=minoverlap)
        output_2, keep_indices_2 = rbox_list_ops.prune_non_overlapping_boxes(
            boxes2, boxes1, min_overlap=minoverlap)

        with self.test_session() as sess:
            (output_1_, keep_indices_1_, output_2_, keep_indices_2_, exp_output_1_,
             exp_output_2_) = sess.run(
                [output_1.get(), keep_indices_1,
                 output_2.get(), keep_indices_2,
                 exp_output_1.get(), exp_output_2.get()])
            self.assertAllClose(output_1_, exp_output_1_)
            self.assertAllClose(output_2_, exp_output_2_)
            self.assertAllEqual(keep_indices_1_, [0, 1])
            self.assertAllEqual(keep_indices_2_, [0])

    def test_prune_small_boxes(self):
        boxes = tf.constant([[4.0, 3.0, 7.0, 5.0, 1],
                             [5.0, 6.0, 10.0, 7.0, 2],
                             [3.0, 4.0, 6.0, 8.0, 3],
                             [14.0, 14.0, 15.0, 15.0, 4],
                             [0.0, 0.0, 20.0, 20.0, 5]])
        exp_boxes = [[5.0, 6.0, 10.0, 7.0, 2],
                     [14.0, 14.0, 15.0, 15.0, 4],
                     [0.0, 0.0, 20.0, 20.0, 5]]

        boxes = rbox_list.RBoxList(boxes)
        pruned_boxes = rbox_list_ops.prune_small_boxes(boxes, 7)
        with self.test_session() as sess:
            pruned_boxes = sess.run(pruned_boxes.get())
            self.assertAllEqual(pruned_boxes, exp_boxes)

    def test_prune_small_boxes_prunes_boxes_with_negative_side(self):
        boxes = tf.constant([[4.0, 3.0, 7.0, 5.0, 1],
                             [5.0, 6.0, 10.0, 7.0, 2],
                             [3.0, 4.0, 6.0, 8.0, 3],
                             [14.0, 14.0, 15.0, 15.0, 4],
                             [0.0, 0.0, 20.0, 20.0, 5],
                             [0.0, 0.0, -20.0, 20.0, 6],  # negative height
                             [0.0, 0.0, 20.0, -20.0, 7]])  # negative width
        exp_boxes = [[5.0, 6.0, 10.0, 7.0, 2],
                     [14.0, 14.0, 15.0, 15.0, 4],
                     [0.0, 0.0, 20.0, 20.0, 5]]
        boxes = rbox_list.RBoxList(boxes)
        pruned_boxes = rbox_list_ops.prune_small_boxes(boxes, 7)
        with self.test_session() as sess:
            pruned_boxes = sess.run(pruned_boxes.get())
            self.assertAllEqual(pruned_boxes, exp_boxes)

    def test_change_coordinate_frame(self):
        corners = tf.constant([[0.25, 0.5, 0.75, 0.75, 0.1], [0.5, 0.0, 1.0, 1.0, 0.2]])
        window = tf.constant([0.25, 0.25, 0.75, 0.75])
        boxes = rbox_list.RBoxList(corners)

        expected_corners = tf.constant([[0, 0.5, 1.5, 1.5, 0.1], [0.5, -0.5, 2.0, 2.0, 0.2]])
        expected_boxes = rbox_list.RBoxList(expected_corners)
        output = rbox_list_ops.change_coordinate_frame(boxes, window)

        with self.test_session() as sess:
            output_, expected_boxes_ = sess.run([output.get(), expected_boxes.get()])
            self.assertAllClose(output_, expected_boxes_)

    def test_ioaworks_on_empty_inputs(self):
        rbox1 = tf.constant([[4.0, 3.0, 7.0, 5.0, 0.1], [5.0, 6.0, 10.0, 7.0, 0.2]])
        rbox2 = tf.constant([[3.0, 4.0, 6.0, 8.0, 0.1], [14.0, 14.0, 15.0, 15.0, 0.2], [0.0, 0.0, 20.0, 20.0, 0.3]])
        boxes1 = rbox_list.RBoxList(rbox1)
        boxes2 = rbox_list.RBoxList(rbox2)
        boxes_empty = rbox_list.RBoxList(tf.zeros((0, 5)))
        ioa_empty_1 = rbox_list_ops.ioa(boxes1, boxes_empty)
        ioa_empty_2 = rbox_list_ops.ioa(boxes_empty, boxes2)
        ioa_empty_3 = rbox_list_ops.ioa(boxes_empty, boxes_empty)
        with self.test_session() as sess:
            ioa_output_1, ioa_output_2, ioa_output_3 = sess.run([ioa_empty_1, ioa_empty_2, ioa_empty_3])
            self.assertAllEqual(ioa_output_1.shape, (2, 0))
            self.assertAllEqual(ioa_output_2.shape, (0, 3))
            self.assertAllEqual(ioa_output_3.shape, (0, 0))

    def test_pairwise_distances(self):
        pass

    def test_boolean_mask(self):
        corners = tf.constant([5 * [0.0], 5 * [1.0], 5 * [2.0], 5 * [3.0], 5 * [4.0]])
        indicator = tf.constant([True, False, True, False, True], tf.bool)
        expected_subset = [5 * [0.0], 5 * [2.0], 5 * [4.0]]
        boxes = rbox_list.RBoxList(corners)
        subset = rbox_list_ops.boolean_mask(boxes, indicator)
        with self.test_session() as sess:
            subset_output = sess.run(subset.get())
            self.assertAllClose(subset_output, expected_subset)

    def test_boolean_mask_with_field(self):
        corners = tf.constant([5 * [0.0], 5 * [1.0], 5 * [2.0], 5 * [3.0], 5 * [4.0]])
        indicator = tf.constant([True, False, True, False, True], tf.bool)
        weights = tf.constant([[.1], [.3], [.5], [.7], [.9]], tf.float32)
        expected_subset = [5 * [0.0], 5 * [2.0], 5 * [4.0]]
        expected_weights = [[.1], [.5], [.9]]

        boxes = rbox_list.RBoxList(corners)
        boxes.add_field('weights', weights)
        subset = rbox_list_ops.boolean_mask(boxes, indicator, ['weights'])
        with self.test_session() as sess:
            subset_output, weights_output = sess.run(
                [subset.get(), subset.get_field('weights')])
            self.assertAllClose(subset_output, expected_subset)
            self.assertAllClose(weights_output, expected_weights)

    def test_gather(self):
        corners = tf.constant([5 * [0.0], 5 * [1.0], 5 * [2.0], 5 * [3.0], 5 * [4.0]])
        indices = tf.constant([0, 2, 4], tf.int32)
        expected_subset = [5 * [0.0], 5 * [2.0], 5 * [4.0]]
        boxes = rbox_list.RBoxList(corners)
        subset = rbox_list_ops.gather(boxes, indices)
        with self.test_session() as sess:
            subset_output = sess.run(subset.get())
            self.assertAllClose(subset_output, expected_subset)

    def test_gather_with_field(self):
        corners = tf.constant([5 * [0.0], 5 * [1.0], 5 * [2.0], 5 * [3.0], 5 * [4.0]])
        indices = tf.constant([0, 2, 4], tf.int32)
        weights = tf.constant([[.1], [.3], [.5], [.7], [.9]], tf.float32)
        expected_subset = [5 * [0.0], 5 * [2.0], 5 * [4.0]]
        expected_weights = [[.1], [.5], [.9]]

        boxes = rbox_list.RBoxList(corners)
        boxes.add_field('weights', weights)
        subset = rbox_list_ops.gather(boxes, indices, ['weights'])
        with self.test_session() as sess:
            subset_output, weights_output = sess.run([subset.get(), subset.get_field('weights')])
            self.assertAllClose(subset_output, expected_subset)
            self.assertAllClose(weights_output, expected_weights)

    def test_gather_with_invalid_field(self):
        corners = tf.constant([5 * [0.0], 5 * [1.0]])
        indices = tf.constant([0, 1], tf.int32)
        weights = tf.constant([[.1], [.3]], tf.float32)

        boxes = rbox_list.RBoxList(corners)
        boxes.add_field('weights', weights)
        with self.assertRaises(ValueError):
            rbox_list_ops.gather(boxes, indices, ['foo', 'bar'])

    def test_gather_with_invalid_inputs(self):
        corners = tf.constant([5 * [0.0], 5 * [1.0], 5 * [2.0], 5 * [3.0], 5 * [4.0]])
        indices_float32 = tf.constant([0, 2, 4], tf.float32)
        boxes = rbox_list.RBoxList(corners)
        with self.assertRaises(ValueError):
            _ = rbox_list_ops.gather(boxes, indices_float32)
        indices_2d = tf.constant([[0, 2, 4]], tf.int32)
        boxes = rbox_list.RBoxList(corners)
        with self.assertRaises(ValueError):
            _ = rbox_list_ops.gather(boxes, indices_2d)

    def test_gather_with_dynamic_indexing(self):
        corners = tf.constant([5 * [0.0], 5 * [1.0], 5 * [2.0], 5 * [3.0], 5 * [4.0]])
        weights = tf.constant([.5, .3, .7, .1, .9], tf.float32)
        indices = tf.reshape(tf.where(tf.greater(weights, 0.4)), [-1])
        expected_subset = [5 * [0.0], 5 * [2.0], 5 * [4.0]]
        expected_weights = [.5, .7, .9]

        boxes = rbox_list.RBoxList(corners)
        boxes.add_field('weights', weights)
        subset = rbox_list_ops.gather(boxes, indices, ['weights'])
        with self.test_session() as sess:
            subset_output, weights_output = sess.run([subset.get(), subset.get_field('weights')])
            self.assertAllClose(subset_output, expected_subset)
            self.assertAllClose(weights_output, expected_weights)

    def test_sort_by_field_ascending_order(self):
        exp_corners = [[0, 0, 1, 1, 0.1], [0, 0.1, 1, 1.1, 0.2], [0, -0.1, 1, 0.9, 0.3],
                       [0, 10, 1, 11, 0.4], [0, 10.1, 1, 11.1, 0.5], [0, 100, 1, 101, 0.6]]
        exp_scores = [.95, .9, .75, .6, .5, .3]
        exp_weights = [.2, .45, .6, .75, .8, .92]
        shuffle = [2, 4, 0, 5, 1, 3]

        rbox = tf.constant([exp_corners[i] for i in shuffle], tf.float32)
        boxes = rbox_list.RBoxList(rbox)
        boxes.add_field('scores', tf.constant([exp_scores[i] for i in shuffle], tf.float32))
        boxes.add_field('weights', tf.constant([exp_weights[i] for i in shuffle], tf.float32))
        sort_by_weight = rbox_list_ops.sort_by_field(boxes, 'weights', order=rbox_list_ops.SortOrder.ascend)
        with self.test_session() as sess:
            corners_out, scores_out, weights_out = sess.run([
                sort_by_weight.get(),
                sort_by_weight.get_field('scores'),
                sort_by_weight.get_field('weights')])
            self.assertAllClose(corners_out, exp_corners)
            self.assertAllClose(scores_out, exp_scores)
            self.assertAllClose(weights_out, exp_weights)

    def test_sort_by_field_descending_order(self):
        exp_corners = [[0, 0, 1, 1, 0.1], [0, 0.1, 1, 1.1, 0.2], [0, -0.1, 1, 0.9, 0.3],
                       [0, 10, 1, 11, 0.4], [0, 10.1, 1, 11.1, 0.5], [0, 100, 1, 101, 0.6]]
        exp_scores = [.95, .9, .75, .6, .5, .3]
        exp_weights = [.2, .45, .6, .75, .8, .92]
        shuffle = [2, 4, 0, 5, 1, 3]

        corners = tf.constant([exp_corners[i] for i in shuffle], tf.float32)
        rboxes = rbox_list.RBoxList(corners)
        rboxes.add_field('scores', tf.constant([exp_scores[i] for i in shuffle], tf.float32))
        rboxes.add_field('weights', tf.constant([exp_weights[i] for i in shuffle], tf.float32))

        sort_by_score = rbox_list_ops.sort_by_field(rboxes, 'scores')
        with self.test_session() as sess:
            rboxes_out, corners_out, scores_out, weights_out = sess.run([rboxes.get_field('scores'), sort_by_score.get(
            ), sort_by_score.get_field('scores'), sort_by_score.get_field('weights')])
            self.assertAllClose(corners_out, exp_corners)
            self.assertAllClose(scores_out, exp_scores)
            self.assertAllClose(weights_out, exp_weights)

    def test_sort_by_field_invalid_inputs(self):
        rbox = tf.constant([5 * [0.0], 5 * [0.5], 5 * [1.0], 5 * [2.0], 5 * [3.0], 5 * [4.0]])
        misc = tf.constant([[.95, .9], [.5, .3]], tf.float32)
        weights = tf.constant([.1, .2], tf.float32)
        boxes = rbox_list.RBoxList(rbox)
        boxes.add_field('misc', misc)
        boxes.add_field('weights', weights)

        with self.test_session() as sess:
            with self.assertRaises(ValueError):
                rbox_list_ops.sort_by_field(boxes, 'area')

            with self.assertRaises(ValueError):
                rbox_list_ops.sort_by_field(boxes, 'misc')

            with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError, 'Incorrect field size'):
                sess.run(rbox_list_ops.sort_by_field(boxes, 'weights').get())

    def test_visualize_boxes_in_image(self):
        pass

    def test_filter_field_value_equals(self):
        rbox = tf.constant([[0, 0, 1, 1, 0.1],
                            [0, 0.1, 1, 1.1, 0.2],
                            [0, -0.1, 1, 0.9, 0.3],
                            [0, 10, 1, 11, 0.4],
                            [0, 10.1, 1, 11.1, 0.5],
                            [0, 100, 1, 101, 0.6]], tf.float32)
        boxes = rbox_list.RBoxList(rbox)
        boxes.add_field('classes', tf.constant([1, 2, 1, 2, 2, 1]))
        exp_output1 = [[0, 0, 1, 1, 0.1], [0, -0.1, 1, 0.9, 0.3], [0, 100, 1, 101, 0.6]]
        exp_output2 = [[0, 0.1, 1, 1.1, 0.2], [0, 10, 1, 11, 0.4], [0, 10.1, 1, 11.1, 0.5]]

        filtered_boxes1 = rbox_list_ops.filter_field_value_equals(boxes, 'classes', 1)
        filtered_boxes2 = rbox_list_ops.filter_field_value_equals(boxes, 'classes', 2)
        with self.test_session() as sess:
            filtered_output1, filtered_output2 = sess.run([filtered_boxes1.get(),
                                                           filtered_boxes2.get()])
            self.assertAllClose(filtered_output1, exp_output1)
            self.assertAllClose(filtered_output2, exp_output2)

    def test_filter_greater_than(self):
        rbox = tf.constant([[0, 0, 1, 1, 0.1],
                            [0, 0.1, 1, 1.1, 0.2],
                            [0, -0.1, 1, 0.9, 0.3],
                            [0, 10, 1, 11, 0.4],
                            [0, 10.1, 1, 11.1, 0.5],
                            [0, 100, 1, 101, 0.6]], tf.float32)
        boxes = rbox_list.RBoxList(rbox)
        boxes.add_field('scores', tf.constant([.1, .75, .9, .5, .5, .8]))
        thresh = .6
        exp_output = [[0, 0.1, 1, 1.1, 0.2], [0, -0.1, 1, 0.9, 0.3], [0, 100, 1, 101, 0.6]]

        filtered_boxes = rbox_list_ops.filter_greater_than(boxes, thresh)
        with self.test_session() as sess:
            filtered_output = sess.run(filtered_boxes.get())
            self.assertAllClose(filtered_output, exp_output)

    def test_clip_box_list(self):
        boxlist = rbox_list.RBoxList(tf.constant([[0.1, 0.1, 0.4, 0.4, 0.1], [0.1, 0.1, 0.5, 0.5, 0.2],
                                                  [0.6, 0.6, 0.8, 0.8, 0.3], [0.2, 0.2, 0.3, 0.3, 0.4]], tf.float32))
        boxlist.add_field('classes', tf.constant([0, 0, 1, 1]))
        boxlist.add_field('scores', tf.constant([0.75, 0.65, 0.3, 0.2]))
        num_boxes = 2
        clipped_boxlist = rbox_list_ops.pad_or_clip_box_list(boxlist, num_boxes)

        expected_boxes = [[0.1, 0.1, 0.4, 0.4, 0.1], [0.1, 0.1, 0.5, 0.5, 0.2]]
        expected_classes = [0, 0]
        expected_scores = [0.75, 0.65]
        with self.test_session() as sess:
            boxes_out, classes_out, scores_out = sess.run(
                [clipped_boxlist.get(), clipped_boxlist.get_field('classes'),
                 clipped_boxlist.get_field('scores')])

            self.assertAllClose(expected_boxes, boxes_out)
            self.assertAllEqual(expected_classes, classes_out)
            self.assertAllClose(expected_scores, scores_out)

    def test_pad_box_list(self):
        boxlist = rbox_list.RBoxList(tf.constant([[0.1, 0.1, 0.4, 0.4, 0.1], [0.1, 0.1, 0.5, 0.5, 0.2]], tf.float32))
        boxlist.add_field('classes', tf.constant([0, 1]))
        boxlist.add_field('scores', tf.constant([0.75, 0.2]))
        num_boxes = 4
        padded_boxlist = rbox_list_ops.pad_or_clip_box_list(boxlist, num_boxes)

        expected_boxes = [[0.1, 0.1, 0.4, 0.4, 0.1], [0.1, 0.1, 0.5, 0.5, 0.2],
                          [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        expected_classes = [0, 1, 0, 0]
        expected_scores = [0.75, 0.2, 0, 0]
        with self.test_session() as sess:
            boxes_out, classes_out, scores_out = sess.run(
                [padded_boxlist.get(), padded_boxlist.get_field('classes'),
                 padded_boxlist.get_field('scores')])

            self.assertAllClose(expected_boxes, boxes_out)
            self.assertAllEqual(expected_classes, classes_out)
            self.assertAllClose(expected_scores, scores_out)


class ConcatenateTest(tf.test.TestCase):
    def test_invalid_input_box_list_list(self):
        with self.assertRaises(ValueError):
            rbox_list_ops.concatenate(None)
        with self.assertRaises(ValueError):
            rbox_list_ops.concatenate([])
        with self.assertRaises(ValueError):
            corners = tf.constant([[0, 0, 0, 0, 0]], tf.float32)
            boxlist = rbox_list.RBoxList(corners)
            rbox_list_ops.concatenate([boxlist, 2])

    def test_concatenate_with_missing_fields(self):
        corners1 = tf.constant([[0, 0, 0, 0, 0], [1, 2, 3, 4, 5]], tf.float32)
        scores1 = tf.constant([1.0, 2.1])
        corners2 = tf.constant([[0, 3, 1, 6, 7], [2, 4, 3, 8, 9]], tf.float32)
        boxlist1 = rbox_list.RBoxList(corners1)
        boxlist1.add_field('scores', scores1)
        boxlist2 = rbox_list.RBoxList(corners2)
        with self.assertRaises(ValueError):
            rbox_list_ops.concatenate([boxlist1, boxlist2])

    def test_concatenate_with_incompatible_field_shapes(self):
        rbox1 = tf.constant([[0, 0, 0, 0, 0], [1, 2, 3, 4, 5]], tf.float32)
        scores1 = tf.constant([1.0, 2.1])
        robx2 = tf.constant([[0, 3, 1, 6, 7], [2, 4, 3, 8, 9]], tf.float32)
        scores2 = tf.constant([[1.0, 1.0], [2.1, 3.2]])
        boxlist1 = rbox_list.RBoxList(rbox1)
        boxlist1.add_field('scores', scores1)
        boxlist2 = rbox_list.RBoxList(robx2)
        boxlist2.add_field('scores', scores2)
        with self.assertRaises(ValueError):
            rbox_list_ops.concatenate([boxlist1, boxlist2])

    def test_concatenate_is_correct(self):
        corners1 = tf.constant([[0, 0, 0, 0, 0], [1, 2, 3, 4, 5]], tf.float32)
        scores1 = tf.constant([1.0, 2.1])
        corners2 = tf.constant([[0, 3, 1, 6, 7], [2, 4, 3, 8, 9], [1, 0, 5, 10, 11]], tf.float32)
        scores2 = tf.constant([1.0, 2.1, 5.6])

        exp_corners = [[0, 0, 0, 0, 0],
                       [1, 2, 3, 4, 5],
                       [0, 3, 1, 6, 7],
                       [2, 4, 3, 8, 9],
                       [1, 0, 5, 10, 11]]
        exp_scores = [1.0, 2.1, 1.0, 2.1, 5.6]

        boxlist1 = rbox_list.RBoxList(corners1)
        boxlist1.add_field('scores', scores1)
        boxlist2 = rbox_list.RBoxList(corners2)
        boxlist2.add_field('scores', scores2)
        result = rbox_list_ops.concatenate([boxlist1, boxlist2])
        with self.test_session() as sess:
            corners_output, scores_output = sess.run(
                [result.get(), result.get_field('scores')])
            self.assertAllClose(corners_output, exp_corners)
            self.assertAllClose(scores_output, exp_scores)


class NonMaxSuppressionTest(tf.test.TestCase):
    def test_with_invalid_scores_field(self):
        pass

    def test_select_from_three_clusters(self):
        corners = tf.constant([[0.5, 0.5, 1, 1, 0.0],
                               [0.5, 0.6, 1, 1, 0.0],
                               [0.5, 0.4, 1, 1, 0.0],
                               [0.5, 10.5, 1, 1, 0.0],
                               [0.5, 10.6, 1, 1, 0.0],
                               [0.5, 100.5, 1, 1, 0.0]], tf.float32)
        boxes = rbox_list.RBoxList(corners)
        boxes.add_field('scores', tf.constant([.9, .75, .6, .95, .5, .3]))
        iou_thresh = .5
        max_output_size = 3

        exp_nms = [[0.5, 10.5, 1, 1, 0.0], [0.5, 0.5, 1, 1, 0.0], [0.5, 100.5, 1, 1, 0.0]]
        nms = rbox_list_ops.non_max_suppression(boxes, iou_thresh, max_output_size)
        with self.test_session() as sess:
            nms_output = sess.run(nms.get())
            self.assertAllClose(nms_output, exp_nms)

    def test_select_at_most_two_boxes_from_three_clusters(self):
        corners = tf.constant([[0.5, 0.5, 1, 1, 0.0],
                               [0.5, 0.6, 1, 1, 0.0],
                               [0.5, 0.4, 1, 1, 0.0],
                               [0.5, 10.5, 1, 1, 0.0],
                               [0.5, 10.6, 1, 1, 0.0],
                               [0.5, 100.5, 1, 1, 0.0]], tf.float32)
        boxes = rbox_list.RBoxList(corners)
        boxes.add_field('scores', tf.constant([.9, .75, .6, .95, .5, .3]))
        iou_thresh = .5
        max_output_size = 2

        exp_nms = [[0.5, 10.5, 1, 1, 0.0], [0.5, 0.5, 1, 1, 0.0]]
        nms = rbox_list_ops.non_max_suppression(boxes, iou_thresh, max_output_size)
        with self.test_session() as sess:
            nms_output = sess.run(nms.get())
            self.assertAllClose(nms_output, exp_nms)

    def test_select_at_most_thirty_boxes_from_three_clusters(self):
        corners = tf.constant([[0.5, 0.5, 1, 1, 0.0],
                               [0.5, 0.6, 1, 1, 0.0],
                               [0.5, 0.4, 1, 1, 0.0],
                               [0.5, 10.5, 1, 1, 0.0],
                               [0.5, 10.6, 1, 1, 0.0],
                               [0.5, 100.5, 1, 1, 0.0]], tf.float32)
        boxes = rbox_list.RBoxList(corners)
        boxes.add_field('scores', tf.constant([.9, .75, .6, .95, .5, .3]))
        iou_thresh = .5
        max_output_size = 30

        exp_nms = [[0.5, 10.5, 1, 1, 0.0], [0.5, 0.5, 1, 1, 0.0], [0.5, 100.5, 1, 1, 0.0]]
        nms = rbox_list_ops.non_max_suppression(boxes, iou_thresh, max_output_size)
        with self.test_session() as sess:
            nms_output = sess.run(nms.get())
            self.assertAllClose(nms_output, exp_nms)

    def test_select_at_most_thirty_boxes_from_three_clusters_intersection_tf(self):
        corners = tf.constant([[0.5, 0.5, 1, 1, 0.0],
                               [0.5, 0.6, 1, 1, 0.0],
                               [0.5, 0.4, 1, 1, 0.0],
                               [0.5, 10.5, 1, 1, 0.0],
                               [0.5, 10.6, 1, 1, 0.0],
                               [0.5, 100.5, 1, 1, 0.0]], tf.float32)
        boxes = rbox_list.RBoxList(corners)
        boxes.add_field('scores', tf.constant([.9, .75, .6, .95, .5, .3]))
        iou_thresh = .5
        max_output_size = 30

        exp_nms = [[0.5, 10.5, 1, 1, 0.0], [0.5, 0.5, 1, 1, 0.0], [0.5, 100.5, 1, 1, 0.0]]
        nms = rbox_list_ops.non_max_suppression(boxes, iou_thresh, max_output_size, intersection_tf=True)
        with self.test_session() as sess:
            nms_output = sess.run(nms.get())
            self.assertAllClose(nms_output, exp_nms)

    def test_select_single_box(self):
        corners = tf.constant([[0, 0, 1, 1, 0]], tf.float32)
        rboxes = rbox_list.RBoxList(corners)
        rboxes.add_field('scores', tf.constant([.9]))
        iou_thresh = .5
        max_output_size = 3

        exp_nms = [[0, 0, 1, 1, 0]]
        nms = rbox_list_ops.non_max_suppression(rboxes, iou_thresh, max_output_size)
        with self.test_session() as sess:
            nms_output = sess.run(nms.get())
            self.assertAllClose(nms_output, exp_nms)

    def test_select_from_ten_identical_boxes(self):
        corners = tf.constant(10 * [[0, 0, 1, 1, 0]], tf.float32)
        rboxes = rbox_list.RBoxList(corners)
        rboxes.add_field('scores', tf.constant(10 * [.9]))
        iou_thresh = .5
        max_output_size = 3

        exp_nms = [[0, 0, 1, 1, 0]]
        nms = rbox_list_ops.non_max_suppression(rboxes, iou_thresh, max_output_size)
        with self.test_session() as sess:
            nms_output = sess.run(nms.get())
            self.assertAllClose(nms_output, exp_nms)

    def test_copy_extra_fields(self):
        corners = tf.constant([[0, 0, 1, 1, 0],
                               [0, 0.1, 1, 1.1, 0]], tf.float32)
        rboxes = rbox_list.RBoxList(corners)
        tensor1 = np.array([[1], [4]])
        tensor2 = np.array([[1, 1], [2, 2]])
        rboxes.add_field('tensor1', tf.constant(tensor1))
        rboxes.add_field('tensor2', tf.constant(tensor2))
        new_boxes = rbox_list.RBoxList(tf.constant([[0, 0, 10, 10, 0],
                                                   [1, 3, 5, 5, 0]], tf.float32))
        new_boxes = rbox_list_ops._copy_extra_fields(new_boxes, rboxes)
        with self.test_session() as sess:
            self.assertAllClose(tensor1, sess.run(new_boxes.get_field('tensor1')))
            self.assertAllClose(tensor2, sess.run(new_boxes.get_field('tensor2')))


class CoordinatesConversionTest(tf.test.TestCase):
    def test_to_normalized_coordinates(self):
        coordinates = tf.constant([[0, 0, 100, 100, 0.1],
                                   [25, 25, 75, 75, 0.2]], tf.float32)
        img = tf.ones((128, 100, 100, 3))
        boxlist = rbox_list.RBoxList(coordinates)
        normalized_boxlist = rbox_list_ops.to_normalized_coordinates(boxlist, tf.shape(img)[1], tf.shape(img)[2])
        expected_boxes = [[0, 0, 1, 1, 0.1],
                          [0.25, 0.25, 0.75, 0.75, 0.2]]

        with self.test_session() as sess:
            normalized_boxes = sess.run(normalized_boxlist.get())
            self.assertAllClose(normalized_boxes, expected_boxes)

    def test_to_normalized_coordinates_already_normalized(self):
        coordinates = tf.constant([[0, 0, 1, 1, 0.1],
                                   [0.25, 0.25, 0.75, 0.75, 0.1]], tf.float32)
        img = tf.ones((128, 100, 100, 3))
        boxlist = rbox_list.RBoxList(coordinates)
        normalized_boxlist = rbox_list_ops.to_normalized_coordinates(boxlist,
                                                                     tf.shape(img)[1],
                                                                     tf.shape(img)[2],
                                                                     check_range=True)

        with self.test_session() as sess:
            with self.assertRaisesOpError('assertion failed'):
                sess.run(normalized_boxlist.get())

    def test_to_absolute_coordinates(self):
        coordinates = tf.constant([[0, 0, 1, 1, 0.1],
                                   [0.25, 0.25, 0.75, 0.75, 0.2]], tf.float32)
        img = tf.ones((128, 100, 100, 3))
        boxlist = rbox_list.RBoxList(coordinates)
        absolute_boxlist = rbox_list_ops.to_absolute_coordinates(boxlist,
                                                                 tf.shape(img)[1],
                                                                 tf.shape(img)[2])
        expected_boxes = [[0, 0, 100, 100, 0.1],
                          [25, 25, 75, 75, 0.2]]

        with self.test_session() as sess:
            absolute_boxes = sess.run(absolute_boxlist.get())
            self.assertAllClose(absolute_boxes, expected_boxes)

    def test_to_absolute_coordinates_already_abolute(self):
        coordinates = tf.constant([[0, 0, 100, 100, 0.1],
                                   [25, 25, 75, 75, 0.1]], tf.float32)
        img = tf.ones((128, 100, 100, 3))
        rboxlist = rbox_list.RBoxList(coordinates)
        absolute_rboxlist = rbox_list_ops.to_absolute_coordinates(rboxlist,
                                                                  tf.shape(img)[1],
                                                                  tf.shape(img)[2],
                                                                  check_range=True)

        with self.test_session() as sess:
            with self.assertRaisesOpError('assertion failed'):
                sess.run(absolute_rboxlist.get())

    def test_convert_to_normalized_and_back(self):
        coordinates = np.random.uniform(size=(100, 5))
        coordinates = np.round(np.sort(coordinates) * 200)
        coordinates[:, 2:4] += 1
        coordinates[99, :] = [0, 0, 201, 201, 0.1]
        img = tf.ones((128, 202, 202, 3))

        boxlist = rbox_list.RBoxList(tf.constant(coordinates, tf.float32))
        boxlist = rbox_list_ops.to_normalized_coordinates(boxlist,
                                                          tf.shape(img)[1],
                                                          tf.shape(img)[2])
        boxlist = rbox_list_ops.to_absolute_coordinates(boxlist,
                                                        tf.shape(img)[1],
                                                        tf.shape(img)[2])

        with self.test_session() as sess:
            out = sess.run(boxlist.get())
            self.assertAllClose(out, coordinates)

    def test_convert_to_absolute_and_back(self):
        coordinates = np.random.uniform(size=(100, 5))
        coordinates = np.sort(coordinates)
        coordinates[99, :] = [0, 0, 1, 1, 0.1]
        img = tf.ones((128, 202, 202, 3))

        boxlist = rbox_list.RBoxList(tf.constant(coordinates, tf.float32))
        boxlist = rbox_list_ops.to_absolute_coordinates(boxlist,
                                                        tf.shape(img)[1],
                                                        tf.shape(img)[2])
        boxlist = rbox_list_ops.to_normalized_coordinates(boxlist,
                                                          tf.shape(img)[1],
                                                          tf.shape(img)[2])

        with self.test_session() as sess:
            out = sess.run(boxlist.get())
            self.assertAllClose(out, coordinates)

    def test_convert_rboxes_to_boxes(self):
        rbox = tf.constant([[3.0, 4.0, 6.0, 8.0, 0.0],
                             [14.0, 14.0, 15.0, 15.0, 0.2],
                             [0.0, 0.0, 20.0, 20.0, 0.3]])
        exp_output = [[0, 0, 6, 8],
                      [5.159481, 5.159481, 22.840519, 22.840521],
                      [-12.508567, -12.508567, 12.508567, 12.508567]]

        bbox = rbox_list_ops.convert_rboxes_to_boxes(rbox)
        with self.test_session() as sess:
            output = sess.run(bbox)
            self.assertAllClose(output, exp_output)

    def test_normalized_to_image_coordinates(self):
        normalized_boxes = tf.placeholder(tf.float32, shape=(None, 1, 5))
        normalized_boxes_np = np.array([[[0.0, 0.0, 1.0, 1.0, 0.0]],
                                        [[0.5, 0.5, 1.0, 1.0, 0.1]]])
        image_shape = tf.convert_to_tensor([1, 4, 4, 3], dtype=tf.int32)
        absolute_boxes = rbox_list_ops.normalized_to_image_coordinates(normalized_boxes,
                                                                       image_shape,
                                                                       parallel_iterations=2)

        expected_boxes = np.array([[[0.0, 0.0, 4.0, 4.0, 0.0]],
                                   [[2.0, 2.0, 4.0, 4.0, 0.1]]], dtype=np.float32)
        with self.test_session() as sess:
            absolute_boxes = sess.run(absolute_boxes, feed_dict={normalized_boxes: normalized_boxes_np})
            self.assertAllEqual(absolute_boxes, expected_boxes)

    def test_expand_rboxes(self):
        rboxes = tf.constant([[1.0, 2.0, 3.0, 4.0, 0.1],
                             [5.0, 6.0, 7.0, 8.0, 0.2]])
        exp_output = np.array([[1.0, 2.0, 6.0, 7.0, 0.1],
                               [5.0, 6.0, 14.0, 15.0, 0.2]], dtype=np.float32)

        expand_rboxes = rbox_list_ops.expand_rboxes(rboxes, ratio=2)
        with self.test_session() as sess:
            expand_rboxes = sess.run(expand_rboxes)
            self.assertAllClose(expand_rboxes, exp_output)


if __name__ == '__main__':
    tf.test.main()
