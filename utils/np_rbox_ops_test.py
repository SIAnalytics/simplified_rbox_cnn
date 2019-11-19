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

"""Tests for object_detection.np_rbox_ops."""
from math import pi
import numpy as np
import tensorflow as tf

from utils import np_rbox_ops

import pyximport; pyximport.install()
from utils import cintersection_rbox


class RBoxOpsTests(tf.test.TestCase):
    def setUp(self):
        rboxes1 = np.array([[4.0, 3.0, 7.0, 5.0, 0.1],
                            [5.0, 6.0, 10.0, 7.0, 0.2]], dtype=np.float32)
        rboxes2 = np.array([[3.0, 4.0, 6.0, 8.0, 0.1],
                            [14.0, 14.0, 15.0, 15.0, 0.2],
                            [0.0, 0.0, 20.0, 20.0, 0.3]], dtype=np.float32)
        expected_intersection = np.array([[27.025812, 0., 35.],
                                          [30.889588, 7.2258, 63.043903]], dtype=np.float32)
        self.rboxes1 = rboxes1
        self.rboxes2 = rboxes2
        self.expected_intersection = expected_intersection

    def get_corner(self, rboxes):
        [cy, cx, h, w, ang] = np.split(rboxes, 5, axis=1)
        h = h / 2
        w = w / 2
        cos = np.cos(ang)
        sin = np.sin(ang)

        lt_x = cx - w * cos + h * sin
        lt_y = cy - w * sin - h * cos
        rt_x = cx + w * cos + h * sin
        rt_y = cy + w * sin - h * cos
        lb_x = cx - w * cos - h * sin
        lb_y = cy - w * sin + h * cos
        rb_x = cx + w * cos - h * sin
        rb_y = cy + w * sin + h * cos

        return np.squeeze(np.stack([lt_y, lt_x, rt_y, rt_x, rb_y, rb_x, lb_y, lb_x], 1), [2])

    def testArea(self):
        areas = np_rbox_ops.area(self.rboxes1)
        expected_areas = np.array([35.0, 70.0], dtype=float)
        self.assertAllClose(expected_areas, areas)

    def testIntersectionShapelyAndFindCorner(self):
        intersection = np_rbox_ops.intersection_shapely_and_find_corner(self.rboxes1, self.rboxes2)
        self.assertAllClose(intersection, self.expected_intersection)

    def testIntersectionShapely(self):
        rboxes1 = self.get_corner(self.rboxes1)
        rboxes2 = self.get_corner(self.rboxes2)

        [cy1, cx1, h1, w1, ang1] = np.split(self.rboxes1, 5, axis=1)
        [cy2, cx2, h2, w2, ang2] = np.split(self.rboxes2, 5, axis=1)

        size = np.sqrt((h1 / 2) ** 2 + (w1 / 2) ** 2) + np.transpose(np.sqrt((h2 / 2) ** 2 + (w2 / 2) ** 2))
        dist = np.sqrt((cx1 - np.transpose(cx2)) ** 2 + (cy1 - np.transpose(cy2)) ** 2)
        is_diff_ang_large_then_pi_2 = np.abs(ang1 - np.transpose(ang2)) > pi / 2

        intersection = np_rbox_ops.intersection_shapely(rboxes1, rboxes2, size, dist, is_diff_ang_large_then_pi_2)
        self.assertAllClose(intersection, self.expected_intersection)

    def testIntersectionOpenCV(self):
        [cy1, cx1, h1, w1, ang1] = np.split(self.rboxes1, 5, axis=1)
        [cy2, cx2, h2, w2, ang2] = np.split(self.rboxes2, 5, axis=1)

        size = np.sqrt((h1 / 2) ** 2 + (w1 / 2) ** 2) + np.transpose(np.sqrt((h2 / 2) ** 2 + (w2 / 2) ** 2))
        dist = np.sqrt((cx1 - np.transpose(cx2)) ** 2 + (cy1 - np.transpose(cy2)) ** 2)
        is_diff_ang_large_then_pi_2 = np.abs(ang1 - np.transpose(ang2)) > pi / 2

        intersection = np_rbox_ops.intersection_opencv(self.rboxes1, self.rboxes2, size, dist,
                                                       is_diff_ang_large_then_pi_2)
        self.assertAllClose(intersection, self.expected_intersection)

    def testIntersectionCython(self):
        [cy1, cx1, h1, w1, ang1] = np.split(self.rboxes1, 5, axis=1)
        [cy2, cx2, h2, w2, ang2] = np.split(self.rboxes2, 5, axis=1)

        size = np.sqrt((h1 / 2) ** 2 + (w1 / 2) ** 2) + np.transpose(np.sqrt((h2 / 2) ** 2 + (w2 / 2) ** 2))
        dist = np.sqrt((cx1 - np.transpose(cx2)) ** 2 + (cy1 - np.transpose(cy2)) ** 2)
        is_diff_ang_large_then_pi_2 = np.abs(ang1 - np.transpose(ang2)) > pi / 2

        intersection = cintersection_rbox.intersection_rbox(self.rboxes1, self.rboxes2, size.astype(np.float32),
                                                            dist.astype(np.float32),
                                                            is_diff_ang_large_then_pi_2.astype(np.uint8))
        self.assertAllClose(intersection, self.expected_intersection)

    def testIntersectionCythonLargeThanPi2(self):
        rboxes1 = np.array([[4.0, 3.0, 7.0, 5.0, pi/2],
                           [5.0, 6.0, 10.0, 7.0, 0.2]], dtype=np.float32)
        rboxes2 = np.array([[3.0, 4.0, 6.0, 8.0, -0.1],
                           [14.0, 14.0, 15.0, 15.0, -pi/2],
                           [0.0, 0.0, 20.0, 20.0, 0.3]], dtype=np.float32)

        expected_intersection = np.array([[-1., -1., 35.],
                                          [30.514746, -1., 63.043903]], dtype=np.float32)

        [cy1, cx1, h1, w1, ang1] = np.split(rboxes1, 5, axis=1)
        [cy2, cx2, h2, w2, ang2] = np.split(rboxes2, 5, axis=1)
        size = np.sqrt((h1 / 2) ** 2 + (w1 / 2) ** 2) + np.transpose(np.sqrt((h2 / 2) ** 2 + (w2 / 2) ** 2))
        dist = np.sqrt((cx1 - np.transpose(cx2)) ** 2 + (cy1 - np.transpose(cy2)) ** 2)
        is_diff_ang_large_then_pi_2 = np.abs(ang1 - np.transpose(ang2)) > pi / 2

        intersection = cintersection_rbox.intersection_rbox(rboxes1, rboxes2, size.astype(np.float32),
                                                            dist.astype(np.float32),
                                                            is_diff_ang_large_then_pi_2.astype(np.uint8))
        self.assertAllClose(intersection, expected_intersection)

    def testMatchedIntersection(self):
        rboxes1 = np.array([[4.0, 3.0, 7.0, 5.0, 0.1],
                            [5.0, 6.0, 10.0, 7.0, 0.2]], dtype=np.float32)
        rboxes2 = np.array([[3.0, 4.0, 6.0, 8.0, 0.1],
                            [14.0, 14.0, 15.0, 15.0, 0.2]], dtype=np.float32)
        expected_intersection = np.array([27.025812, 7.2258], dtype=float)

        intersection = np_rbox_ops.matched_intersection(rboxes1, rboxes2)
        self.assertAllClose(intersection, expected_intersection)

    def testIOU(self):
        iou = np_rbox_ops.iou(self.rboxes1, self.rboxes2)
        expected_iou = np.array([[0.482826, 0., 0.0875],
                                 [0.354603, 0.0251093, 0.154916]],
                                dtype=float)
        self.assertAllClose(iou, expected_iou)


class NonMaximumSuppressionRboxTest(tf.test.TestCase):
    def setUp(self):
        self.rboxes = np.array([[0.5, 0.5, 1, 1, 0.0],
                                [0.5, 0.6, 1, 1, 0.0],
                                [0.5, 0.4, 1, 1, 0.0],
                                [0.5, 10.5, 1, 1, 0.0],
                                [0.5, 10.6, 1, 1, 0.0],
                                [0.5, 100.5, 1, 1, 0.0]], dtype=np.float32)
        self.scores = np.array([.9, .75, .6, .95, .2, .3], dtype=np.float32)

    def test_nms_disabled_max_output_size_equals_three(self):
        max_output_size = 3
        iou_threshold = 1.  # No NMS

        expected_indices = np.array([3, 0, 1], dtype=int)
        nms_indices = np_rbox_ops.non_max_suppression(self.rboxes,
                                                      self.scores,
                                                      max_output_size,
                                                      iou_threshold)
        self.assertAllClose(nms_indices, expected_indices)

    def test_select_from_three_clusters(self):
        max_output_size = 3
        iou_threshold = 0.5

        expected_indices = np.array([3, 0, 5], dtype=np.int32)
        nms_indices = np_rbox_ops.non_max_suppression(self.rboxes,
                                                      self.scores,
                                                      max_output_size,
                                                      iou_threshold)
        self.assertAllClose(nms_indices, expected_indices)

    def test_select_at_most_two_from_three_clusters(self):
        max_output_size = 2
        iou_threshold = 0.5

        expected_indices = np.array([3, 0], dtype=np.int32)
        nms_indices = np_rbox_ops.non_max_suppression(self.rboxes,
                                                      self.scores,
                                                      max_output_size,
                                                      iou_threshold)
        self.assertAllClose(nms_indices, expected_indices)

    def test_select_at_most_thirty_from_three_clusters(self):
        max_output_size = 30
        iou_threshold = 0.5

        expected_indices = np.array([3, 0, 5], dtype=np.int32)
        nms_indices = np_rbox_ops.non_max_suppression(self.rboxes,
                                                      self.scores,
                                                      max_output_size,
                                                      iou_threshold)
        self.assertAllClose(nms_indices, expected_indices)

    def test_select_from_ten_indentical_boxes(self):
        rboxes = np.array(10 * [[0, 0, 1, 1, 0]], dtype=np.float32)
        scores = np.array(10 * [0.8], dtype=np.float32)
        iou_threshold = .5
        max_output_size = 3
        expected_indices = np.array([9], dtype=np.int32)
        nms_indices = np_rbox_ops.non_max_suppression(rboxes,
                                                      scores,
                                                      max_output_size,
                                                      iou_threshold)
        self.assertAllClose(nms_indices, expected_indices)

    def test_different_iou_threshold(self):
        rboxes = np.array([[10, 50, 20, 100, 0],
                          [10, 40, 20, 80, 0],
                          [205, 250, 10, 100, 0],
                          [205, 225, 10, 50, 0]], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
        max_output_size = 4

        iou_threshold = .4
        expected_indices = np.array([0, 2], dtype=np.int32)
        nms_indices = np_rbox_ops.non_max_suppression(rboxes,
                                                      scores,
                                                      max_output_size,
                                                      iou_threshold)
        self.assertAllClose(nms_indices, expected_indices)

        iou_threshold = .5
        expected_indices = np.array([0, 2, 3], dtype=np.int32)
        nms_indices = np_rbox_ops.non_max_suppression(rboxes,
                                                      scores,
                                                      max_output_size,
                                                      iou_threshold)
        self.assertAllClose(nms_indices, expected_indices)

        iou_threshold = .8
        expected_indices = np.array([0, 1, 2, 3], dtype=np.int32)
        nms_indices = np_rbox_ops.non_max_suppression(rboxes,
                                                      scores,
                                                      max_output_size,
                                                      iou_threshold)
        self.assertAllClose(nms_indices, expected_indices)


class SoftNonMaximumSuppressionRboxTest(tf.test.TestCase):
    def setUp(self):
        self.rboxes = np.array([[0.5, 0.5, 1, 1, 0.0],
                                [0.5, 0.6, 1, 1, 0.0],
                                [0.5, 0.4, 1, 1, 0.0],
                                [0.5, 10.5, 1, 1, 0.0],
                                [0.5, 10.6, 1, 1, 0.0],
                                [0.5, 100.5, 1, 1, 0.0]], dtype=np.float32)
        self.scores = np.array([.9, .75, .6, .95, .2, .3], dtype=np.float32)

    def test_select_at_most_thirty_from_three_clusters(self):
        iou_threshold = 0.3

        expected_indices = np.array([3, 0, 5], dtype=np.int32)
        nms_indices = np_rbox_ops.soft_non_max_suppression(self.rboxes,
                                                           self.scores,
                                                           iou_threshold)
        self.assertAllClose(nms_indices, expected_indices)

    def test_select_from_ten_indentical_boxes(self):
        rboxes = np.array(10 * [[0, 0, 1, 1, 0]], dtype=np.float32)
        scores = np.array(9 * [0.8] + [0.9], dtype=np.float32)
        iou_threshold = .5
        expected_indices = np.array([9], dtype=np.int32)
        nms_indices = np_rbox_ops.soft_non_max_suppression(rboxes,
                                                           scores,
                                                           iou_threshold)
        self.assertAllClose(nms_indices, expected_indices)

    def test_different_iou_threshold(self):
        rboxes = np.array([[10, 50, 20, 100, 0],
                          [10, 40, 20, 80, 0],
                          [205, 250, 10, 100, 0],
                          [205, 225, 10, 50, 0]], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)

        iou_threshold = .5
        expected_indices = np.array([0, 2], dtype=np.int32)
        nms_indices = np_rbox_ops.soft_non_max_suppression(rboxes,
                                                           scores,
                                                           iou_threshold)
        self.assertAllClose(nms_indices, expected_indices)


if __name__ == '__main__':
    tf.test.main()
