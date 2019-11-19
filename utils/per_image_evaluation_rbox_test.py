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

"""Tests for object_detection.utils.per_image_evaluation_rbox."""

import numpy as np
import tensorflow as tf

from utils import per_image_evaluation_rbox


class SingleClassTpFpWithDifficultBoxesTest(tf.test.TestCase):

    def setUp(self):
        num_groundtruth_classes = 1
        matching_iou_threshold = 0.5
        self.eval = per_image_evaluation_rbox.PerImageEvaluationRbox(num_groundtruth_classes,
                                                                     matching_iou_threshold)

        self.detected_boxes = np.array([[0, 0, 1, 1, 0.0], [0, 0, 2, 2, 0.0], [0, 0, 3, 3, 0.0]], dtype=np.float32)
        self.detected_scores = np.array([0.6, 0.8, 0.5], dtype=np.float32)
        self.groundtruth_boxes = np.array([[0, 0, 1, 1, 0.0], [0, 0, 10, 10, 0.0]], dtype=np.float32)

    def test_match_to_not_difficult_box(self):
        groundtruth_groundtruth_is_difficult_list = np.array([False, True], dtype=bool)
        scores, tp_fp_labels = self.eval._compute_tp_fp_for_single_class(self.detected_boxes,
                                                                         self.detected_scores,
                                                                         self.groundtruth_boxes,
                                                                         groundtruth_groundtruth_is_difficult_list)
        expected_scores = np.array([0.8, 0.6, 0.5], dtype=np.float32)
        expected_tp_fp_labels = np.array([False, True, False], dtype=bool)
        self.assertTrue(np.allclose(expected_scores, scores))
        self.assertTrue(np.allclose(expected_tp_fp_labels, tp_fp_labels))

    def test_match_to_difficult_box(self):
        groundtruth_groundtruth_is_difficult_list = np.array([True, False], dtype=bool)
        scores, tp_fp_labels = self.eval._compute_tp_fp_for_single_class(self.detected_boxes,
                                                                         self.detected_scores,
                                                                         self.groundtruth_boxes,
                                                                         groundtruth_groundtruth_is_difficult_list)
        expected_scores = np.array([0.8, 0.5], dtype=float)
        expected_tp_fp_labels = np.array([False, False], dtype=bool)
        self.assertTrue(np.allclose(expected_scores, scores))
        self.assertTrue(np.allclose(expected_tp_fp_labels, tp_fp_labels))


class SingleClassTpFpNoDifficultBoxesTest(tf.test.TestCase):

    def setUp(self):
        num_groundtruth_classes = 1
        matching_iou_threshold1 = 0.5
        matching_iou_threshold2 = 0.1

        self.eval1 = per_image_evaluation_rbox.PerImageEvaluationRbox(num_groundtruth_classes,
                                                                      matching_iou_threshold1)

        self.eval2 = per_image_evaluation_rbox.PerImageEvaluationRbox(num_groundtruth_classes,
                                                                      matching_iou_threshold2)

        self.detected_boxes = np.array([[0, 0, 1, 1, 0.0], [0, 0, 2, 2, 0.0], [0, 0, 3, 3, 0.0]], dtype=np.float32)
        self.detected_scores = np.array([0.6, 0.8, 0.5], dtype=np.float32)

    def test_no_true_positives(self):
        groundtruth_boxes = np.array([[100, 100, 105, 105, 0.0]], dtype=np.float32)
        groundtruth_groundtruth_is_difficult_list = np.zeros(1, dtype=bool)
        scores, tp_fp_labels = self.eval1._compute_tp_fp_for_single_class(self.detected_boxes,
                                                                          self.detected_scores,
                                                                          groundtruth_boxes,
                                                                          groundtruth_groundtruth_is_difficult_list)
        expected_scores = np.array([0.8, 0.6, 0.5], dtype=np.float32)
        expected_tp_fp_labels = np.array([False, False, False], dtype=bool)
        self.assertTrue(np.allclose(expected_scores, scores))
        self.assertTrue(np.allclose(expected_tp_fp_labels, tp_fp_labels))

    def test_one_true_positives_with_large_iou_threshold(self):
        groundtruth_boxes = np.array([[0, 0, 1, 1, 0.0]], dtype=np.float32)
        groundtruth_groundtruth_is_difficult_list = np.zeros(1, dtype=bool)
        scores, tp_fp_labels = self.eval1._compute_tp_fp_for_single_class(self.detected_boxes,
                                                                          self.detected_scores,
                                                                          groundtruth_boxes,
                                                                          groundtruth_groundtruth_is_difficult_list)
        expected_scores = np.array([0.8, 0.6, 0.5], dtype=np.float32)
        expected_tp_fp_labels = np.array([False, True, False], dtype=bool)
        self.assertTrue(np.allclose(expected_scores, scores))
        self.assertTrue(np.allclose(expected_tp_fp_labels, tp_fp_labels))

    def test_one_true_positives_with_very_small_iou_threshold(self):
        groundtruth_boxes = np.array([[0, 0, 1, 1, 0.0]], dtype=np.float32)
        groundtruth_groundtruth_is_difficult_list = np.zeros(1, dtype=bool)
        scores, tp_fp_labels = self.eval2._compute_tp_fp_for_single_class(self.detected_boxes,
                                                                          self.detected_scores,
                                                                          groundtruth_boxes,
                                                                          groundtruth_groundtruth_is_difficult_list)
        expected_scores = np.array([0.8, 0.6, 0.5], dtype=np.float32)
        expected_tp_fp_labels = np.array([True, False, False], dtype=bool)
        self.assertTrue(np.allclose(expected_scores, scores))
        self.assertTrue(np.allclose(expected_tp_fp_labels, tp_fp_labels))

    def test_two_true_positives_with_large_iou_threshold(self):
        groundtruth_boxes = np.array([[0, 0, 1, 1, 0.0], [0, 0, 3.5, 3.5, 0.0]], dtype=np.float32)
        groundtruth_groundtruth_is_difficult_list = np.zeros(2, dtype=bool)
        scores, tp_fp_labels = self.eval1._compute_tp_fp_for_single_class(self.detected_boxes,
                                                                          self.detected_scores,
                                                                          groundtruth_boxes,
                                                                          groundtruth_groundtruth_is_difficult_list)
        expected_scores = np.array([0.8, 0.6, 0.5], dtype=float)
        expected_tp_fp_labels = np.array([False, True, True], dtype=bool)
        self.assertTrue(np.allclose(expected_scores, scores))
        self.assertTrue(np.allclose(expected_tp_fp_labels, tp_fp_labels))


class MultiClassesTpFpTest(tf.test.TestCase):

    def test_tp_fp(self):
        num_groundtruth_classes = 3
        matching_iou_threshold = 0.5
        eval1 = per_image_evaluation_rbox.PerImageEvaluationRbox(num_groundtruth_classes,
                                                                 matching_iou_threshold)
        detected_boxes = np.array([[0, 0, 1, 1, 0.0], [0, 0, 2, 2, 0.0], [0, 0, 3, 3, 0.0]], dtype=np.float32)
        detected_scores = np.array([0.8, 0.8, 0.8], dtype=np.float32)
        detected_class_labels = np.array([0, 1, 2], dtype=int)
        groundtruth_boxes = np.array([[0, 0, 1, 1, 0.0], [0, 0, 3.5, 3.5, 0.0]], dtype=np.float32)
        groundtruth_class_labels = np.array([0, 2], dtype=int)
        groundtruth_groundtruth_is_difficult_list = np.zeros(2, dtype=np.float32)
        scores, tp_fp_labels, _ = eval1.compute_object_detection_metrics(detected_boxes,
                                                                         detected_scores,
                                                                         detected_class_labels,
                                                                         groundtruth_boxes,
                                                                         groundtruth_class_labels,
                                                                         groundtruth_groundtruth_is_difficult_list)
        expected_scores = [np.array([0.8], dtype=np.float32)] * 3
        expected_tp_fp_labels = [np.array([True]), np.array([False]), np.array([True])]
        for i in range(len(expected_scores)):
            self.assertTrue(np.allclose(expected_scores[i], scores[i]))
            self.assertTrue(np.array_equal(expected_tp_fp_labels[i], tp_fp_labels[i]))


class CorLocTest(tf.test.TestCase):

    def test_compute_corloc_with_normal_iou_threshold(self):
        num_groundtruth_classes = 3
        matching_iou_threshold = 0.5
        eval1 = per_image_evaluation_rbox.PerImageEvaluationRbox(num_groundtruth_classes, matching_iou_threshold)
        detected_boxes = np.array([[0, 0, 1, 1, 0], [0, 0, 2, 2, 0], [0, 0, 3, 3, 0], [0, 0, 5, 5, 0]], dtype=np.float32)
        detected_scores = np.array([0.9, 0.9, 0.1, 0.9], dtype=np.float32)
        detected_class_labels = np.array([0, 1, 0, 2], dtype=int)
        groundtruth_boxes = np.array([[0, 0, 1, 1, 0], [0, 0, 3, 3, 0], [0, 0, 6, 6, 0]], dtype=np.float32)
        groundtruth_class_labels = np.array([0, 0, 2], dtype=int)

        is_class_correctly_detected_in_image = eval1._compute_cor_loc(detected_boxes,
                                                                      detected_scores,
                                                                      detected_class_labels,
                                                                      groundtruth_boxes,
                                                                      groundtruth_class_labels)
        expected_result = np.array([1, 0, 1], dtype=int)
        self.assertTrue(np.array_equal(expected_result, is_class_correctly_detected_in_image))

    def test_compute_corloc_with_very_large_iou_threshold(self):
        num_groundtruth_classes = 3
        matching_iou_threshold = 0.9

        eval1 = per_image_evaluation_rbox.PerImageEvaluationRbox(num_groundtruth_classes,
                                                                 matching_iou_threshold)
        detected_boxes = np.array([[0, 0, 1, 1, 0], [0, 0, 2, 2, 0], [0, 0, 3, 3, 0], [0, 0, 5, 5, 0]], dtype=np.float32)
        detected_scores = np.array([0.9, 0.9, 0.1, 0.9], dtype=np.float32)
        detected_class_labels = np.array([0, 1, 0, 2], dtype=int)
        groundtruth_boxes = np.array([[0, 0, 1, 1, 0], [0, 0, 3, 3, 0], [0, 0, 6, 6, 0]], dtype=np.float32)
        groundtruth_class_labels = np.array([0, 0, 2], dtype=int)

        is_class_correctly_detected_in_image = eval1._compute_cor_loc(detected_boxes,
                                                                      detected_scores,
                                                                      detected_class_labels,
                                                                      groundtruth_boxes,
                                                                      groundtruth_class_labels)
        expected_result = np.array([1, 0, 0], dtype=int)
        self.assertTrue(np.array_equal(expected_result, is_class_correctly_detected_in_image))


if __name__ == '__main__':
    tf.test.main()
