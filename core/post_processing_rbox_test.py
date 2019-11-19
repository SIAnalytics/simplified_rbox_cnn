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

"""Tests for tensorflow_models.object_detection.core.post_processing."""
import numpy as np
import tensorflow as tf
from core import post_processing_rbox
from core import standard_fields as fields


class MulticlassNonMaxSuppressionRboxTest(tf.test.TestCase):
    def test_with_invalid_scores_size(self):
        rboxes = tf.constant([[[0.5, 0.5, 1, 1, 0.0]],
                              [[0.5, 0.6, 1, 1, 0.0]],
                              [[0.5, 0.4, 1, 1, 0.0]],
                              [[0.5, 10.5, 1, 1, 0.0]],
                              [[0.5, 10.6, 1, 1, 0.0]],
                              [[0.5, 100.5, 1, 1, 0.0]]], tf.float32)
        scores = tf.constant([[.9], [.75], [.6], [.95], [.5]])
        iou_thresh = .5
        score_thresh = 0.6
        max_output_size = 3
        nms = post_processing_rbox.multiclass_non_max_suppression_rbox(rboxes,
                                                                       scores,
                                                                       score_thresh,
                                                                       iou_thresh,
                                                                       max_output_size)
        with self.test_session() as sess:
            with self.assertRaisesWithPredicateMatch(tf.errors.InvalidArgumentError, 'Incorrect scores field length'):
                sess.run(nms.get())

    def test_multiclass_nms_select_with_shared_boxes(self):
        rboxes = tf.constant([[[0.5, 0.5, 1, 1, 0.0]],
                              [[0.5, 0.6, 1, 1, 0.0]],
                              [[0.5, 0.4, 1, 1, 0.0]],
                              [[0.5, 10.5, 1, 1, 0.0]],
                              [[0.5, 10.6, 1, 1, 0.0]],
                              [[0.5, 100.5, 1, 1, 0.0]],
                              [[0.5, 1001, 1, 2, 0.0]],
                              [[0.5, 1001.05, 1, 2.1, 0.0]]], tf.float32)
        scores = tf.constant([[.9, 0.01], [.75, 0.05],
                              [.6, 0.01], [.95, 0],
                              [.5, 0.01], [.3, 0.01],
                              [.01, .85], [.01, .5]])
        score_thresh = 0.1
        iou_thresh = .5
        max_output_size = 4

        exp_nms_corners = [[0.5, 10.5, 1, 1, 0.0],
                           [0.5, 0.5, 1, 1, 0.0],
                           [0.5, 1001, 1, 2, 0.0],
                           [0.5, 100.5, 1, 1, 0.0]]
        exp_nms_scores = [.95, .9, .85, .3]
        exp_nms_classes = [0, 0, 1, 0]

        nms = post_processing_rbox.multiclass_non_max_suppression_rbox(rboxes,
                                                                       scores,
                                                                       score_thresh,
                                                                       iou_thresh,
                                                                       max_output_size)
        with self.test_session() as sess:
            nms_corners_output, nms_scores_output, nms_classes_output = sess.run(
                [nms.get(), nms.get_field(fields.BoxListFields.scores),
                 nms.get_field(fields.BoxListFields.classes)])
            self.assertAllClose(nms_corners_output, exp_nms_corners)
            self.assertAllClose(nms_scores_output, exp_nms_scores)
            self.assertAllClose(nms_classes_output, exp_nms_classes)

    def test_multiclass_nms_select_with_shared_boxes_intersection_tf(self):
        rboxes = tf.constant([[[0.5, 0.5, 1, 1, 0.0]],
                              [[0.5, 0.6, 1, 1, 0.0]],
                              [[0.5, 0.4, 1, 1, 0.0]],
                              [[0.5, 10.5, 1, 1, 0.0]],
                              [[0.5, 10.6, 1, 1, 0.0]],
                              [[0.5, 100.5, 1, 1, 0.0]],
                              [[0.5, 1001, 1, 2, 0.0]],
                              [[0.5, 1001.05, 1, 2.1, 0.0]]], tf.float32)
        scores = tf.constant([[.9, 0.01], [.75, 0.05],
                              [.6, 0.01], [.95, 0],
                              [.5, 0.01], [.3, 0.01],
                              [.01, .85], [.01, .5]])
        score_thresh = 0.1
        iou_thresh = .5
        max_output_size = 4

        exp_nms_corners = [[0.5, 10.5, 1, 1, 0.0],
                           [0.5, 0.5, 1, 1, 0.0],
                           [0.5, 1001, 1, 2, 0.0],
                           [0.5, 100.5, 1, 1, 0.0]]
        exp_nms_scores = [.95, .9, .85, .3]
        exp_nms_classes = [0, 0, 1, 0]

        nms = post_processing_rbox.multiclass_non_max_suppression_rbox(rboxes,
                                                                       scores,
                                                                       score_thresh,
                                                                       iou_thresh,
                                                                       max_output_size,
                                                                       intersection_tf=True)
        with self.test_session() as sess:
            nms_corners_output, nms_scores_output, nms_classes_output = sess.run(
                [nms.get(), nms.get_field(fields.BoxListFields.scores),
                 nms.get_field(fields.BoxListFields.classes)])
            self.assertAllClose(nms_corners_output, exp_nms_corners)
            self.assertAllClose(nms_scores_output, exp_nms_scores)
            self.assertAllClose(nms_classes_output, exp_nms_classes)

    def test_multiclass_nms_select_with_shared_boxes_given_keypoints(self):
        pass

    def test_multiclass_nms_with_shared_boxes_given_keypoint_heatmaps(self):
        pass

    def test_multiclass_nms_with_additional_fields(self):
        rboxes = tf.constant([[[0.5, 0.5, 1, 1, 0.0]],
                              [[0.5, 0.6, 1, 1, 0.0]],
                              [[0.5, 0.4, 1, 1, 0.0]],
                              [[0.5, 10.5, 1, 1, 0.0]],
                              [[0.5, 10.6, 1, 1, 0.0]],
                              [[0.5, 100.5, 1, 1, 0.0]],
                              [[0.5, 1001, 1, 2, 0.0]],
                              [[0.5, 1001.05, 1, 2.1, 0.0]]], tf.float32)

        scores = tf.constant([[.9, 0.01], [.75, 0.05],
                              [.6, 0.01], [.95, 0],
                              [.5, 0.01], [.3, 0.01],
                              [.01, .85], [.01, .5]])

        coarse_boxes_key = 'coarse_boxes'
        coarse_boxes = tf.constant([[0.1, 0.1, 1.1, 1.1],
                                    [0.1, 0.2, 1.1, 1.2],
                                    [0.1, -0.2, 1.1, 1.0],
                                    [0.1, 10.1, 1.1, 11.1],
                                    [0.1, 10.2, 1.1, 11.2],
                                    [0.1, 100.1, 1.1, 101.1],
                                    [0.1, 1000.1, 1.1, 1002.1],
                                    [0.1, 1000.1, 1.1, 1002.2]], tf.float32)

        score_thresh = 0.1
        iou_thresh = .5
        max_output_size = 4

        exp_nms_corners = np.array([[0.5, 10.5, 1, 1, 0.0],
                                    [0.5, 0.5, 1, 1, 0.0],
                                    [0.5, 1001, 1, 2, 0.0],
                                    [0.5, 100.5, 1, 1, 0.0]], dtype=np.float32)

        exp_nms_coarse_corners = np.array([[0.1, 10.1, 1.1, 11.1],
                                           [0.1, 0.1, 1.1, 1.1],
                                           [0.1, 1000.1, 1.1, 1002.1],
                                           [0.1, 100.1, 1.1, 101.1]],
                                          dtype=np.float32)

        exp_nms_scores = [.95, .9, .85, .3]
        exp_nms_classes = [0, 0, 1, 0]

        nms = post_processing_rbox.multiclass_non_max_suppression_rbox(
            rboxes, scores, score_thresh, iou_thresh, max_output_size,
            additional_fields={coarse_boxes_key: coarse_boxes})

        with self.test_session() as sess:
            (nms_corners_output,
             nms_scores_output,
             nms_classes_output,
             nms_coarse_corners) = sess.run(
                [nms.get(),
                 nms.get_field(fields.BoxListFields.scores),
                 nms.get_field(fields.BoxListFields.classes),
                 nms.get_field(coarse_boxes_key)])

            self.assertAllClose(nms_corners_output, exp_nms_corners)
            self.assertAllClose(nms_scores_output, exp_nms_scores)
            self.assertAllClose(nms_classes_output, exp_nms_classes)
            self.assertAllEqual(nms_coarse_corners, exp_nms_coarse_corners)

    def test_multiclass_nms_select_with_shared_boxes_given_masks(self):
        pass

    def test_multiclass_nms_select_with_clip_window(self):
        pass

    def test_multiclass_nms_select_with_clip_window_change_coordinate_frame(self):
        boxes = tf.constant([[[5, 5, 10, 10, 0]],
                             [[6, 6, 10, 10, 0]]], tf.float32)
        scores = tf.constant([[.9], [.75]])
        clip_window = tf.constant([5, 4, 8, 7], tf.float32)
        score_thresh = 0.0
        iou_thresh = 0.5
        max_output_size = 100

        exp_nms_corners = [[(5 - 5)/3, (5 - 4)/3, 10/3, 10/3, 0]]
        exp_nms_scores = [.9]
        exp_nms_classes = [0]

        nms = post_processing_rbox.multiclass_non_max_suppression_rbox(
            boxes, scores, score_thresh, iou_thresh, max_output_size,
            clip_window=clip_window, change_coordinate_frame=True)
        with self.test_session() as sess:
            nms_corners_output, nms_scores_output, nms_classes_output = sess.run(
                [nms.get(), nms.get_field(fields.BoxListFields.scores),
                 nms.get_field(fields.BoxListFields.classes)])
            self.assertAllClose(nms_corners_output, exp_nms_corners)
            self.assertAllClose(nms_scores_output, exp_nms_scores)
            self.assertAllClose(nms_classes_output, exp_nms_classes)

    def test_multiclass_nms_select_with_per_class_cap(self):
        rboxes = tf.constant([[[0.5, 0.5, 1, 1, 0.0]],
                              [[0.5, 0.6, 1, 1, 0.0]],
                              [[0.5, 0.4, 1, 1, 0.0]],
                              [[0.5, 10.5, 1, 1, 0.0]],
                              [[0.5, 10.6, 1, 1, 0.0]],
                              [[0.5, 100.5, 1, 1, 0.0]],
                              [[0.5, 1001, 1, 2, 0.0]],
                              [[0.5, 1001.05, 1, 2.1, 0.0]]], tf.float32)
        scores = tf.constant([[.9, 0.01], [.75, 0.05],
                              [.6, 0.01], [.95, 0],
                              [.5, 0.01], [.3, 0.01],
                              [.01, .85], [.01, .5]])
        score_thresh = 0.1
        iou_thresh = .5
        max_size_per_class = 2

        exp_nms_corners = [[0.5, 10.5, 1, 1, 0.0],
                           [0.5, 0.5, 1, 1, 0.0],
                           [0.5, 1001, 1, 2, 0.0]]
        exp_nms_scores = [.95, .9, .85]
        exp_nms_classes = [0, 0, 1]

        nms = post_processing_rbox.multiclass_non_max_suppression_rbox(
            rboxes, scores, score_thresh, iou_thresh, max_size_per_class)
        with self.test_session() as sess:
            nms_corners_output, nms_scores_output, nms_classes_output = sess.run(
                [nms.get(), nms.get_field(fields.BoxListFields.scores),
                 nms.get_field(fields.BoxListFields.classes)])
            self.assertAllClose(nms_corners_output, exp_nms_corners)
            self.assertAllClose(nms_scores_output, exp_nms_scores)
            self.assertAllClose(nms_classes_output, exp_nms_classes)

    def test_multiclass_nms_select_with_total_cap(self):
        rboxes = tf.constant([[[0.5, 0.5, 1, 1, 0.0]],
                              [[0.5, 0.6, 1, 1, 0.0]],
                              [[0.5, 0.4, 1, 1, 0.0]],
                              [[0.5, 10.5, 1, 1, 0.0]],
                              [[0.5, 10.6, 1, 1, 0.0]],
                              [[0.5, 100.5, 1, 1, 0.0]],
                              [[0.5, 1001, 1, 2, 0.0]],
                              [[0.5, 1001.05, 1, 2.1, 0.0]]], tf.float32)
        scores = tf.constant([[.9, 0.01], [.75, 0.05],
                              [.6, 0.01], [.95, 0],
                              [.5, 0.01], [.3, 0.01],
                              [.01, .85], [.01, .5]])
        score_thresh = 0.1
        iou_thresh = .5
        max_size_per_class = 4
        max_total_size = 2

        exp_nms_corners = [[0.5, 10.5, 1, 1, 0.0],
                           [0.5, 0.5, 1, 1, 0.0]]
        exp_nms_scores = [.95, .9]
        exp_nms_classes = [0, 0]

        nms = post_processing_rbox.multiclass_non_max_suppression_rbox(
            rboxes, scores, score_thresh, iou_thresh, max_size_per_class,
            max_total_size)
        with self.test_session() as sess:
            nms_corners_output, nms_scores_output, nms_classes_output = sess.run(
                [nms.get(), nms.get_field(fields.BoxListFields.scores),
                 nms.get_field(fields.BoxListFields.classes)])
            self.assertAllClose(nms_corners_output, exp_nms_corners)
            self.assertAllClose(nms_scores_output, exp_nms_scores)
            self.assertAllClose(nms_classes_output, exp_nms_classes)

    def test_multiclass_nms_threshold_then_select_with_shared_boxes(self):
        rboxes = tf.constant([[[0.5, 0.5, 1, 1, 0.0]],
                              [[0.5, 0.6, 1, 1, 0.0]],
                              [[0.5, 0.4, 1, 1, 0.0]],
                              [[0.5, 10.5, 1, 1, 0.0]],
                              [[0.5, 10.6, 1, 1, 0.0]],
                              [[0.5, 100.5, 1, 1, 0.0]],
                              [[0.5, 1001, 1, 2, 0.0]],
                              [[0.5, 1001.05, 1, 2.1, 0.0]]], tf.float32)
        scores = tf.constant([[.9], [.75], [.6], [.95], [.5], [.3], [.01], [.01]])
        score_thresh = 0.1
        iou_thresh = .5
        max_output_size = 3

        exp_nms = [[0.5, 10.5, 1, 1, 0.0],
                   [0.5, 0.5, 1, 1, 0.0],
                   [0.5, 100.5, 1, 1, 0.0]]
        nms = post_processing_rbox.multiclass_non_max_suppression_rbox(
            rboxes, scores, score_thresh, iou_thresh, max_output_size)
        with self.test_session() as sess:
            nms_output = sess.run(nms.get())
            self.assertAllClose(nms_output, exp_nms)

    def test_multiclass_nms_select_with_separate_boxes(self):
        rboxes = tf.constant([[[0.5, 0.5, 1, 1, 0.0], [2, 2.5, 4, 5, 0]],
                              [[0.5, 0.6, 1, 1, 0.0], [1, 0.6, 1, 1, 0]],
                              [[0.5, 0.4, 1, 1, 0.0], [0.5, 0.4, 1, 1, 0]],
                              [[0.5, 10.5, 1, 1, 0.0], [0.5, 10.5, 1, 1, 0]],
                              [[0.5, 10.6, 1, 1, 0.0], [0.5, 10.6, 1, 1, 0]],
                              [[0.5, 100.5, 1, 1, 0.0], [0.5, 100.5, 1, 1, 0]],
                              [[0.5, 1001, 1, 2, 0.0], [1, 1001.5, 2, 5, 0]],
                              [[0.5, 1001.05, 1, 2.1, 0.0], [1, 1000.85, 2, 3.7, 0]]], tf.float32)
        scores = tf.constant([[.9, 0.01], [.75, 0.05],
                              [.6, 0.01], [.95, 0],
                              [.5, 0.01], [.3, 0.01],
                              [.01, .85], [.01, .5]])
        score_thresh = 0.1
        iou_thresh = .5
        max_output_size = 4

        exp_nms_corners = [[0.5, 10.5, 1, 1, 0.0],
                           [0.5, 0.5, 1, 1, 0.0],
                           [1, 1001.5, 2, 5, 0],
                           [0.5, 100.5, 1, 1, 0]]
        exp_nms_scores = [.95, .9, .85, .3]
        exp_nms_classes = [0, 0, 1, 0]

        nms = post_processing_rbox.multiclass_non_max_suppression_rbox(
            rboxes, scores, score_thresh, iou_thresh, max_output_size)
        with self.test_session() as sess:
            nms_corners_output, nms_scores_output, nms_classes_output = sess.run(
                [nms.get(), nms.get_field(fields.BoxListFields.scores),
                 nms.get_field(fields.BoxListFields.classes)])
            self.assertAllClose(nms_corners_output, exp_nms_corners)
            self.assertAllClose(nms_scores_output, exp_nms_scores)
            self.assertAllClose(nms_classes_output, exp_nms_classes)

    def test_multiclass_nms_select_with_separate_boxes_handle_as_single_class(self):
        rboxes = tf.constant([[[0.5, 0.5, 1, 1, 0.0], [2, 2.5, 4, 5, 0]],
                              [[0.5, 0.6, 1, 1, 0.0], [1, 0.6, 1, 1, 0]],
                              [[0.5, 0.4, 1, 1, 0.0], [0.5, 0.4, 1, 1, 0]],
                              [[0.5, 10.5, 1, 1, 0.0], [0.5, 10.5, 1, 1, 0]],
                              [[0.5, 10.6, 1, 1, 0.0], [0.5, 10.6, 1, 1, 0]],
                              [[0.5, 100.5, 1, 1, 0.0], [0.5, 101.5, 1, 1, 0]],
                              [[0.5, 1001, 1, 2, 0.0], [1, 1001.5, 2, 5, 0]],
                              [[0.5, 1001.05, 1, 2.1, 0.0], [1, 1000.85, 2, 3.7, 0]]], tf.float32)
        scores = tf.constant([[.9, 0.01], [.75, 0.05],
                              [.6, .01], [.95, 0.94],
                              [.5, 0.01], [.3, 0.31],
                              [.01, .85], [.01, .5]])
        score_thresh = 0.1
        iou_thresh = .5
        max_output_size = 4

        exp_nms_corners = [[0.5, 10.5, 1, 1, 0.0],
                           [0.5, 0.5, 1, 1, 0.0],
                           [1, 1001.5, 2, 5, 0],
                           [0.5, 101.5, 1, 1, 0]]
        exp_nms_scores = [.95, .9, .85, .31]
        exp_nms_classes = [0., 0., 1., 1.]

        nms = post_processing_rbox.multiclass_non_max_suppression_rbox(
            rboxes, scores, score_thresh, iou_thresh, max_output_size, handle_as_single_class=True)
        with self.test_session() as sess:
            nms_corners_output, nms_scores_output, nms_classes_output = sess.run(
                [nms.get(), nms.get_field(fields.BoxListFields.scores),
                 nms.get_field(fields.BoxListFields.classes)])
            self.assertAllClose(nms_corners_output, exp_nms_corners)
            self.assertAllClose(nms_scores_output, exp_nms_scores)
            self.assertAllClose(nms_classes_output, exp_nms_classes)

    def test_batch_multiclass_nms_with_batch_size_1(self):
        rboxes = tf.constant([[[[0.5, 0.5, 1, 1, 0.0], [2, 2.5, 4, 5, 0]],
                               [[0.5, 0.6, 1, 1, 0.0], [1, 0.6, 1, 1, 0]],
                               [[0.5, 0.4, 1, 1, 0.0], [0.5, 0.4, 1, 1, 0]],
                               [[0.5, 10.5, 1, 1, 0.0], [0.5, 10.5, 1, 1, 0]],
                               [[0.5, 10.6, 1, 1, 0.0], [0.5, 10.6, 1, 1, 0]],
                               [[0.5, 100.5, 1, 1, 0.0], [0.5, 100.5, 1, 1, 0]],
                               [[0.5, 1001, 1, 2, 0.0], [1, 1001.5, 2, 5, 0]],
                               [[0.5, 1001.05, 1, 2.1, 0.0], [1, 1000.85, 2, 3.7, 0]]]], tf.float32)
        scores = tf.constant([[[.9, 0.01], [.75, 0.05],
                               [.6, 0.01], [.95, 0],
                               [.5, 0.01], [.3, 0.01],
                               [.01, .85], [.01, .5]]])
        score_thresh = 0.1
        iou_thresh = .5
        max_output_size = 4

        exp_nms_corners = [[[0.5, 10.5, 1, 1, 0.0],
                           [0.5, 0.5, 1, 1, 0.0],
                           [1, 1001.5, 2, 5, 0],
                           [0.5, 100.5, 1, 1, 0]]]
        exp_nms_scores = [[.95, .9, .85, .3]]
        exp_nms_classes = [[0, 0, 1, 0]]

        (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks,
         num_detections) = post_processing_rbox.batch_multiclass_non_max_suppression_rbox(
            rboxes, scores, score_thresh, iou_thresh,
            max_size_per_class=max_output_size, max_total_size=max_output_size)

        self.assertIsNone(nmsed_masks)

        with self.test_session() as sess:
            (nmsed_boxes, nmsed_scores, nmsed_classes,
             num_detections) = sess.run([nmsed_boxes, nmsed_scores, nmsed_classes,
                                         num_detections])
            self.assertAllClose(nmsed_boxes, exp_nms_corners)
            self.assertAllClose(nmsed_scores, exp_nms_scores)
            self.assertAllClose(nmsed_classes, exp_nms_classes)
            self.assertEqual(num_detections, [4])

    def test_batch_multiclass_nms_with_batch_size_2(self):
        rboxes = tf.constant([[[[0.5, 0.5, 1, 1, 0.0], [2, 2.5, 4, 5, 0]],
                               [[0.5, 0.6, 1, 1, 0.0], [1, 0.6, 1, 1, 0]],
                               [[0.5, 0.4, 1, 1, 0.0], [0.5, 0.4, 1, 1, 0]],
                               [[0.5, 10.5, 1, 1, 0.0], [0.5, 10.5, 1, 1, 0]]],
                              [[[0.5, 10.6, 1, 1, 0.0], [0.5, 10.6, 1, 1, 0]],
                               [[0.5, 100.5, 1, 1, 0.0], [0.5, 100.5, 1, 1, 0]],
                               [[0.5, 1001, 1, 2, 0.0], [1, 1001.5, 2, 5, 0]],
                               [[0.5, 1001.05, 1, 2.1, 0.0], [1, 1000.85, 2, 3.7, 0]]]], tf.float32)
        scores = tf.constant([[[.9, 0.01], [.75, 0.05],
                               [.6, 0.01], [.95, 0]],
                              [[.5, 0.01], [.3, 0.01],
                               [.01, .85], [.01, .5]]])
        score_thresh = 0.1
        iou_thresh = .5
        max_output_size = 4

        exp_nms_corners = np.array([[[0.5, 10.5, 1, 1, 0.0],
                                     [0.5, 0.5, 1, 1, 0.0],
                                     [0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0]],
                                    [[1, 1001.5, 2, 5, 0.0],
                                     [0.5, 10.6, 1, 1, 0.0],
                                     [0.5, 100.5, 1, 1, 0.0],
                                     [0, 0, 0, 0, 0]]], dtype=np.float32)
        exp_nms_scores = np.array([[.95, .9, 0, 0],
                                   [.85, .5, .3, 0]])
        exp_nms_classes = np.array([[0, 0, 0, 0],
                                    [1, 0, 0, 0]])

        (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks,
         num_detections) = post_processing_rbox.batch_multiclass_non_max_suppression_rbox(
            rboxes, scores, score_thresh, iou_thresh,
            max_size_per_class=max_output_size, max_total_size=max_output_size)

        self.assertIsNone(nmsed_masks)
        # Check static shapes
        self.assertAllEqual(nmsed_boxes.shape.as_list(), exp_nms_corners.shape)
        self.assertAllEqual(nmsed_scores.shape.as_list(), exp_nms_scores.shape)
        self.assertAllEqual(nmsed_classes.shape.as_list(), exp_nms_classes.shape)
        self.assertEqual(num_detections.shape.as_list(), [2])

        with self.test_session() as sess:
            (nmsed_boxes, nmsed_scores, nmsed_classes,
             num_detections) = sess.run([nmsed_boxes, nmsed_scores, nmsed_classes, num_detections])
            self.assertAllClose(nmsed_boxes, exp_nms_corners)
            self.assertAllClose(nmsed_scores, exp_nms_scores)
            self.assertAllClose(nmsed_classes, exp_nms_classes)
            self.assertAllClose(num_detections, [2, 3])

    def test_batch_multiclass_nms_with_masks(self):
        pass

    def test_batch_multiclass_nms_with_dynamic_batch_size(self):
        boxes_placeholder = tf.placeholder(tf.float32, shape=(None, None, 2, 5))
        scores_placeholder = tf.placeholder(tf.float32, shape=(None, None, 2))

        boxes = np.array([[[[0.5, 0.5, 1, 1, 0.0], [2, 2.5, 4, 5, 0]],
                           [[0.5, 0.6, 1, 1, 0.0], [1, 0.6, 1, 1, 0]],
                           [[0.5, 0.4, 1, 1, 0.0], [0.5, 0.4, 1, 1, 0]],
                           [[0.5, 10.5, 1, 1, 0.0], [0.5, 10.5, 1, 1, 0]]],
                          [[[0.5, 10.6, 1, 1, 0.0], [0.5, 10.6, 1, 1, 0]],
                           [[0.5, 100.5, 1, 1, 0.0], [0.5, 100.5, 1, 1, 0]],
                           [[0.5, 1001, 1, 2, 0.0], [1, 1001.5, 2, 5, 0]],
                           [[0.5, 1001.05, 1, 2.1, 0.0], [1, 1000.85, 2, 3.7, 0]]]])
        scores = np.array([[[.9, 0.01], [.75, 0.05],
                            [.6, 0.01], [.95, 0]],
                           [[.5, 0.01], [.3, 0.01],
                            [.01, .85], [.01, .5]]])
        score_thresh = 0.1
        iou_thresh = .5
        max_output_size = 4

        exp_nms_corners = np.array([[[0.5, 10.5, 1, 1, 0.0],
                                     [0.5, 0.5, 1, 1, 0.0],
                                     [0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0]],
                                    [[1, 1001.5, 2, 5, 0.0],
                                     [0.5, 10.6, 1, 1, 0.0],
                                     [0.5, 100.5, 1, 1, 0.0],
                                     [0, 0, 0, 0, 0]]], dtype=np.float32)
        exp_nms_scores = np.array([[.95, .9, 0, 0],
                                   [.85, .5, .3, 0]])
        exp_nms_classes = np.array([[0, 0, 0, 0],
                                    [1, 0, 0, 0]])

        (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks,
         num_detections) = post_processing_rbox.batch_multiclass_non_max_suppression_rbox(
            boxes_placeholder, scores_placeholder, score_thresh, iou_thresh,
            max_size_per_class=max_output_size, max_total_size=max_output_size)

        # Check static shapes
        self.assertAllEqual(nmsed_boxes.shape.as_list(), [None, 4, 5])
        self.assertAllEqual(nmsed_scores.shape.as_list(), [None, 4])
        self.assertAllEqual(nmsed_classes.shape.as_list(), [None, 4])
        self.assertEqual(num_detections.shape.as_list(), [None])

        with self.test_session() as sess:
            (nmsed_boxes, nmsed_scores, nmsed_classes,
             num_detections) = sess.run([nmsed_boxes, nmsed_scores, nmsed_classes,
                                         num_detections],
                                        feed_dict={boxes_placeholder: boxes,
                                                   scores_placeholder: scores})
            self.assertAllClose(nmsed_boxes, exp_nms_corners)
            self.assertAllClose(nmsed_scores, exp_nms_scores)
            self.assertAllClose(nmsed_classes, exp_nms_classes)
            self.assertAllClose(num_detections, [2, 3])

    def test_batch_multiclass_nms_with_masks_and_num_valid_boxes(self):
        pass


if __name__ == '__main__':
    tf.test.main()
