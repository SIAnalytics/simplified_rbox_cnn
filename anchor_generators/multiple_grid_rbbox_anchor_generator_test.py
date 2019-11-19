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

"""Tests for anchor_generators.multiple_grid_rbbox_anchor_generator_test.py."""
import numpy as np

import tensorflow as tf

from anchor_generators import multiple_grid_rbbox_anchor_generator as ag


class MultipleGridRbboxAnchorGeneratorTest(tf.test.TestCase):
    def test_construct_single_anchor_grid(self):
        """Builds a 1x1 anchor grid to test the size of the output boxes."""
        exp_anchors = [[7., -3., 128., 128., 0.], [7., -3., 128., 128., 0.1],
                       [7., -3., 256., 256., 0], [7., -3., 256., 256., 0.1],
                       [7., -3., 512., 512., 0.], [7., -3., 512., 512., 0.1],
                       [7., -3., 64., 256., 0.], [7., -3., 64., 256., 0.1],
                       [7., -3., 128., 512., 0.], [7., -3., 128., 512., 0.1],
                       [7., -3., 256., 1024., 0.], [7., -3., 256., 1024., 0.1]]

        base_anchor_size = tf.constant([256, 256], dtype=tf.float32)
        box_specs_list = [[(.5, 1.0), (1.0, 1.0), (2.0, 1.0),
                           (.5, 4.0), (1.0, 4.0), (2.0, 4.0)]]
        angles = [0, 0.1]
        anchor_generator = ag.MultipleGridRbboxAnchorGenerator(box_specs_list, angles, base_anchor_size)
        anchors = anchor_generator.generate(feature_map_shape_list=[(1, 1)],
                                            anchor_strides=[(16, 16)],
                                            anchor_offsets=[(7, -3)])
        anchor_corners = anchors.get()
        with self.test_session():
            anchor_corners_out = anchor_corners.eval()
            self.assertAllClose(anchor_corners_out, exp_anchors)

    def test_construct_anchor_grid(self):
        base_anchor_size = tf.constant([10, 10], dtype=tf.float32)
        box_specs_list = [[(1.0, 1.0), (2.0, 1.0)]]
        angles = [0, 0.1]

        exp_anchor_corners = [[0., 0., 10., 10., 0.],
                              [0., 0., 10., 10., 0.1],
                              [0., 0., 20., 20., 0.],
                              [0., 0., 20., 20., 0.1],
                              [0., 19., 10., 10., 0.],
                              [0., 19., 10., 10., 0.1],
                              [0., 19., 20., 20., 0.],
                              [0., 19., 20., 20., 0.1],
                              [19., 0., 10., 10., 0.],
                              [19., 0., 10., 10., 0.1],
                              [19., 0., 20., 20., 0.],
                              [19., 0., 20., 20., 0.1],
                              [19., 19., 10., 10., 0.],
                              [19., 19., 10., 10., 0.1],
                              [19., 19., 20., 20., 0.],
                              [19., 19., 20., 20., 0.1]]

        anchor_generator = ag.MultipleGridRbboxAnchorGenerator(box_specs_list, angles, base_anchor_size)
        anchors = anchor_generator.generate(feature_map_shape_list=[(2, 2)],
                                            anchor_strides=[(19, 19)],
                                            anchor_offsets=[(0, 0)])
        anchor_corners = anchors.get()

        with self.test_session():
            anchor_corners_out = anchor_corners.eval()
            self.assertAllClose(anchor_corners_out, exp_anchor_corners)

    def test_construct_anchor_grid_non_square(self):
        base_anchor_size = tf.constant([1, 1], dtype=tf.float32)
        box_specs_list = [[(1.0, 1.0)]]
        angles = [0, 0.1]

        exp_anchor_corners = [[0.5, 0.25, 1., 1., 0], [0.5, 0.25, 1., 1., 0.1],
                              [0.5, 0.75, 1., 1., 0], [0.5, 0.75, 1., 1., 0.1]]

        anchor_generator = ag.MultipleGridRbboxAnchorGenerator(box_specs_list, angles, base_anchor_size)
        anchors = anchor_generator.generate(feature_map_shape_list=[(tf.constant(1, dtype=tf.int32),
                                                                     tf.constant(2, dtype=tf.int32))])
        anchor_corners = anchors.get()

        with self.test_session():
            anchor_corners_out = anchor_corners.eval()
            self.assertAllClose(anchor_corners_out, exp_anchor_corners)

    def test_construct_anchor_grid_unnormalized(self):
        base_anchor_size = tf.constant([1, 1], dtype=tf.float32)
        box_specs_list = [[(1.0, 1.0)]]
        angles = [0, 0.1]

        exp_anchor_corners = [[160., 160., 320., 320., 0.], [160., 160., 320., 320., 0.1],
                              [160., 480., 320., 320., 0.], [160., 480., 320., 320., 0.1]]

        anchor_generator = ag.MultipleGridRbboxAnchorGenerator(box_specs_list, angles, base_anchor_size)
        anchors = anchor_generator.generate(
            feature_map_shape_list=[(tf.constant(1, dtype=tf.int32), tf.constant(2, dtype=tf.int32))],
            im_height=320,
            im_width=640)
        anchor_corners = anchors.get()

        with self.test_session():
            anchor_corners_out = anchor_corners.eval()
            self.assertAllClose(anchor_corners_out, exp_anchor_corners)

    def test_construct_multiple_grids(self):
        base_anchor_size = tf.constant([1.0, 1.0], dtype=tf.float32)
        box_specs_list = [[(1.0, 1.0), (2.0, 1.0), (1.0, 0.5)],
                          [(1.0, 1.0), (1.0, 0.5)]]
        angles = [0, 0.1]

        # height and width of box with .5 aspect ratio
        h = np.sqrt(2)
        w = 1.0 / np.sqrt(2)
        exp_small_grid_corners = [[.75, .25, 1., 1., 0.0],
                                  [.75, .25, 1., 1., 0.1],
                                  [.75, .25, 1. * h, 1. * w, 0.0],
                                  [.75, .25, 1. * h, 1. * w, 0.1],
                                  [.75, .75, 1., 1., 0.0],
                                  [.75, .75, 1., 1., 0.1],
                                  [.75, .75, 1. * h, 1. * w, 0.0],
                                  [.75, .75, 1. * h, 1. * w, 0.1]]
        # only test first entry of larger set of anchors
        exp_big_grid_corners = [[.125, .125, 1., 1., 0.],
                                [.125, .125, 1., 1., 0.1],
                                [.125, .125, 2., 2., 0.]]

        anchor_generator = ag.MultipleGridRbboxAnchorGenerator(box_specs_list, angles, base_anchor_size)
        anchors = anchor_generator.generate(feature_map_shape_list=[(4, 4), (2, 2)],
                                            anchor_strides=[(.25, .25), (.5, .5)],
                                            anchor_offsets=[(.125, .125),
                                                            (.25, .25)])
        anchor_corners = anchors.get()

        with self.test_session():
            anchor_corners_out = anchor_corners.eval()
            self.assertEqual(anchor_corners_out.shape, (112, 5))
            big_grid_corners = anchor_corners_out[0:3, :]
            small_grid_corners = anchor_corners_out[104:, :]
            self.assertAllClose(small_grid_corners, exp_small_grid_corners)
            self.assertAllClose(big_grid_corners, exp_big_grid_corners)

    def test_construct_multiple_grids_with_clipping(self):
        pass
        # base_anchor_size = tf.constant([1.0, 1.0], dtype=tf.float32)
        # box_specs_list = [[(1.0, 1.0), (2.0, 1.0), (1.0, 0.5)],
        #                   [(1.0, 1.0), (1.0, 0.5)]]
        #
        # # height and width of box with .5 aspect ratio
        # h = np.sqrt(2)
        # w = 1.0 / np.sqrt(2)
        # exp_small_grid_corners = [[0, 0, .75, .75],
        #                           [0, 0, .25 + .5 * h, .25 + .5 * w],
        #                           [0, .25, .75, 1],
        #                           [0, .75 - .5 * w, .25 + .5 * h, 1],
        #                           [.25, 0, 1, .75],
        #                           [.75 - .5 * h, 0, 1, .25 + .5 * w],
        #                           [.25, .25, 1, 1],
        #                           [.75 - .5 * h, .75 - .5 * w, 1, 1]]
        #
        # clip_window = tf.constant([0, 0, 1, 1], dtype=tf.float32)
        # anchor_generator = ag.MultipleGridAnchorGenerator(
        #     box_specs_list, base_anchor_size, clip_window=clip_window)
        # anchors = anchor_generator.generate(feature_map_shape_list=[(4, 4), (2, 2)])
        # anchor_corners = anchors.get()
        #
        # with self.test_session():
        #     anchor_corners_out = anchor_corners.eval()
        #     small_grid_corners = anchor_corners_out[48:, :]
        #     self.assertAllClose(small_grid_corners, exp_small_grid_corners)

    def test_invalid_box_specs(self):
        # not all box specs are pairs
        box_specs_list = [[(1.0, 1.0), (2.0, 1.0), (1.0, 0.5)],
                          [(1.0, 1.0), (1.0, 0.5, .3)]]
        angles = [0, 0.1]
        with self.assertRaises(ValueError):
            ag.MultipleGridRbboxAnchorGenerator(box_specs_list, angles)

        # box_specs_list is not a list of lists
        box_specs_list = [(1.0, 1.0), (2.0, 1.0), (1.0, 0.5)]
        with self.assertRaises(ValueError):
            ag.MultipleGridRbboxAnchorGenerator(box_specs_list, angles)

    def test_invalid_generate_arguments(self):
        base_anchor_size = tf.constant([1.0, 1.0], dtype=tf.float32)
        box_specs_list = [[(1.0, 1.0), (2.0, 1.0), (1.0, 0.5)],
                          [(1.0, 1.0), (1.0, 0.5)]]
        angles = [0, 0.1]
        anchor_generator = ag.MultipleGridRbboxAnchorGenerator(box_specs_list, angles, base_anchor_size)

        # incompatible lengths with box_specs_list
        with self.assertRaises(ValueError):
            anchor_generator.generate(feature_map_shape_list=[(4, 4), (2, 2)],
                                      anchor_strides=[(.25, .25)],
                                      anchor_offsets=[(.125, .125), (.25, .25)])
        with self.assertRaises(ValueError):
            anchor_generator.generate(feature_map_shape_list=[(4, 4), (2, 2), (1, 1)],
                                      anchor_strides=[(.25, .25), (.5, .5)],
                                      anchor_offsets=[(.125, .125), (.25, .25)])
        with self.assertRaises(ValueError):
            anchor_generator.generate(feature_map_shape_list=[(4, 4), (2, 2)],
                                      anchor_strides=[(.5, .5)],
                                      anchor_offsets=[(.25, .25)])

        # not pairs
        with self.assertRaises(ValueError):
            anchor_generator.generate(feature_map_shape_list=[(4, 4, 4), (2, 2)],
                                      anchor_strides=[(.25, .25), (.5, .5)],
                                      anchor_offsets=[(.125, .125), (.25, .25)])
        with self.assertRaises(ValueError):
            anchor_generator.generate(feature_map_shape_list=[(4, 4), (2, 2)],
                                      anchor_strides=[(.25, .25, .1), (.5, .5)],
                                      anchor_offsets=[(.125, .125),
                                                      (.25, .25)])
        with self.assertRaises(ValueError):
            anchor_generator.generate(feature_map_shape_list=[(4), (2, 2)],
                                      anchor_strides=[(.25, .25), (.5, .5)],
                                      anchor_offsets=[(.125), (.25)])


class CreateSSDAnchorsTest(tf.test.TestCase):
    def test_create_ssd_anchors_returns_correct_shape(self):
        anchor_generator = ag.create_rssd_anchors(
            num_layers=6, min_scale=0.2, max_scale=0.95,
            aspect_ratios=(1.0, 2.0, 3.0),
            angles=(0, 0.1, 0.2),
            reduce_boxes_in_lowest_layer=True)

        feature_map_shape_list = [(38, 38), (19, 19), (10, 10),
                                  (5, 5), (3, 3), (1, 1)]
        anchors = anchor_generator.generate(
            feature_map_shape_list=feature_map_shape_list)
        anchor_corners = anchors.get()
        with self.test_session():
            anchor_corners_out = anchor_corners.eval()
            self.assertEqual(anchor_corners_out.shape, (14616, 5))

        anchor_generator = ag.create_rssd_anchors(
            num_layers=6, min_scale=0.2, max_scale=0.95,
            aspect_ratios=(1.0, 2.0, 3.0),
            reduce_boxes_in_lowest_layer=False)

        feature_map_shape_list = [(38, 38), (19, 19), (10, 10),
                                  (5, 5), (3, 3), (1, 1)]
        anchors = anchor_generator.generate(
            feature_map_shape_list=feature_map_shape_list)
        anchor_corners = anchors.get()
        with self.test_session():
            anchor_corners_out = anchor_corners.eval()
            self.assertEqual(anchor_corners_out.shape, (23280, 5))


if __name__ == '__main__':
    tf.test.main()
