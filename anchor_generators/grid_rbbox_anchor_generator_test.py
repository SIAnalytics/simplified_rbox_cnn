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

"""Tests for object_detection.grid_rbox_anchor_generator."""
import tensorflow as tf

from anchor_generators import grid_rbbox_anchor_generator


class GridRBboxAnchorGeneratorTest(tf.test.TestCase):
    def test_construct_single_anchor(self):
        """Builds a 1x1 anchor grid to test the size of the output boxes."""
        scales = [0.5, 1.0, 2.0]
        aspect_ratios = [1.0, 4.0]
        anchor_offset = [7, -3]
        angles = [0, 0.1]

        exp_anchor = [[7., -3., 128., 128., 0.], [7., -3., 128., 128., 0.1],
                      [7., -3., 256., 256., 0], [7., -3., 256., 256., 0.1],
                      [7., -3., 512., 512., 0.], [7., -3., 512., 512., 0.1],
                      [7., -3., 64., 256., 0.], [7., -3., 64., 256., 0.1],
                      [7., -3., 128., 512., 0.], [7., -3., 128., 512., 0.1],
                      [7., -3., 256., 1024., 0.], [7., -3., 256., 1024., 0.1]]

        anchor_generator = grid_rbbox_anchor_generator.GridRbboxAnchorGenerator(scales,
                                                                                aspect_ratios,
                                                                                angles,
                                                                                anchor_offset=anchor_offset)
        anchors = anchor_generator.generate(feature_map_shape_list=[(1, 1)])
        anchor_rbox = anchors.get()

        with self.test_session():
            anchor_rbox_out = anchor_rbox.eval()
            self.assertAllClose(anchor_rbox_out, exp_anchor)

    def test_construct_anchor_grid(self):
        base_anchor_size = [10, 10]
        anchor_stride = [19, 19]
        anchor_offset = [0, 0]
        scales = [1.0, 2.0]
        aspect_ratios = [1.0]
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

        anchor_generator = grid_rbbox_anchor_generator.GridRbboxAnchorGenerator(
            scales,
            aspect_ratios,
            angles,
            base_anchor_size=base_anchor_size,
            anchor_stride=anchor_stride,
            anchor_offset=anchor_offset)

        anchors = anchor_generator.generate(feature_map_shape_list=[(2, 2)])
        anchor_rbox = anchors.get()

        with self.test_session():
            anchor_rbox = anchor_rbox.eval()
            self.assertAllClose(anchor_rbox, exp_anchor_corners)


if __name__ == '__main__':
    tf.test.main()
