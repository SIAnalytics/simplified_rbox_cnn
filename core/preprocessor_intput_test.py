"""Tests for object_detection.core.preprocessor_input.
This code referred to preprocessor_test.py
"""

# Author: Jamyoung Koo
# Date created: 2018.02.21
# Date last modified: 2018.05.14
# Python Version: 3.5

import tensorflow as tf

from core import standard_fields as fields
from core import preprocessor_input


class PreprocessorTest(tf.test.TestCase):

    def createTestImages(self):
        images_r = tf.constant([[[128, 128, 128, 128], [0, 0, 128, 128],
                                 [0, 128, 128, 128], [192, 192, 128, 128]]],
                               dtype=tf.float32)
        images_r = tf.expand_dims(images_r, 3)
        images_g = tf.constant([[[0, 0, 128, 128], [0, 0, 128, 128],
                                 [0, 128, 192, 192], [192, 192, 128, 192]]],
                               dtype=tf.float32)
        images_g = tf.expand_dims(images_g, 3)
        images_b = tf.constant([[[128, 128, 192, 0], [0, 0, 128, 192],
                                 [0, 128, 128, 0], [192, 192, 192, 128]]],
                               dtype=tf.float32)
        images_b = tf.expand_dims(images_b, 3)
        images = tf.concat([images_r, images_g, images_b], 3)
        return images

    def expectedImagesAfterNormalizationTiff(self):
        images_r = tf.constant([[[127.5, 127.5, 127.5, 127.5], [0, 0, 127.5, 127.5],
                                 [0, 127.5, 127.5, 127.5], [255, 255, 127.5, 127.5]]],
                               dtype=tf.float32)
        images_r = tf.expand_dims(images_r, 3)
        images_g = tf.constant([[[0, 0, 127.5, 127.5], [0, 0, 127.5, 127.5],
                                 [0, 127.5, 255, 255], [255, 255, 127.5, 255]]],
                               dtype=tf.float32)
        images_g = tf.expand_dims(images_g, 3)
        images_b = tf.constant([[[127.5, 127.5, 255, 0], [0, 0, 127.5, 255],
                                 [0, 127.5, 127.5, 0], [255, 255, 255, 127.5]]],
                               dtype=tf.float32)
        images_b = tf.expand_dims(images_b, 3)
        images = tf.concat([images_r, images_g, images_b], 3)
        return images

    def createTestImagesK3(self):
        images_r = tf.constant([[[2905, 2905, 2905, 2905], [984, 984, 2905, 2905],
                                 [984, 2905, 2905, 2905], [4826, 4826, 2905, 2905]]],
                               dtype=tf.float32)
        images_r = tf.expand_dims(images_r, 3)
        images_g = tf.constant([[[1405, 1405, 2820.5, 2820.5], [1405, 1405, 2820.5, 2820.5],
                                 [1405, 2820.5, 4236, 4236], [4236, 4236, 2820.5, 4236]]],
                               dtype=tf.float32)
        images_g = tf.expand_dims(images_g, 3)
        images_b = tf.constant([[[4676.5, 4676.5, 6581, 2772], [2772, 2772, 4676.5, 6581],
                                 [2772, 4676.5, 4676.5, 2772], [6581, 6581, 6581, 4676.5]]],
                               dtype=tf.float32)
        images_b = tf.expand_dims(images_b, 3)
        images = tf.concat([images_r, images_g, images_b], 3)
        return images

    def createTestRBoxes(self):
        rboxes = tf.constant(
            [[0.0, 0.25, 0.75, 1.0, -1.0], [0.25, 0.5, 0.75, 1.0, 0.2]], dtype=tf.float32)
        return rboxes

    def expectedRBoxesSwapHeightWidth(self):
        rboxes = tf.constant(
            [[0.0, 0.25, 1.0, 0.75, -1.0], [0.25, 0.5, 1.0, 0.75, 0.2]], dtype=tf.float32)
        return rboxes

    def expectedRBoxesSubtractAngle(self):
        rboxes = tf.constant(
            [[0.0, 0.25, 0.75, 1.0, -1.5], [0.25, 0.5, 0.75, 1.0, -0.3]], dtype=tf.float32)
        return rboxes

    def testSwapHeightWidthOfRboxes(self):
        rboxes = self.createTestRBoxes()
        rboxes_expected = self.expectedRBoxesSwapHeightWidth()
        swapped_rbboxes = preprocessor_input.swap_height_width_of_rboxes(rboxes)

        with self.test_session() as sess:
            swapped_rbboxes, rboxes_expected = sess.run([swapped_rbboxes, rboxes_expected])
            self.assertAllEqual(swapped_rbboxes, rboxes_expected)

    def testSubtractAngleOfRboxes(self):
        rboxes = self.createTestRBoxes()
        rboxes_expected = self.expectedRBoxesSubtractAngle()
        subtracted_rbboxes = preprocessor_input.subtract_angle_of_rboxes(rboxes, 0.5)

        with self.test_session() as sess:
            subtracted_rbboxes, rboxes_expected = sess.run([subtracted_rbboxes, rboxes_expected])
            self.assertAllEqual(subtracted_rbboxes, rboxes_expected)

    def testNormalizeTiffOfK3(self):
        preprocess_options = [(preprocessor_input.normalize_tiff, {})]
        images = self.createTestImagesK3()
        tensor_dict = {fields.InputDataFields.image: images,
                       fields.InputDataFields.sensor: 'K3'}
        tensor_dict = preprocessor_input.preprocess(tensor_dict, preprocess_options)
        images = tensor_dict[fields.InputDataFields.image]
        images_expected = self.expectedImagesAfterNormalizationTiff()

        with self.test_session() as sess:
            (images_, images_expected_) = sess.run([images, images_expected])
            images_shape_ = images_.shape
            images_expected_shape_ = images_expected_.shape
            expected_shape = [1, 4, 4, 3]
            self.assertAllEqual(images_expected_shape_, images_shape_)
            self.assertAllEqual(images_shape_, expected_shape)
            self.assertAllClose(images_, images_expected_)

    def testCutoffMaxGTs(self):
        preprocess_options = [(preprocessor_input.cutoff_max_gts, {'max_gts': 4})]
        image = self.createTestImages()
        rboxes = tf.constant([[0, 0, 1, 1, 0.1],
                              [1, 0.1, 1, 1.1, 0.2],
                              [2, -0.1, 1, 0.9, 0.3],
                              [3, 10, 1, 11, 0.4],
                              [4, 10.1, 1, 11.1, 0.5],
                              [5, 100, 1, 101, 0.6]], tf.float32)
        classes = tf.constant([0, 1, 2, 3, 4, 5])

        tensor_dict = {fields.InputDataFields.image: image,
                       fields.InputDataFields.groundtruth_rboxes: rboxes,
                       fields.InputDataFields.groundtruth_classes: classes}
        tensor_dict = preprocessor_input.preprocess(tensor_dict, preprocess_options)
        image = tensor_dict[fields.InputDataFields.image]
        rboxes = tensor_dict[fields.InputDataFields.groundtruth_rboxes]
        classes = tensor_dict[fields.InputDataFields.groundtruth_classes]

        with self.test_session() as sess:
            (image, rboxes_, classes_) = sess.run([image, rboxes, classes])
            self.assertAllEqual(rboxes_.shape[0], 4)
            self.assertAllEqual(classes_.shape[0], 4)
            self.assertAllEqual(rboxes_[:, 0], classes_)


if __name__ == '__main__':
    tf.test.main()
