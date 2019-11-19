"""Tests for preprocessor_builder.
This code referred to preprocessor_builder_tester.py
"""

# Author: Jamyoung Koo
# Date created: 2018.02.21
# Date last modified:
# Python Version: 3.5


import tensorflow as tf

from google.protobuf import text_format

from builders import preprocessor_input_builder
from core import preprocessor_input
from protos import input_reader_pb2


class PreprocessorBuilderTest(tf.test.TestCase):

    def assert_dictionary_close(self, dict1, dict2):
        """Helper to check if two dicts with floatst or integers are close."""
        self.assertEqual(sorted(dict1.keys()), sorted(dict2.keys()))
        for key in dict1:
            value = dict1[key]
            if isinstance(value, float):
                self.assertAlmostEqual(value, dict2[key])
            else:
                self.assertEqual(value, dict2[key])

    def test_build_swap_height_width_of_rboxes(self):
        preprocessor_input_text_proto = """
            swap_height_width_of_rboxes {
            } 
        """
        preprocessor_input_proto = input_reader_pb2.PreprocessInputStep()
        text_format.Merge(preprocessor_input_text_proto, preprocessor_input_proto)
        function, args = preprocessor_input_builder.build(preprocessor_input_proto)
        self.assertEqual(function, preprocessor_input.swap_height_width_of_rboxes)
        self.assertEqual(args, {})

    def test_build_subtract_angle_of_rboxes(self):
        preprocessor_input_text_proto = """
            subtract_angle_of_rboxes {
                angle: 0.1 
            } 
        """
        preprocessor_input_proto = input_reader_pb2.PreprocessInputStep()
        text_format.Merge(preprocessor_input_text_proto, preprocessor_input_proto)
        function, args = preprocessor_input_builder.build(preprocessor_input_proto)
        self.assertEqual(function, preprocessor_input.subtract_angle_of_rboxes)
        self.assert_dictionary_close(args, {'angle': 0.1})

    def test_build_normalize_tiff(self):
        preprocessor_input_text_proto = """
        normalize_tiff {
        }
        """
        preprocessor_input_proto = input_reader_pb2.PreprocessInputStep()
        text_format.Merge(preprocessor_input_text_proto, preprocessor_input_proto)
        function, args = preprocessor_input_builder.build(preprocessor_input_proto)
        self.assertEqual(function, preprocessor_input.normalize_tiff)
        self.assertEqual(args, {})


if __name__ == '__main__':
    tf.test.main()
