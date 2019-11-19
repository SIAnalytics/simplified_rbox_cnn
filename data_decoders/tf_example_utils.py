"""Utilities for tfrecords"""

# Author: JamyoungKoo
# Date created: 2018.08.13
# Python Version: 3.5

import os
import numpy as np
import tensorflow as tf
from data_decoders import tf_example_decoder


def read_tfrecord(src_path, n_sample=9, dtype='uint8'):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    def _read_and_decode(queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(queue)
        example_decoder = tf_example_decoder.TfExampleDecoder(dtype=dtype)
        tensor_dict = example_decoder.decode(tf.convert_to_tensor(serialized_example))
        return tensor_dict

    queue = tf.train.string_input_producer([src_path])
    tensor_dict = _read_and_decode(queue)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    images = []
    rboxes = []
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(n_sample):
            input_data = sess.run(tensor_dict)
            images.append(input_data['image'].astype(dtype=np.uint8))
            rboxes.append(input_data['groundtruth_rboxes'])

        coord.request_stop()
        coord.join(threads)

    return images, rboxes

