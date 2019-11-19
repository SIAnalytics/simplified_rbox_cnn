###############################################################################
# Developed 2018 by SIA, SI Analytics, Co., Ltd.                              #
# 441 Expo-ro, Yuseong-gu, Daejeon, 305-714, Korea (Munji R&D)                #
# SI Analytics http://www.si-a.ai All rights reserved.                        #
#                                                                             #
# This software is the confidential information of SIA                        #
# You shall not disclose such Confidential Information                        #
# and shall use it only in accordance with the terms of the license agreement #
# you entered into with SIA.                                                  #
###############################################################################


import os
import argparse
import csv
import math

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from skimage.io import imread
from google.protobuf import text_format

from builders import model_builder
from protos import pipeline_pb2
from utils.np_rbox_ops import non_max_suppression


def get_detection_graph(pipeline_config_path):
    """build a graph from pipline_config_path

    :param: str pipeline_config_path: path to pipeline config file

    :return: graph
    """

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile(pipeline_config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)

    detection_model = model_builder.build(pipeline_config.model, is_training=False)
    input_tensor = tf.placeholder(dtype=tf.uint8, shape=(None, None, None, 3), name='image_tensor')
    inputs = tf.to_float(input_tensor)
    preprocessed_inputs = detection_model.preprocess(inputs)
    output_tensors = detection_model.predict(preprocessed_inputs)
    postprocessed_tensors = detection_model.postprocess(output_tensors)

    output_collection_name = 'inference_op'
    boxes = postprocessed_tensors.get('detection_boxes')
    scores = postprocessed_tensors.get('detection_scores')
    classes = postprocessed_tensors.get('detection_classes') + 1
    num_detections = postprocessed_tensors.get('num_detections')
    outputs = dict()
    outputs['detection_boxes'] = tf.identity(boxes, name='detection_boxes')
    outputs['detection_scores'] = tf.identity(scores, name='detection_scores')
    outputs['detection_classes'] = tf.identity(classes, name='detection_classes')
    outputs['num_detections'] = tf.identity(num_detections, name='num_detections')
    for output_key in outputs:
        tf.add_to_collection(output_collection_name, outputs[output_key])

    graph = tf.get_default_graph()

    return graph


def convert_rbox_to_poly(rbox):
    """ Convert RBox to polygon as 4 points

    :param numpy rbox: rotated bounding box as [cy, cx, height, width, angle]
    :return: list of tuple as 4 corner points
    """

    cy, cx = rbox[0], rbox[1]
    height, width = rbox[2], rbox[3]
    angle = rbox[4]

    lt_x, lt_y = -width / 2, -height / 2
    rt_x, rt_y = width / 2, -height / 2
    lb_x, lb_y = -width / 2, height / 2
    rb_x, rb_y = width / 2, height / 2

    lt_x_ = lt_x * math.cos(angle) - lt_y * math.sin(angle)
    lt_y_ = lt_x * math.sin(angle) + lt_y * math.cos(angle)
    rt_x_ = rt_x * math.cos(angle) - rt_y * math.sin(angle)
    rt_y_ = rt_x * math.sin(angle) + rt_y * math.cos(angle)
    lb_x_ = lb_x * math.cos(angle) - lb_y * math.sin(angle)
    lb_y_ = lb_x * math.sin(angle) + lb_y * math.cos(angle)
    rb_x_ = rb_x * math.cos(angle) - rb_y * math.sin(angle)
    rb_y_ = rb_x * math.sin(angle) + rb_y * math.cos(angle)

    lt_x_ = lt_x_ + cx
    lt_y_ = lt_y_ + cy
    rt_x_ = rt_x_ + cx
    rt_y_ = rt_y_ + cy
    lb_x_ = lb_x_ + cx
    lb_y_ = lb_y_ + cy
    rb_x_ = rb_x_ + cx
    rb_y_ = rb_y_ + cy

    return [(lt_x_, lt_y_), (rt_x_, rt_y_), (rb_x_, rb_y_), (lb_x_, lb_y_)]


def save_det_to_csv(dst_path, det_by_file):
    """ Save detected objects to CSV format

    :param str dst_path: Path to save csv
    :param dict det_by_file: detected objects that key is filename
    :return: None (save csv file)
    """
    with open(dst_path, 'w') as f:
        w = csv.DictWriter(f, ['file_name', 'class_id', 'confidence', 'point1_x', 'point1_y', 'point2_x', 'point2_y',
                               'point3_x', 'point3_y', 'point4_x', 'point4_y'])
        w.writeheader()

        for file_path, det in det_by_file.items():
            rboxes = det['rboxes']
            classes = det['classes']
            scores = det['scores']

            for rbox, cls, score in zip(rboxes, classes, scores):
                poly = convert_rbox_to_poly(rbox)
                det_dict = {'file_name': os.path.basename(file_path),
                            'class_id': cls,
                            'confidence': score,
                            'point1_x': poly[0][0],
                            'point1_y': poly[0][1],
                            'point2_x': poly[1][0],
                            'point2_y': poly[1][1],
                            'point3_x': poly[2][0],
                            'point3_y': poly[2][1],
                            'point4_x': poly[3][0],
                            'point4_y': poly[3][1],
                            }
                w.writerow(det_dict)


def get_patch_generator(image, patch_size, overlay_size):
    """ Patch Generator to split image by grid

    :param numpy image: source image
    :param int patch_size: patch size that width and height of patch is equal
    :param overlay_size: overlay size in patches
    :return: generator for patch image, row and col coordinates
    """
    step = patch_size - overlay_size
    for row in range(0, image.shape[0] - overlay_size, step):
        for col in range(0, image.shape[1] - overlay_size, step):
            # Handling for out of bounds
            patch_image_height = patch_size if image.shape[0] - row > patch_size else image.shape[0] - row
            patch_image_width = patch_size if image.shape[1] - col > patch_size else image.shape[1] - col

            # Set patch image
            patch_image = image[row: row + patch_image_height, col: col + patch_image_width]

            # Zero padding if patch image is smaller than patch size
            if patch_image_height < patch_size or patch_image_width < patch_size:
                pad_height = patch_size - patch_image_height
                pad_width = patch_size - patch_image_width
                patch_image = np.pad(patch_image, ((0, pad_height), (0, pad_width), (0, 0)), 'constant')

            yield patch_image, row, col


def inference(pipeline_config_path, ckpt_path, image_dir, dst_path, patch_size, overlay_size):
    """ Inference images to detect objects

    :param str pipeline_config_path: path to a pipeline_pb2.TrainEvalPipelineConfig config file
    :param str ckpt_path: path to trained checkpoint
    :param str image_dir: directory to source images
    :param str dst_path: path to save detection output
    :param int patch_size: patch size that width and height of patch is equal
    :param int overlay_size: overlay size in patches
    :return: None (save detection output)

    """
    # Get filenames
    file_paths = [os.path.join(root, name) for root, dirs, files in os.walk(image_dir) for name in files if
                  name.endswith('png')]

    # Create graph
    graph = get_detection_graph(pipeline_config_path)

    # Inference
    with tf.Session(graph=graph) as sess:
        # Load weights from a checkpoint file
        variables_to_restore = tf.global_variables()
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(sess, ckpt_path)

        # Get tensors of detection model
        image_tensor = graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = graph.get_tensor_by_name('detection_scores:0')
        detection_classes = graph.get_tensor_by_name('detection_classes:0')

        # Run detection
        det_by_file = dict()
        for file_path in tqdm(file_paths):
            image = imread(file_path)
            patch_generator = get_patch_generator(image, patch_size=patch_size, overlay_size=overlay_size)

            classes_list, scores_list, rboxes_list = list(), list(), list()
            for patch_image, row, col in patch_generator:
                classes, scores, rboxes = sess.run([detection_classes, detection_scores, detection_boxes],
                                                   feed_dict={image_tensor: [patch_image]})

                rboxes = rboxes[0]
                classes = classes[0]
                scores = scores[0]

                rboxes *= [patch_image.shape[0], patch_image.shape[1], patch_image.shape[0], patch_image.shape[1], 1]
                rboxes[:, 0] = rboxes[:, 0] + row
                rboxes[:, 1] = rboxes[:, 1] + col


                rboxes_list.append(rboxes)
                classes_list.append(classes)
                scores_list.append(scores)

            rboxes = np.array(rboxes_list).reshape(-1, 5)
            classes = np.array(classes_list).flatten()
            scores = np.array(scores_list).flatten()

            rboxes = rboxes[scores > 0]
            classes = classes[scores > 0]
            scores = scores[scores > 0]

            indices = non_max_suppression(rboxes, scores, iou_threshold=0.3)
            rboxes = rboxes[indices]
            classes = classes[indices]
            scores = scores[indices]

            det_by_file[file_path] = {'rboxes': rboxes, 'classes': classes, 'scores': scores}

        # Save detection output
        save_det_to_csv(dst_path, det_by_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pipeline_config_path', type=str,
                        help='Path to a pipeline_pb2.TrainEvalPipelineConfig config file.')
    parser.add_argument('--ckpt_path', type=str,
                        help='Path to trained checkpoint, typically of the form path/to/model-%step.ckpt')
    parser.add_argument('--image_dir', type=str,
                        help='Path to images to be inferred')
    parser.add_argument('--dst_path', type=str,
                        help='Path to save detection output')
    parser.add_argument('--patch_size', type=int, default=1024,
                        help='Patch size, width and height of patch is equal.')
    parser.add_argument('--overlay_size', type=int, default=384,
                        help='Overlay size for patching.')

    args = parser.parse_args()

    inference(**vars(args))
