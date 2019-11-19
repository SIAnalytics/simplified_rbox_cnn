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

"""Detection model evaluator.

This file provides a generic evaluation method that can be used to evaluate a
DetectionModel.
"""
from glob import glob
import re

import logging
import tensorflow as tf

import eval_util
from builders import preprocessor_input_builder
from core import preprocessor_input
from core import box_list
from core import box_list_ops
from core import rbox_list
from core import rbox_list_ops
from core import prefetcher
from core import standard_fields as fields
from utils import ops

slim = tf.contrib.slim

EVAL_METRICS_FN_DICT = {
    'pascal_voc_metrics': eval_util.evaluate_detection_results_pascal_voc
}


def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


def _extract_prediction_tensors(model,
                                create_input_dict_fn,
                                ignore_groundtruth=False,
                                preprocess_input_options=None):
    """Restores the model in a tensorflow session.

    Args:
      model: model to perform predictions with.
      create_input_dict_fn: function to create input tensor dictionaries.
      ignore_groundtruth: whether groundtruth should be ignored.
      preprocess_input_options: a list of tuples, where each tuple contains a
        preprocess input function and a dictionary containing arguments and their
        values (see preprocessor_input.py).

    Returns:
      tensor_dict: A tensor dictionary with evaluations.
    """
    input_dict = create_input_dict_fn()
    prefetch_queue = prefetcher.prefetch(input_dict, capacity=500)
    input_dict = prefetch_queue.dequeue()
    images = tf.expand_dims(input_dict[fields.InputDataFields.image], 0)
    float_images = tf.to_float(images)
    input_dict[fields.InputDataFields.image] = float_images

    if preprocess_input_options:
        input_dict = preprocessor_input.preprocess(input_dict, preprocess_input_options)

    original_image = input_dict[fields.InputDataFields.image]
    preprocessed_image = model.preprocess(original_image)

    prediction_dict = model.predict(preprocessed_image)

    detections = model.postprocess(prediction_dict)

    original_image_shape = tf.shape(original_image)
    if model.is_rbbox:
        absolute_detection_boxlist = rbox_list_ops.to_absolute_coordinates(
            rbox_list.RBoxList(tf.squeeze(detections['detection_boxes'], axis=0)),
            original_image_shape[1], original_image_shape[2])
    else:
        absolute_detection_boxlist = box_list_ops.to_absolute_coordinates(
            box_list.BoxList(tf.squeeze(detections['detection_boxes'], axis=0)),
            original_image_shape[1], original_image_shape[2])
    label_id_offset = 1
    tensor_dict = {
        'original_image': original_image,
        'image_id': input_dict[fields.InputDataFields.source_id],
        'filename': input_dict[fields.InputDataFields.filename],
        'sensor': input_dict[fields.InputDataFields.sensor],
        'detection_boxes': absolute_detection_boxlist.get(),
        'detection_scores': tf.squeeze(detections['detection_scores'], axis=0),
        'detection_classes': (
            tf.squeeze(detections['detection_classes'], axis=0) +
            label_id_offset),
    }
    if 'detection_masks' in detections:
        detection_masks = tf.squeeze(detections['detection_masks'],
                                     axis=0)
        detection_boxes = tf.squeeze(detections['detection_boxes'],
                                     axis=0)
        # TODO: This should be done in model's postprocess function ideally.
        detection_masks_reframed = ops.reframe_box_masks_to_image_masks(
            detection_masks,
            detection_boxes,
            original_image_shape[1], original_image_shape[2])
        detection_masks_reframed = tf.to_float(tf.greater(detection_masks_reframed,
                                                          0.5))

        tensor_dict['detection_masks'] = detection_masks_reframed
    # load groundtruth fields into tensor_dict
    if not ignore_groundtruth:
        if model.is_rbbox:
            normalized_gt_boxlist = rbox_list.RBoxList(input_dict[fields.InputDataFields.groundtruth_rboxes])
            gt_boxlist = rbox_list_ops.scale(normalized_gt_boxlist,
                                             tf.shape(original_image)[1],
                                             tf.shape(original_image)[2])
        else:
            normalized_gt_boxlist = box_list.BoxList(input_dict[fields.InputDataFields.groundtruth_boxes])
            gt_boxlist = box_list_ops.scale(normalized_gt_boxlist,
                                            tf.shape(original_image)[1],
                                            tf.shape(original_image)[2])
        groundtruth_boxes = gt_boxlist.get()
        groundtruth_classes = input_dict[fields.InputDataFields.groundtruth_classes]
        tensor_dict['groundtruth_boxes'] = groundtruth_boxes
        tensor_dict['groundtruth_classes'] = groundtruth_classes
        tensor_dict['area'] = input_dict[fields.InputDataFields.groundtruth_area]
        tensor_dict['is_crowd'] = input_dict[fields.InputDataFields.groundtruth_is_crowd]
        tensor_dict['difficult'] = input_dict[fields.InputDataFields.groundtruth_difficult]
        if 'detection_masks' in tensor_dict:
            tensor_dict['groundtruth_instance_masks'] = input_dict[
                fields.InputDataFields.groundtruth_instance_masks]
    return tensor_dict


def evaluate(create_input_dict_fn, create_model_fn, eval_config, input_config, categories,
             checkpoint_dir, eval_dir, run_mode='latest', is_save_detection_results=False, detection_results_name=''):
    """Evaluation function for detection models.

    Args:
      create_input_dict_fn: a function to create a tensor input dictionary.
      create_model_fn: a function that creates a DetectionModel.
      eval_config: a eval_pb2.EvalConfig protobuf.
      input_config: a input_reader.InputReader protobuf.
      categories: a list of category dictionaries. Each dict in the list should
                  have an integer 'id' field and string 'name' field.
      checkpoint_dir: directory to load the checkpoints to evaluate from.
      eval_dir: directory to write evaluation metrics summary to.
      run_mode: when run_mode is latest, it run infinite and the latest checkpoint is evaluated.
                when run_mode is all, all checkpoints are evaluated and finish evaluation)
      is_save_detection_results: whether or not to save detection results.
      detection_results_name: a filename to a detection_results pickle file.
    """

    model = create_model_fn()

    if eval_config.ignore_groundtruth and not eval_config.export_path:
        logging.fatal('If ignore_groundtruth=True then an export_path is required. Aborting!!!')

    preprocess_input_options = [preprocessor_input_builder.build(step)
                                for step in input_config.preprocess_input_options]

    tensor_dict = _extract_prediction_tensors(model=model,
                                              create_input_dict_fn=create_input_dict_fn,
                                              ignore_groundtruth=eval_config.ignore_groundtruth,
                                              preprocess_input_options=preprocess_input_options)

    def _process_batch(tensor_dict, sess, batch_index, counters, update_op):
        """Evaluates tensors in tensor_dict, visualizing the first K examples.

        This function calls sess.run on tensor_dict, evaluating the original_image
        tensor only on the first K examples and visualizing detections overlaid
        on this original_image.

        Args:
          tensor_dict: a dictionary of tensors
          sess: tensorflow session
          batch_index: the index of the batch amongst all batches in the run.
          counters: a dictionary holding 'success' and 'skipped' fields which can
            be updated to keep track of number of successful and failed runs,
            respectively.  If these fields are not updated, then the success/skipped
            counter values shown at the end of evaluation will be incorrect.
          update_op: An update op that has to be run along with output tensors. For
            example this could be an op to compute statistics for slim metrics.

        Returns:
          result_dict: a dictionary of numpy arrays
        """
        if batch_index >= eval_config.num_visualizations and not is_save_detection_results:
            if 'original_image' in tensor_dict:
                tensor_dict = {k: v for (k, v) in tensor_dict.items()
                               if k != 'original_image'}
        try:
            (result_dict, _) = sess.run([tensor_dict, update_op])
            counters['success'] += 1
        except tf.errors.InvalidArgumentError:
            logging.info('Skipping image')
            counters['skipped'] += 1
            return {}
        global_step = tf.train.global_step(sess, slim.get_global_step())
        if batch_index < eval_config.num_visualizations:
            tag = 'image-{}'.format(batch_index)
            eval_util.visualize_detection_results(
                result_dict, tag, global_step, categories=categories,
                summary_dir=eval_dir,
                export_dir=eval_config.visualization_export_dir)
        return result_dict

    def _process_aggregated_results(result_lists):
        eval_metric_fn_key = eval_config.metrics_set
        if eval_metric_fn_key not in EVAL_METRICS_FN_DICT:
            raise ValueError('Metric not found: {}'.format(eval_metric_fn_key))
        return EVAL_METRICS_FN_DICT[eval_metric_fn_key](result_lists,
                                                        categories=categories,
                                                        box_type='rbox' if model.is_rbbox else 'box',
                                                        is_save_detection_results=is_save_detection_results,
                                                        detection_results_name=detection_results_name)

    variables_to_restore = tf.global_variables()
    global_step = slim.get_or_create_global_step()
    variables_to_restore.append(global_step)
    if eval_config.use_moving_averages:
        variable_averages = tf.train.ExponentialMovingAverage(0.0)
        variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    def _restore_latest_checkpoint(sess):
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        saver.restore(sess, latest_checkpoint)

    class _Restorer(object):
        def __init__(self, checkpoint_dir, run_mode):
            ckpt_list = glob(checkpoint_dir + '/*.index')
            ckpt_list = [ckpt[:-6] for ckpt in ckpt_list]
            sort_nicely(ckpt_list)

            if run_mode == 'latest_once':
                self.ckpt_list = ckpt_list[-1:]
            else:  # run_mode == 'all':
                self.ckpt_list = ckpt_list[1:]

            self.n_ckpt = len(self.ckpt_list)
            self.cnt = 0

        def __call__(self, sess):
            saver.restore(sess, self.ckpt_list[self.cnt])
            self.cnt += 1

        def all_checked(self):
            if self.cnt == self.n_ckpt:
                return True
            else:
                return False

    if run_mode == 'all' or run_mode == 'latest_once':
        eval_util.all_checkpoint_run(
            tensor_dict=tensor_dict,
            update_op=tf.no_op(),
            summary_dir=eval_dir,
            aggregated_result_processor=_process_aggregated_results,
            batch_processor=_process_batch,
            checkpoint_dirs=[checkpoint_dir],
            variables_to_restore=None,
            restorer=_Restorer(checkpoint_dir, run_mode),
            num_batches=eval_config.num_examples,
            master=eval_config.eval_master,
            save_graph=eval_config.save_graph,
            save_graph_dir=(eval_dir if eval_config.save_graph else ''))
    else:  # run_mode == 'latest'
        eval_util.repeated_checkpoint_run(
            tensor_dict=tensor_dict,
            update_op=tf.no_op(),
            summary_dir=eval_dir,
            aggregated_result_processor=_process_aggregated_results,
            batch_processor=_process_batch,
            checkpoint_dirs=[checkpoint_dir],
            variables_to_restore=None,
            restore_fn=_restore_latest_checkpoint,
            num_batches=eval_config.num_examples,
            eval_interval_secs=eval_config.eval_interval_secs,
            max_number_of_evaluations=(
                1 if eval_config.ignore_groundtruth else
                eval_config.max_evals if eval_config.max_evals else
                None),
            master=eval_config.eval_master,
            save_graph=eval_config.save_graph,
            save_graph_dir=(eval_dir if eval_config.save_graph else ''))
