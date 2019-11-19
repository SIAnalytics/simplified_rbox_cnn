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

r"""Training executable for detection models.

This executable is used to train DetectionModels.

Example usage:
    ./train \
        --logtostderr \
        --train_dir=path/to/train_dir \
        --pipeline_config_path=pipeline_config.pbtxt
"""
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import sys

sys.path.append('./slim')

import functools
import json
import os
import tensorflow as tf

from google.protobuf import text_format

import trainer
from builders import input_reader_builder
from builders import model_builder
from protos import input_reader_pb2
from protos import model_pb2
from protos import pipeline_pb2
from protos import train_pb2

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_string('train_dir', '',
                    'Directory to save the checkpoints and training summaries.')

flags.DEFINE_string('pipeline_config_path', '',
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file. If provided, other configs are ignored')
flags.DEFINE_integer('save_interval_secs', 3600,
                     'Interval in seconds to save a check point file')
flags.DEFINE_integer('log_every_n_steps', 1,
                     'The frequency, in terms of global steps, that the loss and global step are logged.')
FLAGS = flags.FLAGS


def get_configs_from_pipeline_file():
    """Reads training configuration from a pipeline_pb2.TrainEvalPipelineConfig.

    Reads training config from file specified by pipeline_config_path flag.

    Returns:
      model_config: model_pb2.DetectionModel
      train_config: train_pb2.TrainConfig
      input_config: input_reader_pb2.InputReader
    """
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile(FLAGS.pipeline_config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)

    model_config = pipeline_config.model
    train_config = pipeline_config.train_config
    input_config = pipeline_config.train_input_reader

    return model_config, train_config, input_config


def get_configs_from_multiple_files():
    """Reads training configuration from multiple config files.

    Reads the training config from the following files:
      model_config: Read from --model_config_path
      train_config: Read from --train_config_path
      input_config: Read from --input_config_path

    Returns:
      model_config: model_pb2.DetectionModel
      train_config: train_pb2.TrainConfig
      input_config: input_reader_pb2.InputReader
    """
    train_config = train_pb2.TrainConfig()
    with tf.gfile.GFile(FLAGS.train_config_path, 'r') as f:
        text_format.Merge(f.read(), train_config)

    model_config = model_pb2.DetectionModel()
    with tf.gfile.GFile(FLAGS.model_config_path, 'r') as f:
        text_format.Merge(f.read(), model_config)

    input_config = input_reader_pb2.InputReader()
    with tf.gfile.GFile(FLAGS.input_config_path, 'r') as f:
        text_format.Merge(f.read(), input_config)

    return model_config, train_config, input_config


def main(_):
    assert FLAGS.train_dir, '`train_dir` is missing.'

    if FLAGS.pipeline_config_path:
        model_config, train_config, input_config = get_configs_from_pipeline_file()
    else:
        model_config, train_config, input_config = get_configs_from_multiple_files()

    model_fn = functools.partial(
        model_builder.build,
        model_config=model_config,
        is_training=True)

    create_input_dict_fn = functools.partial(
        input_reader_builder.build, input_config)

    env = json.loads(os.environ.get('TF_CONFIG', '{}'))
    cluster_data = env.get('cluster', None)
    cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None
    task_data = env.get('task', None) or {'type': 'master', 'index': 0}
    task_info = type('TaskSpec', (object,), task_data)

    # Parameters for a single worker.
    ps_tasks = 0
    worker_replicas = 1
    worker_job_name = 'lonely_worker'
    task = 0
    is_chief = True
    master = ''

    if cluster_data and 'worker' in cluster_data:
        # Number of total worker replicas include "worker"s and the "master".
        worker_replicas = len(cluster_data['worker']) + 1
    if cluster_data and 'ps' in cluster_data:
        ps_tasks = len(cluster_data['ps'])

    if worker_replicas > 1 and ps_tasks < 1:
        raise ValueError('At least 1 ps task is needed for distributed training.')

    if worker_replicas >= 1 and ps_tasks > 0:
        # Set up distributed training.
        server = tf.train.Server(tf.train.ClusterSpec(cluster), protocol='grpc',
                                 job_name=task_info.type,
                                 task_index=task_info.index)
        if task_info.type == 'ps':
            server.join()
            return

        worker_job_name = '%s/task:%d' % (task_info.type, task_info.index)
        task = task_info.index
        is_chief = (task_info.type == 'master')
        master = server.target

    trainer.train(create_input_dict_fn, model_fn, train_config, input_config, master, task,
                  1, worker_replicas, False, ps_tasks,
                  worker_job_name, is_chief, FLAGS.train_dir, FLAGS.save_interval_secs, FLAGS.log_every_n_steps)


if __name__ == '__main__':
    tf.app.run()
