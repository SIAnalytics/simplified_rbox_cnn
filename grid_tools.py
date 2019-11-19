from math import *

from protos import pipeline_pb2
from google.protobuf import text_format
from sklearn.model_selection import ParameterGrid


def get_config_type(model_config, train_config):
    model_type = model_config.WhichOneof('model')
    optimizer_type = train_config.optimizer.WhichOneof('optimizer')
    learning_rate_type = ''

    if optimizer_type == 'rms_prop_optimizer':
        learning_rate_type = get_learning_rate(train_config.optimizer.rms_prop_optimizer)

    elif optimizer_type == 'momentum_optimizer':
        learning_rate_type = get_learning_rate(train_config.optimizer.momentum_optimizer)

    elif optimizer_type == 'adam_optimizer':
        learning_rate_type = get_learning_rate(train_config.optimizer.adam_optimizer)

    return model_type, optimizer_type, learning_rate_type


def get_learning_rate(optimizer):
    return optimizer.learning_rate.WhichOneof('learning_rate')


def set_rssd_hyperparameter(rssd, config):
    if 'angles' in config.keys():
        rssd.anchor_generator.rssd_anchor_generator.angles[:] = config['angles']

    if 'aspect_ratios' in config.keys():
        rssd.anchor_generator.rssd_anchor_generator.aspect_ratios[:] = config['aspect_ratios']

    if 'max_scale' in config.keys():
        rssd.anchor_generator.rssd_anchor_generator.max_scale = config['max_scale']

    if 'min_scale' in config.keys():
        rssd.anchor_generator.rssd_anchor_generator.max_scale = config['min_scale']


def set_ssd_hyperparameter(ssd, config):
    if 'aspect_ratios' in config.keys():
        ssd.anchor_generator.ssd_anchor_generator.aspect_ratios[:] = config['aspect_ratios']

    if 'max_scale' in config.keys():
        ssd.anchor_generator.ssd_anchor_generator.max_scale = config['max_scale']

    if 'min_scale' in config.keys():
        ssd.anchor_generator.ssd_anchor_generator.max_scale = config['min_scale']


def set_faster_rcnn_hyperparameter(faster_rcnn, config):
    # TODO(SeungHyunJeon): implementation faster rcnn hyperparameter
    pass


def set_rms_hyperparameter(rms, lr_type, config):
    if 'momentum_optimizer_value' in config.keys():
        rms.momentum_optimizer_value = config['momentum_optimizer_value']

    if 'decay' in config.keys():
        rms.decay = config['decay']

    if 'epsilon' in config.keys():
        rms.epsilon = config['epsilon']

    set_learning_rate_hyperparameter(rms.learning_rate, lr_type, config)


def set_momentum_hyperparameter(momentum, lr_type, config):
    # TODO(SeungHyunJeon): implementation momentum hyperparameter
    pass


def set_adam_hyperparameter(adam, lr_type, config):
    # TODO(SeungHyunJeon): implementation adam hyperparameter
    pass


def set_learning_rate_hyperparameter(lr, lr_type, config):
    if lr_type == 'constant_learning_rate':
        pass
    elif lr_type == 'exponential_decay_learning_rate':
        if 'init_learning_rate' in config.keys():
            lr.exponential_decay_learning_rate.initial_learning_rate = config['init_learning_rate']

        if 'decay_steps' in config.keys():
            lr.exponential_decay_learning_rate.decay_steps = config['decay_steps']

        if 'decay_factor' in config.keys():
            lr.exponential_decay_learning_rate.decay_factor = config['decay_factor']

    elif lr_type == 'manual_step_learning_rate':
        pass


def set_hyperparameter(model_type, optimizer_type, learning_rate_type, model_config, train_config, config):
    if model_type == 'rssd':
        set_rssd_hyperparameter(model_config.rssd, config)
    elif model_type == 'ssd':
        set_ssd_hyperparameter(model_config.ssd, config)
    elif model_type == 'faster_rcnn':
        set_ssd_hyperparameter(model_config.faster_rcnn, config)

    if optimizer_type == 'rms_prop_optimizer':
        set_rms_hyperparameter(train_config.optimizer.rms_prop_optimizer, learning_rate_type, config)
    elif optimizer_type == 'momentum_optimizer':
        set_momentum_hyperparameter(train_config.optimizer.momentum_optimizer, learning_rate_type, config)
    elif optimizer_type == 'adam_optimizer':
        set_adam_hyperparameter(train_config.optimizer.adam_optimizer, learning_rate_type, config)


def parse_config(argv):
    argc = len(argv)
    config = {}

    for i in range(1, argc):
        if argv[i][:2] == '--':
            continue
        key, value = argv[i].split('=')
        value = list(map(float, value.split(',')))

        config[key] = value

    return config


def build_config(config):
    if 'angle' in config.keys():
        angles = []

        for step in config['angle']:
            angle = [radians(x) for x in range(-90, 91, int(step))]
            angles.append(angle)

        config['angles'] = angles
        config.pop('angle', None)


def main(argv):
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

    flags = tf.app.flags
    flags.DEFINE_string('pipeline_config_path', 'configs/ssd_inception_v2_hsrc2016_rbbox.config',
                        'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                        'file. If provided, other configs are ignored')

    FLAGS = flags.FLAGS

    config = parse_config(argv)
    build_config(config)
    configs = ParameterGrid(config)

    model_config, train_config, input_config = get_configs_from_pipeline_file()
    model_type, optimizer_type, learning_rate_type = get_config_type(model_config, train_config)
    set_hyperparameter(model_type, optimizer_type, learning_rate_type, model_config, train_config, configs[0])
    print(model_config)


if __name__ == '__main__':
    import sys
    import tensorflow as tf
    tf.app.run(argv=sys.argv)
