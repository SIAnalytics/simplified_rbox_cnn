"""Builder for preprocessing steps for input data.
This code referred to preprocessor_builder.py
"""

# Author: Jamyoung Koo
# Date created: 2018.02.21
# Date last modified: 2018.05.14
# Python Version: 3.5

from core import preprocessor_input


def _get_step_config_from_proto(preprocessor_input_step_config, step_name):
    """Returns the value of a field named step_name from proto.

    Args:
      preprocessor_input_step_config: A input_reader_pb2.PreprocessInputStep object.
      step_name: Name of the field to get value from.

    Returns:
      result_dict: a sub proto message from preprocessor_input_step_config which will be
                   later converted to a dictionary.

    Raises:
      ValueError: If field does not exist in proto.
    """
    for field, value in preprocessor_input_step_config.ListFields():
        if field.name == step_name:
            return value

    raise ValueError('Could not get field %s from proto!', step_name)


def _get_dict_from_proto(config):
    """Helper function to put all proto fields into a dictionary.

    For many preprocessing input steps, there's an trivial 1-1 mapping from proto fields
    to function arguments. This function automatically populates a dictionary with
    the arguments from the proto.

    Protos that CANNOT be trivially populated include:
    * nested messages.
    * steps that check if an optional field is set (ie. where None != 0).
    * protos that don't map 1-1 to arguments (ie. list should be reshaped).
    * fields requiring additional validation (ie. repeated field has n elements).

    Args:
      config: A protobuf object that does not violate the conditions above.

    Returns:
      result_dict: |config| converted into a python dictionary.
    """
    result_dict = {}
    for field, value in config.ListFields():
        result_dict[field.name] = value
    return result_dict


# A map from a PreprocessingInputStep proto config field name to the preprocessing
# function that should be used. The PreprocessingInputStep proto should be parsable
# with _get_dict_from_proto.
PREPROCESSING_FUNCTION_MAP = {
    'swap_height_width_of_rboxes': preprocessor_input.swap_height_width_of_rboxes,
    'subtract_angle_of_rboxes': preprocessor_input.subtract_angle_of_rboxes,
    'normalize_tiff': preprocessor_input.normalize_tiff,
    'cutoff_max_gts': preprocessor_input.cutoff_max_gts,
}


def build(preprocessor_input_step_config):
    """Builds preprocessing input step based on the configuration.

    Args:
      preprocessor_input_step_config: PreprocessingInputStep configuration proto.

    Returns:
      function, argmap: A callable function and an argument map to call function with.

    Raises:
      ValueError: On invalid configuration.
    """
    step_type = preprocessor_input_step_config.WhichOneof('preprocess_input_step')

    if step_type in PREPROCESSING_FUNCTION_MAP:
        preprocessing_function = PREPROCESSING_FUNCTION_MAP[step_type]
        step_config = _get_step_config_from_proto(preprocessor_input_step_config, step_type)
        function_args = _get_dict_from_proto(step_config)
        return preprocessing_function, function_args

    raise ValueError('Unknown preprocessing input step.')
