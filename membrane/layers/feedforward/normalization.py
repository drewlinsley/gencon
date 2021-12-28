from tensorflow.compat import v1 as tf
from tensorflow.python.layers import normalization as normalization_layers
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.training import moving_averages
tf.disable_v2_behavior()


DATA_FORMAT_NCHW = 'NCHW'
DATA_FORMAT_NHWC = 'NHWC'
DATA_FORMAT_NCDHW = 'NCDHW'
DATA_FORMAT_NDHWC = 'NDHWC'
MODEL_VARIABLES = '_model_variables_'


def batch(
        bottom,
        name,
        scale=True,
        center=True,
        fused=True,
        renorm=False,
        data_format='NHWC',
        reuse=False,
        renorm_decay=0.99,
        decay=0.999,
        training=True):
    if data_format == 'NHWC' or data_format == 'channels_last':
        axis = -1
    elif data_format == 'NCHW' or data_format == 'channels_first':
        axis = 1
    else:
        raise NotImplementedError(data_format)
    return tf.layers.batch_normalization(
        inputs=bottom,
        name=name,
        scale=scale,
        center=center,
        fused=fused,
        renorm=renorm,
        reuse=reuse,
        axis=axis,
        momentum=decay,
        renorm_momentum=renorm_decay,
        training=training)

