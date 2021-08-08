import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers


def _compute_fans(shape):
    """Computes the number of input and output units for a weight shape.
    Args:
        shape: Integer shape tuple or TF tensor shape.
    Returns:
        A tuple of integer scalars (fan_in, fan_out).

    Extracted from tensorflow/keras/initializers. Available at
    https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/python/keras/initializers/initializers_v2.py
    """

    if len(shape) < 1:  # Just to avoid errors for constants.
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        # Assuming convolution kernels (2D, 3D, or more).
        # kernel shape: (..., input_depth, depth)
        receptive_field_size = 1
        for dim in shape[:-2]:
            receptive_field_size *= dim
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return int(fan_in), int(fan_out)


class HyperImaginary4DInitializer(initializers.Initializer):
    """
    Computes initialization based on quaternion variance.
    Options: he uniform, he normal, glorot uniform, glorot normal.
    References:
    [1] He, K., Zhang, X., Ren, S., and Sun, J. (2015b).  Delving deep into rectifiers: Surpassing human-level performance on imagenet classification.
    [2] Glorot, X. and Bengio, Y. (2010).  Understanding the difficulty of training deep feedforward neural networks.
    In Teh, Y. W. and Titterington, M., editors, Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics,
    volume 9 of Proceedings of Machine Learning Research, pages 249–256, Chia Laguna Resort, Sardinia, Italy. PMLR.
    """
    def __init__(self, criterion='he', distribution='uniform', seed=31337):
        self.criterion = criterion
        self.distribution = distribution
        self.seed = seed

    def __call__(self, shape, dtype):
        fan_in, fan_out = _compute_fans(shape)

        if self.criterion == 'he':
            std = 1. / np.sqrt(2 * fan_in)
        elif self.criterion == 'glorot':
            std = 1. / np.sqrt(2 * (fan_in + fan_out))
        else:
            raise ValueError("Chosen criterion was not identified.")

        if self.distribution == 'normal':
            return tf.random.normal(shape, mean=0, stddev=std, dtype=dtype, seed=self.seed)
        elif self.distribution == 'uniform':
            lim = std * np.sqrt(3)
            return tf.random.uniform(shape, minval=-lim, maxval=lim, dtype=dtype, seed=self.seed)
        else:
            raise ValueError("Chosen distribution was not identified")
