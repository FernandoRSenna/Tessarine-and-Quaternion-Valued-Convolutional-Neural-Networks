import tensorflow as tf
import tensorflow.keras as kr
from tensorflow.keras import layers, activations
from tensorflow.python.framework import tensor_shape
from Hypercomplex4DInitializer import Hypercomplex4DInitializer


class QuatConv2D(layers.Layer):
    """
    Quaternion valued 2D convolution layer.
    References:
    [1] Trabelsi, C., Bilaniuk, O., Serdyuk, D., Subramanian, S., Santos, J. F., Mehri, S., Rostamzadeh, N., Bengio, Y., and Pal, C. J. (2017). Deep complex networks.
    [2] Gaudet, C. and Maida, A. (2017). Deep quaternion networks.
    """
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='SAME',
                 use_bias=False,
                 activation=None,
                 initializer=Hypercomplex4DInitializer(),
                 data_format=None,
                 kernel_regularizer=1e-4):
        super(QuatConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.activation = activations.get(activation)
        self.initializer = initializer
        self.data_format = data_format
        self.kernel_regularizer = kernel_regularizer

    def _get_channel_axis(self):
        if self.data_format == 'channels_first':
            raise ValueError('QuatConv2d is designed only for channels_last. '
                             'The input has been changed to channels last!')
        else:
            return -1

    def _get_input_channel(self, input_shape):
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        return int(input_shape[channel_axis])

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        if input_channel % 4 != 0:
            raise ValueError('The number of input channels must be divisible by 4.')

        input_dim = input_channel // 4
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.f0 = self.add_weight(
            name='real_kernel',
            shape=kernel_shape,
            initializer=self.initializer,
            trainable=True,
            regularizer=kr.regularizers.l2(self.kernel_regularizer)
        )
        self.f1 = self.add_weight(
            name='imag_i_kernel',
            shape=kernel_shape,
            initializer=self.initializer,
            trainable=True,
            regularizer=kr.regularizers.l2(self.kernel_regularizer)
        )
        self.f2 = self.add_weight(
            name='imag_j_kernel',
            shape=kernel_shape,
            initializer=self.initializer,
            trainable=True,
            regularizer=kr.regularizers.l2(self.kernel_regularizer)
        )
        self.f3 = self.add_weight(
            name='imag_k_kernel',
            shape=kernel_shape,
            initializer=self.initializer,
            trainable=True,
            regularizer=kr.regularizers.l2(self.kernel_regularizer)
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(4 * self.filters,),
                initializer="zeros",
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None

    def call(self, inputs):
        # Filter multiplied from the right!
        F_r = tf.concat([self.f0, -self.f1, -self.f2, -self.f3], axis=2)
        F_i = tf.concat([self.f1, self.f0, self.f3, -self.f2], axis=2)
        F_j = tf.concat([self.f2, -self.f3, self.f0, self.f1], axis=2)
        F_k = tf.concat([self.f3, self.f2, -self.f1, self.f0], axis=2)

        y_r = tf.nn.conv2d(inputs, F_r, strides=self.strides, padding=self.padding)
        y_i = tf.nn.conv2d(inputs, F_i, strides=self.strides, padding=self.padding)
        y_j = tf.nn.conv2d(inputs, F_j, strides=self.strides, padding=self.padding)
        y_k = tf.nn.conv2d(inputs, F_k, strides=self.strides, padding=self.padding)

        outputs = tf.concat([y_r, y_i, y_j, y_k], axis=3)

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs
