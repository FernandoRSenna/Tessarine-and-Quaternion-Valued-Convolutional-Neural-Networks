import tensorflow.keras as kr
from tensorflow.keras import layers
from functools import partial
from HyperImaginary4DBNActivation import HyperImaginary4DBNActivation
from QuatConv2D import QuatConv2D


DefaultConvQuat = partial(QuatConv2D, kernel_size=(3, 3), strides=1, padding="SAME", kernel_regularizer=1e-3, use_bias=False)


class QuatResidualUnit(layers.Layer):
    """
    Quaternion valued residual unit.
    References:
    [1] He, K., Zhang, X., Ren, S., and Sun, J. (2015).  Deep residual learning for image recog-nition.
    [2] He, K., Zhang, X., Ren, S., and Sun, J. (2016).  Identity mappings in deep residual net-works.
    """
    def __init__(self, filters, strides=1, activation="elu", activation_position="after", **kwargs):
        super().__init__(**kwargs)
        self.activation = kr.activations.get(activation)
        self.main_layers = [
            DefaultConvQuat(filters, strides=strides),
            HyperImaginary4DBNActivation(activation=activation, activation_position=activation_position),
            DefaultConvQuat(filters),
            HyperImaginary4DBNActivation(activation=activation, activation_position=activation_position)]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConvQuat(filters, kernel_size=(1, 1), strides=strides),
                HyperImaginary4DBNActivation(activation_position="no_activation")]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)
