import tensorflow.keras as kr
from tensorflow.keras import layers
from functools import partial
from TessConv2D import TessConv2D
from Hypercomplex4DBNActivation import Hypercomplex4DBNActivation


DefaultConvTess = partial(TessConv2D, kernel_size=(3, 3), strides=1, padding="SAME", kernel_regularizer=1e-3, use_bias=False)


class TessResidualUnit(layers.Layer):
    """
    Tessarine valued residual unit.
    References:
    [1] He, K., Zhang, X., Ren, S., and Sun, J. (2015).  Deep residual learning for image recog-nition.
    [2] He, K., Zhang, X., Ren, S., and Sun, J. (2016).  Identity mappings in deep residual net-works.
    """
    def __init__(self, filters, strides=1, activation="elu", activation_position='after', **kwargs):
        super().__init__(**kwargs)
        self.activation = kr.activations.get(activation)
        self.main_layers = [
            DefaultConvTess(filters, strides=strides),
            Hypercomplex4DBNActivation(activation=activation, activation_position=activation_position),
            DefaultConvTess(filters),
            Hypercomplex4DBNActivation(activation=activation, activation_position=activation_position)]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConvTess(filters, kernel_size=(1, 1), strides=strides),
                Hypercomplex4DBNActivation(activation=None, activation_position='no_activation')]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)