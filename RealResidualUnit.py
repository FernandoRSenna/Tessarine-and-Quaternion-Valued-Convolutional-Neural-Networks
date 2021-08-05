import tensorflow.keras as keras
from functools import partial


DefaultConv2D = partial(keras.layers.Conv2D,
                        kernel_size=3,
                        strides=1,
                        padding="SAME",
                        use_bias=False,
                        kernel_initializer='he_uniform',
                        kernel_regularizer=keras.regularizers.l2(1e-4))


class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, conv_first=True, activation="elu", include_bn=True, **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.conv_first = conv_first
        self.include_bn = include_bn

        self.main_layers = []
        if self.conv_first:
            self.main_layers.append(DefaultConv2D(filters, strides=strides))
            if self.include_bn:
                self.main_layers.append(keras.layers.BatchNormalization())
            self.main_layers.append(self.activation)
            self.main_layers.append(DefaultConv2D(filters))
            if self.include_bn:
                self.main_layers.append(keras.layers.BatchNormalization())
        else:
            if self.include_bn:
                self.main_layers.append(keras.layers.BatchNormalization())
            self.main_layers.append(self.activation)
            self.main_layers.append(DefaultConv2D(filters, strides=strides))
            if self.include_bn:
                self.main_layers.append(keras.layers.BatchNormalization())
            self.main_layers.append(self.activation)
            self.main_layers.append(DefaultConv2D(filters))

        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [DefaultConv2D(filters, kernel_size=1, strides=strides)]
            if self.include_bn:
                self.skip_layers.append(keras.layers.BatchNormalization())

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        if self.conv_first:
            return self.activation(Z + skip_Z)
        else:
            return Z + skip_Z
