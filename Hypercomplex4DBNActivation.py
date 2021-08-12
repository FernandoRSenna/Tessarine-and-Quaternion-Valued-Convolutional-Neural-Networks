import tensorflow as tf
import tensorflow.keras as kr
from tensorflow.keras import layers, activations, initializers, regularizers


def diag_init(shape, dtype=None):
    return tf.ones(shape) / 2.


class Hypercomplex4DBNActivation(layers.Layer):
    """
    Batch Normalization for tessarines and quaternions.
    Based on matrix whitening. Decorrelates each component of tessarine/quaternion.
    Includes activation: can be placed before, after or in the middle of BN.
    References:
    [1] Ioffe, S. and Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift.
    [2] Kessy, A., Lewin, A., and Strimmer, K. (2018). Optimal whitening and decorrelation. The American Statistician, 72(4):309–314.
    [3] Trabelsi, C., Bilaniuk, O., Serdyuk, D., Subramanian, S., Santos, J. F., Mehri, S., Ros-tamzadeh, N., Bengio, Y., and Pal, C. J. (2017). Deep complex networks.
    [4] Gaudet, C. and Maida, A. (2017). Deep quaternion networks.
    """
    def __init__(self,
                 center=True,
                 scale=True,
                 momentum=0.9,
                 beta_init='zeros',
                 gam_diag_init='diag_init',
                 gam_off_init='zeros',
                 mov_mean_init='zeros',
                 mov_var_init='diag_init',
                 mov_cov_init='zeros',
                 beta_reg=None,
                 gam_diag_reg=None,
                 gam_off_reg=None,
                 activation="elu",
                 activation_position="after",
                 epsilon=1e-6,
                 **kwargs):
        super(Hypercomplex4DBNActivation, self).__init__(**kwargs)
        self.center = center
        self.scale = scale
        self.momentum = momentum
        self.beta_init = initializers.get(beta_init)

        if gam_diag_init == 'diag_init':
            self.gam_diag_init = diag_init
        else:
            self.gam_diag_init = initializers.get(gam_diag_init)

        self.gam_off_init = initializers.get(gam_off_init)
        self.mov_mean_init = initializers.get(mov_mean_init)

        if mov_var_init == 'diag_init':
            self.mov_var_init = diag_init
        else:
            self.mov_var_init = initializers.get(mov_var_init)

        self.mov_cov_init = initializers.get(mov_cov_init)
        self.beta_reg = regularizers.get(beta_reg)
        self.gam_diag_reg = regularizers.get(gam_diag_reg)
        self.gam_off_reg = regularizers.get(gam_off_reg)
        self.activation = activations.get(activation)
        self.activation_position = activation_position
        self.epsilon = epsilon

    def build(self, input_shape):
        input_dim = input_shape[-1] // 4
        vars_shape = [input_dim, 1]
        gamma_shape = (input_dim,)

        if self.scale:
            self.mov_Vrr = self.add_weight(shape=vars_shape,
                                           initializer=self.mov_var_init,
                                           trainable=False,
                                           name="mov_Vrr")
            self.mov_Vri = self.add_weight(shape=vars_shape,
                                           initializer=self.mov_cov_init,
                                           trainable=False,
                                           name="mov_Vri")
            self.mov_Vrj = self.add_weight(shape=vars_shape,
                                           initializer=self.mov_cov_init,
                                           trainable=False,
                                           name="mov_Vrj")
            self.mov_Vrk = self.add_weight(shape=vars_shape,
                                           initializer=self.mov_cov_init,
                                           trainable=False,
                                           name="mov_Vrk")
            self.mov_Vii = self.add_weight(shape=vars_shape,
                                           initializer=self.mov_var_init,
                                           trainable=False,
                                           name="mov_Vii")
            self.mov_Vij = self.add_weight(shape=vars_shape,
                                           initializer=self.mov_cov_init,
                                           trainable=False,
                                           name="mov_Vij")
            self.mov_Vik = self.add_weight(shape=vars_shape,
                                           initializer=self.mov_cov_init,
                                           trainable=False,
                                           name="mov_Vik")
            self.mov_Vjj = self.add_weight(shape=vars_shape,
                                           initializer=self.mov_var_init,
                                           trainable=False,
                                           name="mov_Vjj")
            self.mov_Vjk = self.add_weight(shape=vars_shape,
                                           initializer=self.mov_cov_init,
                                           trainable=False,
                                           name="mov_Vjk")
            self.mov_Vkk = self.add_weight(shape=vars_shape,
                                           initializer=self.mov_var_init,
                                           trainable=False,
                                           name="mov_Vkk")

            self.gam_rr = self.add_weight(shape=gamma_shape,
                                          initializer=self.gam_diag_init,
                                          regularizer=self.gam_diag_reg,
                                          name="gam_rr")
            self.gam_ri = self.add_weight(shape=gamma_shape,
                                          initializer=self.gam_off_init,
                                          regularizer=self.gam_off_reg,
                                          name="gam_ri")
            self.gam_rj = self.add_weight(shape=gamma_shape,
                                          initializer=self.gam_off_init,
                                          regularizer=self.gam_off_reg,
                                          name="gam_rj")
            self.gam_rk = self.add_weight(shape=gamma_shape,
                                          initializer=self.gam_off_init,
                                          regularizer=self.gam_off_reg,
                                          name="gam_rk")
            self.gam_ii = self.add_weight(shape=gamma_shape,
                                          initializer=self.gam_diag_init,
                                          regularizer=self.gam_diag_reg,
                                          name="gam_ii")
            self.gam_ij = self.add_weight(shape=gamma_shape,
                                          initializer=self.gam_off_init,
                                          regularizer=self.gam_off_reg,
                                          name="gam_ij")
            self.gam_ik = self.add_weight(shape=gamma_shape,
                                          initializer=self.gam_off_init,
                                          regularizer=self.gam_off_reg,
                                          name="gam_ik")
            self.gam_jj = self.add_weight(shape=gamma_shape,
                                          initializer=self.gam_diag_init,
                                          regularizer=self.gam_diag_reg,
                                          name="gam_jj")
            self.gam_jk = self.add_weight(shape=gamma_shape,
                                          initializer=self.gam_off_init,
                                          regularizer=self.gam_off_reg,
                                          name="gam_jk")
            self.gam_kk = self.add_weight(shape=gamma_shape,
                                          initializer=self.gam_diag_init,
                                          regularizer=self.gam_diag_reg,
                                          name="gam_kk")
        else:
            self.mov_Vrr = None
            self.mov_Vri = None
            self.mov_Vrj = None
            self.mov_Vrk = None
            self.mov_Vii = None
            self.mov_Vij = None
            self.mov_Vik = None
            self.mov_Vjj = None
            self.mov_Vjk = None
            self.mov_Vkk = None
            self.gam_rr = None
            self.gam_ri = None
            self.gam_rj = None
            self.gam_rk = None
            self.gam_ii = None
            self.gam_ij = None
            self.gam_ik = None
            self.gam_jj = None
            self.gam_jk = None
            self.gam_kk = None

        if self.center:
            self.beta = self.add_weight(shape=(1, 1, 1, input_shape[-1]),
                                        initializer=self.beta_init,
                                        regularizer=self.beta_reg,
                                        name="beta")
            self.mov_mean = self.add_weight(shape=(1, 1, 1, input_shape[-1]),
                                            initializer=self.mov_mean_init,
                                            trainable=False,
                                            name="mov_mean")
        else:
            self.beta = None
            self.mov_mean = None

    def _compute_variances(self, centered_r, centered_i, centered_j, centered_k, input_dim):
        Vrr = kr.backend.mean(
            centered_r ** 2,
            axis=[0, 1, 2]
        ) + self.epsilon

        Vri = kr.backend.mean(
            centered_r * centered_i,
            axis=[0, 1, 2]
        )

        Vrj = kr.backend.mean(
            centered_r * centered_j,
            axis=[0, 1, 2]
        )

        Vrk = kr.backend.mean(
            centered_r * centered_k,
            axis=[0, 1, 2]
        )

        Vii = kr.backend.mean(
            centered_i ** 2,
            axis=[0, 1, 2]
        ) + self.epsilon

        Vij = kr.backend.mean(
            centered_i * centered_j,
            axis=[0, 1, 2]
        )

        Vik = kr.backend.mean(
            centered_i * centered_k,
            axis=[0, 1, 2]
        )

        Vjj = kr.backend.mean(
            centered_j ** 2,
            axis=[0, 1, 2]
        ) + self.epsilon

        Vjk = kr.backend.mean(
            centered_j * centered_k,
            axis=[0, 1, 2]
        )

        Vkk = kr.backend.mean(
            centered_k ** 2,
            axis=[0, 1, 2]
        ) + self.epsilon

        pars_shape = [input_dim, 1]
        Vrr = tf.reshape(Vrr, pars_shape)
        Vri = tf.reshape(Vri, pars_shape)
        Vrj = tf.reshape(Vrj, pars_shape)
        Vrk = tf.reshape(Vrk, pars_shape)
        Vii = tf.reshape(Vii, pars_shape)
        Vij = tf.reshape(Vij, pars_shape)
        Vik = tf.reshape(Vik, pars_shape)
        Vjj = tf.reshape(Vjj, pars_shape)
        Vjk = tf.reshape(Vjk, pars_shape)
        Vkk = tf.reshape(Vkk, pars_shape)

        return Vrr, Vri, Vrj, Vrk, Vii, Vij, Vik, Vjj, Vjk, Vkk

    def _moving_exponential_update(self, var, value):
        decay = 1 - self.momentum
        var.assign_sub(var * decay)
        var.assign_add(value * decay)

    def _update_moving_parameters(self, mean, Vrr, Vri, Vrj, Vrk, Vii, Vij, Vik, Vjj, Vjk, Vkk):
        if self.center:
            self._moving_exponential_update(self.mov_mean, mean)

        if self.scale:
            self._moving_exponential_update(self.mov_Vrr, Vrr)
            self._moving_exponential_update(self.mov_Vri, Vri)
            self._moving_exponential_update(self.mov_Vrj, Vrj)
            self._moving_exponential_update(self.mov_Vrk, Vrk)
            self._moving_exponential_update(self.mov_Vii, Vii)
            self._moving_exponential_update(self.mov_Vij, Vij)
            self._moving_exponential_update(self.mov_Vik, Vik)
            self._moving_exponential_update(self.mov_Vjj, Vjj)
            self._moving_exponential_update(self.mov_Vjk, Vjk)
            self._moving_exponential_update(self.mov_Vkk, Vkk)

    def call(self, inputs, training=None):
        if (not self.center) or (not self.scale):
            raise ValueError("Batch Normalization should scale or center.")

        input_shape = kr.backend.int_shape(inputs)
        input_dim = input_shape[-1] // 4

        # Activation before
        if self.activation_position == "before":
            output = self.activation(inputs)
        else:
            output = inputs

        if training in {0, False}:
            mean = self.mov_mean
            centered = output - mean

            if self.scale:
                centered_r = centered[:, :, :, :input_dim]
                centered_i = centered[:, :, :, input_dim:input_dim * 2]
                centered_j = centered[:, :, :, input_dim * 2:input_dim * 3]
                centered_k = centered[:, :, :, input_dim * 3:]

                Vrr = self.mov_Vrr
                Vri = self.mov_Vri
                Vrj = self.mov_Vrj
                Vrk = self.mov_Vrk
                Vii = self.mov_Vii
                Vij = self.mov_Vij
                Vik = self.mov_Vik
                Vjj = self.mov_Vjj
                Vjk = self.mov_Vjk
                Vkk = self.mov_Vkk
        else:
            # mean and centering
            mean = kr.backend.mean(output, axis=[0, 1, 2])
            mean = kr.backend.reshape(mean, [1, 1, 1, input_dim * 4])
            centered = output - mean

            if self.scale:
                centered_r = centered[:, :, :, :input_dim]
                centered_i = centered[:, :, :, input_dim:input_dim * 2]
                centered_j = centered[:, :, :, input_dim * 2:input_dim * 3]
                centered_k = centered[:, :, :, input_dim * 3:]

                Vrr, Vri, Vrj, Vrk, Vii, Vij, Vik, Vjj, Vjk, Vkk = self._compute_variances(centered_r, centered_i,
                                                                                           centered_j, centered_k,
                                                                                           input_dim)
            else:
                Vrr, Vri, Vrj, Vrk, Vii, Vij, Vik, Vjj, Vjk, Vkk = [None for i in range(10)]

            self._update_moving_parameters(mean, Vrr, Vri, Vrj, Vrk, Vii, Vij, Vik, Vjj, Vjk, Vkk)

        if self.scale:
            var_reshape = [input_dim, 1, 4]
            # covariance matrix
            V = tf.concat([[tf.reshape(tf.concat([Vrr, Vri, Vrj, Vrk], axis=1), var_reshape)],
                           [tf.reshape(tf.concat([Vri, Vii, Vij, Vik], axis=1), var_reshape)],
                           [tf.reshape(tf.concat([Vrj, Vij, Vjj, Vjk], axis=1), var_reshape)],
                           [tf.reshape(tf.concat([Vrk, Vik, Vjk, Vkk], axis=1), var_reshape)]], axis=2)

            # Whitening
            R = tf.reshape(tf.linalg.cholesky(V), [input_dim, 4, 4])
            W = tf.linalg.inv(tf.transpose(R, perm=[0, 2, 1]))

            Wrr = W[:, 0, 0]
            Wri = W[:, 0, 1]
            Wrj = W[:, 0, 2]
            Wrk = W[:, 0, 3]
            Wii = W[:, 1, 1]
            Wij = W[:, 1, 2]
            Wik = W[:, 1, 3]
            Wjj = W[:, 2, 2]
            Wjk = W[:, 2, 3]
            Wkk = W[:, 3, 3]

            output_r = centered_r * Wrr
            output_i = centered_r * Wri + centered_i * Wii
            output_j = centered_r * Wrj + centered_i * Wij + centered_j * Wjj
            output_k = centered_r * Wrk + centered_i * Wik + centered_j * Wjk + centered_k * Wkk

            if self.activation_position == "middle":
                output_r = self.activation(output_r)
                output_i = self.activation(output_i)
                output_j = self.activation(output_j)
                output_k = self.activation(output_k)

            out_r = output_r * self.gam_rr
            out_i = output_r * self.gam_ri + output_i * self.gam_ii
            out_j = output_r * self.gam_rj + output_i * self.gam_ij + output_j * self.gam_jj
            out_k = output_r * self.gam_rk + output_i * self.gam_ik + output_j * self.gam_jk + output_k * self.gam_kk
            output = tf.concat([out_r, out_i, out_j, out_k], axis=-1)
        else:
            output = centered
            if self.activation_position == "middle":
                output = self.activation(output)

        if self.center:
            output = output + self.beta

        if self.activation_position == "after":
            output = self.activation(output)

        return output
