"""A factorized noisy linear layer."""
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.layers.ops import core as core_ops


class NoisyDense(Dense):
  """A noisy dense layer."""

  def __init__(self, units,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               initial_sigma=0.5,
               **kwargs):
    """Initialize variables."""
    super().__init__(
      units, activation=activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      kernel_constraint=kernel_constraint,
      bias_constraint=bias_constraint,
      **kwargs
    )
    self.initial_sigma = initial_sigma

  def build(self, input_shape):
    """Build the layer."""
    super().build(input_shape)
    self.q, self.p = self.kernel.shape
    self.sigma = self.add_weight(
      'sigma',
      shape=[self.p + self.q],
      initializer=initializers.constant(self.initial_sigma),
      dtype=self.dtype,
      trainable=True
    )

  def call(self, inputs, training=None):
    """Call the layer."""
    def noised():
      kernel_sigma = tf.tensordot(self.sigma[self.p:], self.sigma[:self.p], axes=0)
      kernel = self.kernel + kernel_sigma * K.random_normal(
          shape=(self.q, self.p),
          mean=0.,
          stddev=1.,
          dtype=inputs.dtype
      )
      bias = self.bias
      if self.bias is not None:
        bias = bias + self.sigma[:self.p] * K.random_normal(
          shape=(self.p,),
          mean=0.,
          stddev=1.,
          dtype=inputs.dtype
      )
      return core_ops.dense(
        inputs,
        kernel,
        bias,
        self.activation,
        dtype=self._compute_dtype_object
      )

    call = super().call
    def normal():
      return call(inputs)

    return K.in_train_phase(noised, normal, training=training)

  def get_config(self):
    """Get the config."""
    config = super().get_config()
    config.update({
        'initial_sigma':
            self.initial_sigma
    })
    return config
