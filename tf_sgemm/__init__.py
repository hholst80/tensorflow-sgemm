import os
import tensorflow as tf

library_path = os.path.join(os.path.dirname(__file__), 'openblas.so')
openblas_library = tf.load_op_library(library_path)
sgemm = openblas_library.sgemm

@tf.RegisterGradient("SGEMM")
def _sgemm_grad(op, grad):
    """The gradients for `sgemm`.

    Args:
      op: The `sgemm` `Operation` that we are differentiating, which we can
          use to find the inputs and outputs of the original op.
      grad: Gradient with respect to the output of the `sgemm` op.

    Returns:
      Gradients with respect to the input of `sgemm`.

    NOTE: See `tensorflow/python/ops/math_grad.py`.
    """
    a, b = op.inputs
    transa = op.get_attr('transpose_a')
    transb = op.get_attr('transpose_b')
    return [
        sgemm(b, grad, transpose_a=True) if transb else
        sgemm(grad, b, transpose_b=True),
        sgemm(grad, a, transpose_b=True) if transa else
        sgemm(a, grad, transpose_a=True),
      ]

# This has some kind of C++ implementation as well, how do they interop?
@tf.RegisterShape("SGEMM")
def _sgemm_shape(op):
  """Shape function for the `sgemm` `Operation`.
  """
  a, b = op.inputs
  transa = op.get_attr('transpose_a')
  transb = op.get_attr('transpose_b')
  a_shape = a.get_shape().with_rank(2)
  b_shape = b.get_shape().with_rank(2)
  ma, na = a_shape.as_list()
  mb, nb = b_shape.as_list()
  shape = tf.TensorShape([ma if not transa else na, nb if not transb else mb])
  return [shape]
