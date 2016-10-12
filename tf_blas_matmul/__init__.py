import os
import tensorflow as tf

library_path = os.path.join(os.path.dirname(__file__), 'blas_matmul.so')
openblas_library = tf.load_op_library(library_path)
blas_matmul = openblas_library.blas_matmul

@tf.RegisterGradient("BLASMatmul")
def _blas_matmul_grad(op, grad):
    """The gradients for `blas_matmul`.

    Args:
      op: The `blas_matmul` `Operation` that we are differentiating, which we can
          use to find the inputs and outputs of the original op.
      grad: Gradient with respect to the output of the `blas_matmul` op.

    Returns:
      Gradients with respect to the input of `blas_matmul`.

    NOTE: See `tensorflow/python/ops/math_grad.py`.
    """
    a, b = op.inputs
    transa = op.get_attr('transpose_a')
    transb = op.get_attr('transpose_b')
    return [
        blas_matmul(b, grad, transpose_a=True) if transb else
        blas_matmul(grad, b, transpose_b=True),
        blas_matmul(grad, a, transpose_b=True) if transa else
        blas_matmul(a, grad, transpose_a=True),
      ]

# This has some kind of C++ implementation as well, how do they interop?
@tf.RegisterShape("BLASMatmul")
def _blas_matmul_shape(op):
  """Shape function for the `blas_matmul` `Operation`.
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
