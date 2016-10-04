import tensorflow as tf

openblas_library = tf.load_op_library('./openblas/openblas.so')
sgemm_op = openblas_library.sgemm_op

@tf.RegisterGradient("SGEMMOp")
def _sgemm_grad(op, grad):
    """The gradients for `sgemm_op`.

    Args:
      op: The `sgemm_op` `Operation` that we are differentiating, which we can
          use to find the inputs and outputs of the original op.
      grad: Gradient with respect to the output of the `sgemm_op` op.

    Returns:
      Gradients with respect to the input of `sgemm_op`.

    NOTE: See `tensorflow/python/ops/math_grad.py`.
    """
    a, b = op.inputs
    transa = op.get_attr('transpose_a')
    transb = op.get_attr('transpose_b')
    return [
        sgemm_op(b, grad, transpose_a=True) if transb else
        sgemm_op(grad, b, transpose_b=True),
        sgemm_op(grad, a, transpose_b=True) if transa else
        sgemm_op(a, grad, transpose_a=True),
      ]

# This has some kind of C++ implementation as well, how do they interop?
@tf.RegisterShape("SGEMMOp")
def _sgemm_shape(op):
  """Shape function for the `sgemm_op` `Operation`.
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
