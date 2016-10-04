import tensorflow as tf
import openblas
sess = tf.InteractiveSession()

a = tf.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
b = tf.convert_to_tensor([[5.0, 6.0], [7.0, 8.0]])
matmul = openblas.sgemm
c = matmul(a, b)
print(c.eval(), 'expected value: [[19., 22.], [43., 50.]]')
a = tf.convert_to_tensor([[1.0, 2.0]])
b = tf.convert_to_tensor([[5.0], [6.0]])
c = matmul(a, b)
print(c.eval(), 'expected value: 17.0')

x = tf.get_variable('x', [], tf.float32,
                    initializer=tf.constant_initializer(1.0))
y = tf.get_variable('y', [], tf.float32,
                    initializer=tf.constant_initializer(2.0))

sess.run(x.initializer)
sess.run(y.initializer)

a = tf.convert_to_tensor([[x, x]])
b = tf.convert_to_tensor([[y], [y]])
print('a')
print(a.eval())
print('b')
print(b.eval())
c = matmul(a, b)
# c = tf.matmul(a,b)
print('a*b')
print(c.eval(), 'expected value: 4.0')

c = matmul(b, a, transpose_a=True, transpose_b=True)
print('b^T*a^T')
print(c.eval(), 'expected value: 4.0')

gx, gy = tf.gradients(c[0, 0], [x, y])
print('dc/dx =', gx.eval(), 'expected value: 4.0')
print('dc/dy =', gy.eval(), 'expected value: 2.0')
