import tensorflow as tf
import numpy as np

Y = np.array([
             [0.1, 0.2, 0.3, 0.4],
             [0.0, 0.8, 0.2, 0.0],
             [0.0, 0.4, 0.5, 0.1]
             ])

Y_ = np.array([
              [0.0, 0.0, 1.0, 0.0],
              [0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0]
              ])

sess = tf.Session()

# tf.argmax()から。 第2パラメーターに1をセットすると、行ごとに最大となる列を返す。
print('Y = \n', Y)
print('Y_ = \n', Y_)
print('tf.argmax(Y) : ', sess.run(tf.argmax(Y, axis = 1)))
print('tf.argmax(Y_) : ', sess.run(tf.argmax(Y_, axis = 1)))
#print( sess.run(tf.argmax(Y, axis = 0)), '\n' )
#print( sess.run(tf.argmax(Y_, axis = 0)), '\n' )

eq = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
print('equal? :', sess.run(eq))

print('cast to float : ', sess.run(tf.cast(eq, tf.float32)))

print('average : ', sess.run(tf.reduce_mean(tf.cast(eq, tf.float32))))
