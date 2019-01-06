import tensorflow as tf


holder1 = tf.placeholder(tf.int32)
#holder2 = tf.placeholder(tf.int32)
#holder2 = tf.placeholder(tf.int32, shape = [3])
holder2 = tf.placeholder(tf.int32, shape = [None])
holder3 = tf.placeholder(tf.int32, shape = [None])

mul_op = holder1 * holder2

with tf.Session() as sess:
    result = sess.run(mul_op, feed_dict = { holder1: 2, holder2: [0, 1, 2, 3, 4] })
    print(result)

    result = sess.run(mul_op, feed_dict = { holder1: 5, holder2: [0, 10, 20], holder3: [1, 2, 3] })
    print(result)
