import tensorflow as tf

const = tf.constant(1)
holder = tf.placeholder(tf.int32)
add_op = const + holder

with tf.Session() as sess:
    result = sess.run(add_op, feed_dict = {holder: 5})
    print(result)

    result = sess.run(add_op, feed_dict = {holder: 10})
    print(result)
