import tensorflow as tf

labels = tf.placeholder(tf.int64, [None])
y_ = tf.one_hot(labels, depth = 3, dtype = tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    teacher = [1, 2, 0] # 正解のあるカラム
    print(sess.run(y_, feed_dict = {labels: teacher}))
