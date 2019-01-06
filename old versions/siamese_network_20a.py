# -*- coding: utf-8 -*-
from PIL import Image
import glob
import numpy as np
from PIL import Image
import tensorflow as tf
import time
import traceback
from sklearn.utils import shuffle

X = {}
y = {}

train_img =[]
train_label = []
test_img = []
test_label = []
rep_img = []

nb_class_list = list(range(2))
nb_class = len(nb_class_list)
e = np.eye(nb_class)
IMG_W, IMG_H = 28, 28
IMG_SIZE = IMG_W * IMG_H

print("Image read started")
for i in nb_class_list:

    for filename in glob.glob('./image/train/' + str(i) + '/*.jpg'):
        img = Image.open(filename).resize( (IMG_W, IMG_H) ).convert('L') # original is 28x28. so no need
        img = ( np.asarray(img)/255 ).reshape(IMG_SIZE)
        rep_img.append(img)

    for filename in glob.glob('./image/train/' + str(i) + '/*.jpg'):
        img = Image.open(filename).resize( (IMG_W, IMG_H) ).convert('L') # original is 28x28. so no need
        img = ( np.asarray(img)/255 ).reshape(IMG_SIZE)
        train_img.append(img)
        train_label.append(e[i])

    for filename in glob.glob('./image/test/' + str(i) + '/*.jpg'):
        img = Image.open(filename).resize( (IMG_W, IMG_H) ).convert('L')
        img = ( np.asarray(img)/255 ).reshape(IMG_SIZE)
        test_img.append(img)
        test_label.append(e[i])
print(len(rep_img))
print("Image read finished")


# sessionの開始
sess = tf.InteractiveSession()

################################################################
##### Define variables #########################################
# Input type
x1 = tf.placeholder(dtype = tf.float32, shape = [None, IMG_SIZE])
x2 = tf.placeholder(dtype = tf.float32, shape = [None, IMG_SIZE])

# Output type
y1_ = tf.placeholder(dtype = tf.int32, shape = [None, nb_class])
y2_ = tf.placeholder(dtype = tf.int32, shape = [None, nb_class])

distance = tf.placeholder(dtype = tf.float32, shape = [None])
##### Define variables #########################################
################################################################


#######################################################################
##### Define conv and pooling           ###############################
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)
##### Define conv and pooling           ###############################
#######################################################################


###########################################################################
##### Define layers #######################################################
W_conv1 = weight_variable([4, 4, 1, 16]) # H, W, in_ch, out_ch
b_conv1 = bias_variable([16])

x1_image = tf.reshape(x1, [-1, 28, 28, 1])
x2_image = tf.reshape(x2, [-1, 28, 28, 1])
h1_conv1 = tf.nn.relu(conv2d(x1_image, W_conv1) + b_conv1) # padding = 'SAME'
h2_conv1 = tf.nn.relu(conv2d(x2_image, W_conv1) + b_conv1)
h1_pool1 = max_pool_2x2(h1_conv1) # stride = 2 => img size (14, 14)
h2_pool1 = max_pool_2x2(h2_conv1)

W_conv2 = weight_variable([5, 5, 16, 64])
b_conv2 = bias_variable([64])

h1_conv2 = tf.nn.relu(conv2d(h1_pool1, W_conv2) + b_conv2)
h2_conv2 = tf.nn.relu(conv2d(h2_pool1, W_conv2) + b_conv2)
h1_pool2 = max_pool_2x2(h1_conv2) # stride = 2 => img size (7, 7)
h2_pool2 = max_pool_2x2(h2_conv2)

W_conv3 = weight_variable([5, 5, 64, 32])
b_conv3 = bias_variable([32])

h1_conv3 = tf.nn.relu(conv2d(h1_pool2, W_conv3) + b_conv3)
h2_conv3 = tf.nn.relu(conv2d(h2_pool2, W_conv3) + b_conv3)
h1_pool3 = max_pool_2x2(h1_conv3) # stride = 2 => img size (4, 4)
h2_pool3 = max_pool_2x2(h2_conv3)

W_fc1 = weight_variable([4 * 4 * 32, 32]) # (4x4) * 32
b_fc1 = bias_variable([32])

h1_pool1_flat = tf.reshape(h1_pool3, [-1, 4 * 4 * 32])
h2_pool1_flat = tf.reshape(h2_pool3, [-1, 4 * 4 * 32])
h1_fc1 = tf.nn.relu( tf.matmul(h1_pool1_flat, W_fc1) + b_fc1 )
h2_fc1 = tf.nn.relu( tf.matmul(h2_pool1_flat, W_fc1) + b_fc1 ) # out_ch = 32

keep_prob = tf.placeholder(tf.float32)
h1_fc1_drop = tf.nn.dropout(h1_fc1, keep_prob) # テスト時にはドロップアウトは行わない「1.0」を設定
h2_fc1_drop = tf.nn.dropout(h2_fc1, keep_prob)

distance_l = tf.abs( tf.subtract(h2_fc1_drop, h1_fc1_drop) ) # h_fc1_drop(out_ch) = 32

WB_fc1 = weight_variable([32, 1]) # row, col

y_conv = tf.nn.sigmoid( tf.matmul(distance_l, WB_fc1) ) # probability # print(y_conv.shape)
##### Define layers #######################################################
###########################################################################


###########################################################################
##### 評価関数および最適化方法の設定 ###########################################
EPOCHS, BATCH_SIZE, Optrate = 1000, 1, 0.001

# y_conv: softmax after distance calc                                        axis=1 -> ある行の最大値があるカラム
sum_acc = tf.reduce_mean( tf.abs( tf.subtract( y_conv, tf.cast(tf.argmax(y1_, axis = 1), tf.float32) ) ) )

# <DISS> is teacher (real) probablity. log(Softmax output probability (deduced probability)
cross_entropy = -tf.reduce_sum( distance * tf.log(tf.cast(y_conv, dtype = tf.float32) + (1e-7)) ) #+\
                 # sum_acc +\
                 # tf.reduce_sum(tf.abs(WB_fc1))

train_step = tf.train.AdamOptimizer(Optrate).minimize(cross_entropy)

correct_prediction = tf.equal( tf.argmax(y_conv, 1), tf.argmax(y1_, 1) )
##### 評価関数および最適化方法の設定 ###########################################
###########################################################################


X['train'] = train_img # ["0", "1"]
y['train'] = train_label # [[1, 0], [0, 1]]
X['test']  = test_img # print(np.asarray(test_img).shape) (200, 784)
y['test']  = test_label# print(np.asarray(test_label).shape) (200, 2)


sess.run(tf.global_variables_initializer())
start = time.time()

for i in range(EPOCHS):
    print('EPOCHS = ', i)
    X1_train, y1_train = shuffle(X['train'], y['train']) # image and label of "0" or "1"
    X2_train, y2_train = shuffle(X['train'], y['train']) # X_train = (2, 784), y_train = (2, 2), (r,c)

    for OFFSET in range(0, len(X['train']), BATCH_SIZE): # len(X['train']) = 2, BATCH_SIZE = 1

        batch_x1, batch_y1 = X1_train[OFFSET: (OFFSET + BATCH_SIZE)], y1_train[OFFSET: (OFFSET + BATCH_SIZE)]
        batch_x2, batch_y2 = X2_train[OFFSET: (OFFSET + BATCH_SIZE)], y2_train[OFFSET: (OFFSET + BATCH_SIZE)]

        simi = np.asarray(batch_y1) * np.asarray(batch_y2)
        distance_labels = [(1 - np.sum(x)) for x in simi]
        # optimizer
        train_step.run(feed_dict = {x1: batch_x1, y1_: batch_y1,
                                    x2: batch_x2, y2_: batch_y2,
                                    distance: distance_labels, keep_prob: 1, })

Elapsed_time = time.time() - start
print('Time : %f' % Elapsed_time)


# テストデータで今回のモデルを最終評価　指標は精度
test_len = len(y['test'])

dummy_0 = np.asarray([1.0, 0.0] * test_len).reshape(test_len, 2)
dummy_1 = np.asarray([0.0, 1.0] * test_len).reshape(test_len, 2)

img_0 = np.asarray(list(rep_img[0]) * test_len).reshape(test_len, 784)
img_1 = np.asarray(list(rep_img[1]) * test_len).reshape(test_len, 784)

distance_labels_0 = [     np.sum(x)  for x in dummy_0]
distance_labels_1 = [(1 - np.sum(x)) for x in dummy_1]

eval_acc = sess.run(y_conv, feed_dict = {x1: X['test'], y1_: y['test'],
                                         x2: img_0,     y2_: dummy_0,
                                         distance: distance_labels_0, keep_prob:1,})
eval_1acc = sess.run(y_conv, feed_dict = {x1: X['test'], y1_: y['test'],
                                          x2: img_1,     y2_: dummy_1,
                                          distance: distance_labels_1, keep_prob: 1,})# print(eval_acc)# print(eval_1acc)

matome = np.concatenate([eval_acc, eval_1acc], axis = 1)
predict_result = np.argmin(matome, axis = 1) # print(predict_result.shape) # (200, )
# print(predict_result)

y_right = np.argmax(np.asarray(y['test']), axis = 1) # print(y_right.shape) # (200, )
# print(y_right)

acc_result = np.equal(y_right, predict_result)
tmp = np.sum(acc_result)/len(acc_result)

print('Final: TestAccuracy: %f' % tmp)
