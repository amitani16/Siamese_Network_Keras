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
train_label=[]
test_img=[]
test_label=[]

# データの読み込み
for filename in glob.glob('./image/train/0/*.jpg'):
    img = Image.open(filename)
    img = img.resize((28, 28)).convert('L')
    img = np.asarray(img)/255
    img = img.reshape(784)
    rep_0_img = img
    train_img.append(img)
    train_label.append([1.0, 0.0])
#print(list(rep_0_img))

for filename in glob.glob('./image/train/1/*.jpg'):
    img = Image.open(filename)
    img = img.resize((28, 28)).convert('L')
    img = np.asarray(img)
    img = img.reshape(784)/255
    rep_1_img = img
    train_img.append(img)
    train_label.append([0.0, 1.0])

for filename in glob.glob('./image/test/0/*.jpg'):
    img = Image.open(filename)
    img = img.resize((28, 28)).convert('L')
    img = np.asarray(img)
    img = img.reshape(784)/255
    test_img.append(img)
    test_label.append([1.0, 0.0])
#print(len(test_img))


for filename in glob.glob('./image/test/1/*.jpg'):
    img = Image.open(filename)
    img = img.resize((28, 28)).convert('L')
    img = np.asarray(img)
    img = img.reshape(784)/255
    test_img.append(img)
    test_label.append([0.0, 1.0])


print("Image_read_finished")

# sessionの開始
sess = tf.InteractiveSession()

################################################################
##### Define variables #########################################
# 入力層の型 linealized image
x  = tf.placeholder(dtype = tf.float32, shape = [None, 28 * 28])
xL = tf.placeholder(dtype = tf.float32, shape = [None, 28 * 28])

# 出力層の型 two values "0" or "1"
y_  = tf.placeholder(dtype = tf.int32, shape = [None, 2])
yL_ = tf.placeholder(dtype = tf.int32, shape = [None, 2])

# 距離vectorパラメータを設定
diss = tf.placeholder(dtype = tf.float32, shape = [None])
##### Define variables #########################################
################################################################


#######################################################################
##### 畳み込みとプーリングの関数を設定しておく ###############################
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
##### 畳み込みとプーリングの関数を設定しておく ###############################
#######################################################################


###########################################################################
##### Define layers #######################################################
# 第一層の畳み込み
W_conv1 = weight_variable([4, 4, 1, 16]) # H, W, in_ch, out_ch
b_conv1 = bias_variable([16])
x_image  = tf.reshape(x,  [-1, 28, 28, 1])
xL_image = tf.reshape(xL, [-1, 28, 28, 1]) # W_conv1 and b_conv1 are shared
h_conv1  = tf.nn.relu(conv2d(x_image,  W_conv1) + b_conv1) # padding = 'SAME'
hL_conv1 = tf.nn.relu(conv2d(xL_image, W_conv1) + b_conv1)
h_pool1  = max_pool_2x2(h_conv1) # padding = 'SAME'
hL_pool1 = max_pool_2x2(hL_conv1)

# 第二層の畳み込み
W_conv2 = weight_variable([5, 5, 16, 64])
b_conv2 = bias_variable([64])
h_conv2  = tf.nn.relu(conv2d(h_pool1,  W_conv2) + b_conv2)
hL_conv2 = tf.nn.relu(conv2d(hL_pool1, W_conv2) + b_conv2)
h_pool2  = max_pool_2x2(h_conv2) # padding = 'SAME'
hL_pool2 = max_pool_2x2(hL_conv2)

# 第三層の畳み込み
W_conv3 = weight_variable([5, 5, 64, 32])
b_conv3 = bias_variable([32])
h_conv3  = tf.nn.relu(conv2d(h_pool2,  W_conv3) + b_conv3)
hL_conv3 = tf.nn.relu(conv2d(hL_pool2, W_conv3) + b_conv3)
h_pool3  = max_pool_2x2(h_conv3) # padding = 'SAME'
hL_pool3 = max_pool_2x2(hL_conv3)

# 全結合層の設定
W_fc1 = weight_variable([512, 32]) # (4x4) * 32, max_pool_2x2 makes image size half
b_fc1 = bias_variable([32])
h_pool1_flat  = tf.reshape(h_pool3,  [-1, 512])
hL_pool1_flat = tf.reshape(hL_pool3, [-1, 512]) # (4x4) * 32 (out_ch) = 512, max_pool_2x2 makes image size half
h_fc1  = tf.nn.relu(tf.matmul(h_pool1_flat,  W_fc1) + b_fc1)
hL_fc1 = tf.nn.relu(tf.matmul(hL_pool1_flat, W_fc1) + b_fc1) # out_ch = 32

# Dropoutの設定
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop  = tf.nn.dropout(h_fc1,  keep_prob) # テスト時にはドロップアウトは行わない「1.0」を設定
hL_fc1_drop = tf.nn.dropout(hL_fc1, keep_prob)

# Distance
diss_l = tf.abs(tf.subtract(hL_fc1_drop, h_fc1_drop)) # h_fc1_drop(out_ch) = 32
#print(diss_l.shape)
WB_fc1 = weight_variable([32, 1]) # row, col
y_conv = tf.nn.sigmoid(tf.matmul(diss_l, WB_fc1)) # probability
#print(y_conv.shape)
##### Define layers #######################################################
###########################################################################


###########################################################################
##### 評価関数および最適化方法の設定 ###########################################
EPOCHS, BATCH_SIZE, Optrate = 500, 1, 0.001

# y_conv: softmax after distance calc                                        axis=1 -> ある行の最大値があるカラム
sum_acc = tf.reduce_mean( tf.abs( tf.subtract( y_conv, tf.cast(tf.argmax(y_, axis = 1), tf.float32) ) ) )

# <DISS> is teacher (real) probablity. log(Softmax output probability (deduced probability)
cross_entropy = -tf.reduce_sum( diss * tf.log(tf.cast(y_conv, dtype = tf.float32) + (1e-7)) ) +\
                 sum_acc +\
                 tf.reduce_sum(tf.abs(WB_fc1))

train_step = tf.train.AdamOptimizer(Optrate).minimize(cross_entropy)

correct_prediction = tf.equal( tf.argmax(y_conv, 1), tf.argmax(y_, 1) )
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
##### 評価関数および最適化方法の設定 ###########################################
###########################################################################


X['train'] = train_img # ["0", "1"]
y['train'] = train_label # [[1, 0], [0, 1]]
X['test']  = test_img # print(np.asarray(test_img).shape) (200, 784)
y['test']  = test_label# print(np.asarray(test_label).shape) (200, 2)

# 0かどうかの判定
dummy_0 = np.asarray([1.0, 0.0] * 2).reshape(2, 2) # [[1, 0], [1, 0]]
distance_labels_0 = [np.sum(x) for x in dummy_0] # sum in [1, 0] => [1, 1] ?????????

# 1かどうかの判定
dummy_1 = np.asarray([0.0, 1.0] * 2).reshape(2, 2)
distance_labels_1 = [(1 - np.sum(x)) for x in dummy_1] # [0, 0]

img_0 = np.asarray(list(rep_0_img) * 2).reshape(2, 784) # (r, c)
img_1 = np.asarray(list(rep_1_img) * 2).reshape(2, 784)


sess.run(tf.global_variables_initializer())
start = time.time()
# minibatch実行
for i in range(EPOCHS):

    X_train,  y_train  = shuffle(X['train'], y['train']) # image and label of "0" or "1"
    XL_train, yL_train = shuffle(X['train'], y['train']) # X_train = (2, 784), y_train = (2, 2), (r,c)

    for OFFSET in range(0, len(X['train']), BATCH_SIZE): # len(X['train']) = 2, BATCH_SIZE = 1

        batch_x, batch_y   = X_train[OFFSET:  (OFFSET + BATCH_SIZE)], y_train[OFFSET:  (OFFSET + BATCH_SIZE)]
        batch_xL, batch_yL = XL_train[OFFSET: (OFFSET + BATCH_SIZE)], yL_train[OFFSET: (OFFSET + BATCH_SIZE)]

        simi = np.asarray(batch_y) * np.asarray(batch_yL)
        distance_labels = [(1 - np.sum(x)) for x in simi]
        # optimizer
        train_step.run(feed_dict = {x:  batch_x,  y_:  batch_y,
                                    xL: batch_xL, yL_: batch_yL,
                                    diss: distance_labels, keep_prob: 1, })

Elapsed_time = time.time() - start


# テストデータで今回のモデルを最終評価　指標は精度
test_len = len(y['test'])

dummy_0 = np.asarray([1.0, 0.0] * test_len).reshape(test_len, 2)
dummy_1 = np.asarray([0.0, 1.0] * test_len).reshape(test_len, 2)

img_0 = np.asarray(list(rep_0_img) * test_len).reshape(test_len, 784)
img_1 = np.asarray(list(rep_1_img) * test_len).reshape(test_len, 784)

distance_labels_0 = [     np.sum(x)  for x in dummy_0]
distance_labels_1 = [(1 - np.sum(x)) for x in dummy_1]

eval_acc = sess.run(y_conv, feed_dict = {x:  X['test'], y_:  y['test'],
                                         xL: img_0,     yL_: dummy_0,
                                         diss: distance_labels_0, keep_prob:1,})
eval_1acc = sess.run(y_conv, feed_dict = {x:  X['test'], y_:  y['test'],
                                          xL: img_1,     yL_: dummy_1,
                                          diss: distance_labels_1, keep_prob: 1,})# print(eval_acc)# print(eval_1acc)

matome = np.concatenate([eval_acc, eval_1acc], axis = 1)
predict_result = np.argmin(matome, axis = 1) # print(predict_result.shape) # (200, )
print(predict_result)

y_right = np.argmax(np.asarray(y['test']), axis = 1) # print(y_right.shape) # (200, )
print(y_right)

acc_result = np.equal(y_right, predict_result)
tmp = np.sum(acc_result)/len(acc_result)

print('Final: TestAccuracy: %f' % tmp)
print('Time : %f' % Elapsed_time)
