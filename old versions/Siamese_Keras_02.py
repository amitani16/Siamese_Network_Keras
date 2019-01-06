import tensorflow.keras as K
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Lambda, MaxPooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import glob
from PIL import Image

X = {}
y = {}

train_img_list =[]
train_label_list = []
test_img_list = []
test_label_list = []

nb_class = 10
nb_class_list = list(range(nb_class))
e = np.eye(nb_class)

IMG_W, IMG_H, IMG_D = 28, 28, 1
PATH = '/Users/ichiroamitani/Documents/Software/Python/Siamese_network-master/image/'

print("Image Read Started")
for i in nb_class_list:

    for filename in glob.glob(PATH + 'train/'  + str(i) + '/*.jpg'):
        img = Image.open(filename).resize( (IMG_W, IMG_H) ).convert('L') # original is 28x28. so no need
        img = ( np.asarray(img)/255 ).reshape(IMG_W * IMG_H)
        train_img_list.append(img)
        train_label_list.append(e[i])

    for filename in glob.glob(PATH + 'test/' + str(i) + '/*.jpg'):
        img = Image.open(filename).resize( (IMG_W, IMG_H) ).convert('L')
        img = ( np.asarray(img)/255 ).reshape(IMG_W * IMG_H)
        test_img_list.append(img)
        test_label_list.append(e[i])
print("Image Read Finished")


print('Model Building Started')
input_shape = (IMG_H, IMG_W, IMG_D)
input_A = Input(input_shape)
input_B = Input(input_shape)

#build conv_net to use in each siamese 'leg'
conv_net = Sequential()

# First layer 任意のkernel_initializerの設定方法を調べる
conv_net.add(Conv2D(filters = 16, kernel_size = (4, 4), padding  = 'same', activation = 'relu',
                    kernel_initializer = RandomNormal(mean = 0, stddev = 0.01)))
conv_net.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))

# Second layer
conv_net.add(Conv2D(filters = 64, kernel_size = (5, 5), padding  = 'same', activation = 'relu',
                    kernel_initializer = RandomNormal(mean = 0, stddev = 0.01)))
conv_net.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))

# Third layer
conv_net.add(Conv2D(filters = 32, kernel_size = (5, 5), padding  = 'same', activation = 'relu',
                    kernel_initializer = RandomNormal(mean = 0, stddev = 0.01)))
conv_net.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))

conv_net.add(Flatten())
conv_net.add(Dense(units = 512, activation = "sigmoid",
                   kernel_initializer = RandomNormal(mean = 0, stddev = 0.01),
                   bias_initializer = RandomNormal(mean = 0.5, stddev = 0.01)))

#call the convnet Sequential model on each of the input tensors so params will be shared
encoded_A = conv_net(input_A)
encoded_B = conv_net(input_B)

#layer to merge two encoded inputs with the l1 distance between them
L1_layer = Lambda(lambda tensors:K.backend.abs(tensors[0] - tensors[1]))
L1_distance = L1_layer([encoded_A, encoded_B])

prediction = Dense(units = 1, activation = 'sigmoid', bias_initializer = RandomNormal(mean = 0.5, stddev = 0.01))(L1_distance)
siamese_net = Model(inputs = [input_A, input_B], outputs = prediction)
optimizer = Adam(0.001)

siamese_net.compile(loss = "binary_crossentropy", optimizer = optimizer)
siamese_net.count_params()
print('Model Building Finished')


def get_pair_train_data(label_list, img_list, replace = False):

    labels = np.random.choice(nb_class, size = 2, replace = replace)
    label_pair = [ label_list[labels[0]], label_list[labels[1]], ]
    img_pair   = [ img_list[labels[0]],   img_list[labels[1]], ]

    return label_pair, img_pair


print('Training Loop Started')
n_iter = 10

for i in range(1, n_iter):

    train_label_pair, train_img_pair = get_pair_train_data(train_label_list, train_img_list)

    loss = siamese_net.train_on_batch(train_img_pair, train_label_pair)
    print(loss)

print('Training Loop Finished')









#
#
#
