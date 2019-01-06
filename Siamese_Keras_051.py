import tensorflow.keras as K
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Lambda, MaxPooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import glob
from PIL import Image
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

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
        img = Image.open(filename).convert('L')
        img = np.asarray(img)/255
        train_img_list.append(img)
        train_label_list.append(e[i])

    for filename in glob.glob(PATH + 'test/' + str(i) + '/*.jpg'):
        img = Image.open(filename).convert('L')
        img = np.asarray(img)/255
        test_img_list.append(img)
        test_label_list.append(e[i])
print("Image Read Finished")


def get_siamese_model(input_shape):
    '''
    https://github.com/akshaysharma096/Siamese-Networks/blob/master/Few%20Shot%20Learning%20-%20V1.ipynb
    '''
    input_A = Input(input_shape)
    input_B = Input(input_shape)

    #build conv_net to use in each siamese 'leg'
    conv_net = Sequential()

    # First layer 任意のkernel_initializerの設定方法を調べる
    conv_net.add(Conv2D(filters = 16, kernel_size = (4, 4), padding  = 'same', activation = 'relu',
                        kernel_initializer = RandomNormal(mean = 0, stddev = 0.01)))
    conv_net.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))

    # Second layer
    conv_net.add(Conv2D(filters = 64, kernel_size = (4, 4), padding  = 'same', activation = 'relu',
                        kernel_initializer = RandomNormal(mean = 0, stddev = 0.01)))
    conv_net.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))

    # Third layer
    conv_net.add(Conv2D(filters = 32, kernel_size = (4, 4), padding  = 'same', activation = 'relu',
                        kernel_initializer = RandomNormal(mean = 0, stddev = 0.01)))
    conv_net.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))

    conv_net.add(Flatten())
    conv_net.add(Dense(units = 512, activation = "sigmoid",
    # conv_net.add(Dense(units = 1024 * 2 * 2, activation = "sigmoid",
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

    return siamese_net


def get_pair_train_data(img_list):

    labels = np.random.choice(nb_class, size = 2, replace = False)

    # each class contains only one example and chosen from different classes.
    # so target = 0
    target = np.zeros(1)

    img_0 = img_list[ labels[0] ].reshape(-1, IMG_W, IMG_H, IMG_D)
    img_1 = img_list[ labels[1] ].reshape(-1, IMG_W, IMG_H, IMG_D)

    return (img_0, img_1), target


def get_diff_pair_test_data(img_list, sample_size):
    '''
    https://github.com/akshaysharma096/Siamese-Networks/blob/master/Few%20Shot%20Learning%20-%20V1.ipynb
    '''
    labels  = np.random.choice(nb_class, size = 2, replace = False)
    indices = np.random.randint(0, 100, size = sample_size)

    # each class contains only one example and chosen from different classes.
    # so target = 0
    target = np.zeros(sample_size)

    img_0_list = img_list[ (labels[0] * 100):( (labels[0] + 1) * 100 ) ]
    img_1_list = img_list[ (labels[1] * 100):( (labels[1] + 1) * 100 ) ]

    selected_img_0_list = []
    selected_img_1_list = []
    for i in indices:
        selected_img_0_list.append(img_0_list[i])
        selected_img_1_list.append(img_1_list[i])

    selected_img_0 = np.asarray(selected_img_0_list).reshape(sample_size, IMG_W, IMG_H, IMG_D)
    selected_img_1 = np.asarray(selected_img_1_list).reshape(sample_size, IMG_W, IMG_H, IMG_D)

    return (selected_img_0, selected_img_1), target


def test_oneshot(model, img_list, nb_validation):

    n_correct = 0
    sample_size = 5

    for i in range(nb_validation):

        inputs, targets = get_diff_pair_test_data(img_list = img_list, sample_size = sample_size)
        probs = model.predict(inputs)
        # print('probs = ', probs)
        if np.argmax(probs) == np.argmax(targets):
            n_correct += 1

    percent_correct = n_correct / nb_validation

    return percent_correct


if __name__ == '__main__':

    print('Model Building Started')
    input_shape = (IMG_H, IMG_W, IMG_D)
    siamese_net = get_siamese_model(input_shape)
    siamese_net.summary()
    optimizer = Adam(lr = 0.00006)
    siamese_net.compile(loss = "binary_crossentropy", optimizer = optimizer)
    print('Model Building Finished')

    data_path = '/Users/ichiroamitani/Documents/Software/Python/Siamese_network-master/'
    weights_path = data_path + 'model_weights.h5'

    print('Training Loop Started')
    n_iter = 2000
    best = -1
    evaluate_every = 10
    for i in range(1, n_iter):

        train_img_pair, target = get_pair_train_data(train_img_list)
        loss = siamese_net.train_on_batch(train_img_pair, target) # print(loss)

        if i % evaluate_every == 0:
            val_acc = test_oneshot(siamese_net, img_list = test_img_list, nb_validation = 5)

            if val_acc >= best:
                print("Current best: {:.2f}, previous best: {:.2f}".format(val_acc, best))
                # print("Saving weights to: {0} \n".format(weights_path))
                # siamese_net.save_weights(weights_path)
                best = val_acc

    print('Training Loop Finished')









#
#
#
