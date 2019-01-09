import tensorflow.keras as K
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Lambda, MaxPooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import glob
from PIL import Image
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import keras_util
import pickle


nb_class = 10
nb_class_list = list(range(nb_class))
nb_test_data = 100

(IMG_W, IMG_H, IMG_D) = (28, 28, 1)

PATH = '/Users/ichiroamitani/Documents/Software/GitHub/Siamese_network/image/'
weight_data_path = '/Users/ichiroamitani/Documents/Software/GitHub/Siamese_network/'


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
    conv_net.add(Conv2D(filters = 256, kernel_size = (4, 4), padding  = 'same', activation = 'relu',
                        kernel_initializer = RandomNormal(mean = 0, stddev = 0.01)))
    conv_net.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))

    conv_net.add(Flatten())
    conv_net.add(Dense(units = 1024, activation = "sigmoid",
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


def get_train_data_pair(img_list, sample_size = 100):

    # different : target = 0, same : target = 1s
    labels  = np.random.choice(nb_class, size = 3, replace = False)
    target_diff = np.zeros(sample_size)
    target_same = np.ones(sample_size)
    target = np.concatenate([target_diff, target_same])

    label_diff_A = labels[0]
    label_diff_B = labels[1]
    label_same_A = labels[2] # make same data set
    label_same_B = labels[2] # make same data set

    img_list_len = len( img_list[ label_diff_A ] )
    indices = np.random.randint(0, img_list_len, size = sample_size)

    selected_diff_img_A_list = []
    selected_diff_img_B_list = []
    selected_same_img_A_list = []
    selected_same_img_B_list = []
    for i in indices:
        selected_diff_img_A_list.append(img_list[label_diff_A][i])
        selected_diff_img_B_list.append(img_list[label_diff_B][i])
        selected_same_img_A_list.append(img_list[label_same_A][i])
        selected_same_img_B_list.append(img_list[label_same_B][i])

    selected_same_img_A_list = shuffle(selected_same_img_A_list)
    selected_same_img_B_list = shuffle(selected_same_img_B_list)

    A = np.concatenate([selected_diff_img_A_list, selected_same_img_A_list])
    B = np.concatenate([selected_diff_img_B_list, selected_same_img_B_list])

    return (A, B), target


def get_test_data_pair(img_list):

    A_label = np.random.randint(0, nb_class)
    A_index = np.random.randint(0, nb_test_data)

    B_indices  = np.random.randint(0, nb_test_data, size = nb_test_data)

    # print('len(img_list) = ', len(img_list))
    img = []
    for i in range(nb_class):

        start = i       * nb_test_data
        end   = (i + 1) * nb_test_data
        # print('i = {}, start = {}, end = {}'.format(i, start, end))

        img.append(img_list[ start:end ])
        # print(type(a))

    # diff class : target = 0, same class : target = 1
    targets = np.zeros(nb_class)
    targets[A_label] = 1

    selected_img_A_list = []
    selected_img_B_list = []

    img_A = np.asarray(img[A_label][A_index]).reshape(IMG_W, IMG_H, IMG_D)
    for i in range(nb_class):
        selected_img_A_list.append(img_A)

    for i in range(nb_class):
        index = B_indices[i]
        img_B = np.asarray(img[i][index]).reshape(IMG_W, IMG_H, IMG_D)
        selected_img_B_list.append(img_B)

    return (selected_img_A_list, selected_img_B_list), targets


def test_oneshot(model, img_list, nb_validation):

    n_correct = 0
    for i in range(nb_validation):

        inputs, targets = get_test_data_pair(img_list)
        probs = model.predict(inputs)
        # print('targets = ', targets)
        # print('probs = ', probs)
        if np.argmax(probs) == np.argmax(targets):
            n_correct += 1

    percent_correct = n_correct / nb_validation

    return percent_correct


def main():

    ''''''
    print('Model Building Started')
    input_shape = (IMG_H, IMG_W, IMG_D)
    siamese_net = get_siamese_model(input_shape)
    siamese_net.summary()
    optimizer = Adam(lr = 0.00006)
    siamese_net.compile(loss = "binary_crossentropy", optimizer = optimizer)
    print('Model Building Finished')
    ''''''

    print("Image Read Started")
    train_image_list = []
    for i in nb_class_list:
        with open(PATH + str(i) + '.pickle','rb') as f:
            train_image_list.append(pickle.load(f)[0])

    train_data, test_data = keras_util.load_MNIST_image_label(PATH)
    test_img_list  = test_data[0] # train_data is not used
    print("Image Read Finished")

    print('Training Loop Started')
    n_iter = 200
    best = -1
    evaluate_every = 10
    for i in range(1, n_iter):

        train_img_pair, target = get_train_data_pair(train_image_list, sample_size = 50)
        loss = siamese_net.train_on_batch(train_img_pair, target)

        # print(loss)
        ''''''
        if i % evaluate_every == 0:
            val_acc = test_oneshot(siamese_net, img_list = test_img_list, nb_validation = 5)

            if val_acc >= best:
                print("Current best: {:.2f}, previous best: {:.2f}".format(val_acc, best))
                # print("Saving weights to: {0} \n".format(weight_data_path))
                # siamese_net.save_weights(weight_data_path + 'model_weights.h5')
                best = val_acc
        ''''''

    # open('SiameseNet.json',"w").write(siamese_net.to_json())
    print('Training Loop Finished')


if __name__ == '__main__':

    main()






#
#
#
