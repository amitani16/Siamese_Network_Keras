from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
from tensorflow.keras.preprocessing.image import array_to_img

nb_class = 10
nb_class_list = list(range(nb_class))

(IMG_W, IMG_H, IMG_D) = (28, 28, 1)
img_list = []

PATH = '/Users/ichiroamitani/Documents/Software/GitHub/Siamese_network/image/'


def load_MNIST_image_label(data_path, nb_class = 10):

    nb_class_list = list(range(nb_class))
    e = np.eye(nb_class)

    train_img_list =[]
    train_label_list = []
    test_img_list = []
    test_label_list = []

    for i in nb_class_list:

        for filename in glob.glob(data_path + 'train/'  + str(i) + '/*.jpg'):
            img = Image.open(filename).convert('L')
            img = np.asarray(img)/255
            train_img_list.append(img)
            train_label_list.append(e[i])

        for filename in glob.glob(PATH + 'test/' + str(i) + '/*.jpg'):
            img = Image.open(filename).convert('L')
            img = np.asarray(img)/255
            test_img_list.append(img)
            test_label_list.append(e[i])

    return (train_img_list, train_label_list), (test_img_list, test_label_list)


def show_effect(img_list, grid_shape = (3, 3)):

    (r, c) = grid_shape
    fig, axes = plt.subplots(r, c)
    k = 0
    for i in range(c):
        for j in range(r):
            axes[i, j].matshow(img_list[k].reshape(IMG_W, IMG_H), cmap = 'gray')
            axes[i, j].get_yaxis().set_visible(False)
            axes[i, j].get_xaxis().set_visible(False)
            k = k + 1

    plt.show()


def show_effect_2(img_list, grid_shape = (3, 3)):

    (r, c) = grid_shape
    gs = gridspec.GridSpec(r, c)
    gs.update(wspace = 0.1, hspace = 0.1)

    for i in range(r * c):

        plt.subplot(gs[i])
        plt.imshow(img_list[i].reshape(IMG_W, IMG_H), cmap = 'gray', aspect = 'auto')
        plt.axis("off")

    plt.show()


def generate_augmented_image(data_generator, img_src, nb_images = 9):

    g = data_generator.flow(img_src, batch_size = 1)

    img_list = []
    for i in range(nb_images):
        img_list.append(g.next())

    return img_list


def generate_random_augmented_image(data_generator, img_src, nb_images = 9):

    img_list = []
    for i in range(nb_images):
        img_list.append( data_generator.random_transform(img_src) )

    return img_list


def augment_MNIST_image(label_src, img_src, nb_images = 2500):

    datagen = ImageDataGenerator(rotation_range     = 15.0, # degree
                                 width_shift_range  = 0.2,
                                 height_shift_range = 0.2,
                                 shear_range        = 5.0, # degree
                                 zoom_range         = 0.5,
                                 horizontal_flip    = True,
                                 vertical_flip      = True,
                                )

    augmented_img_list = generate_random_augmented_image(datagen, img_src, nb_images)

    return (augmented_img_list, label_src, )


def display_n_images_in_n_windows(img_list):

    for i in range(len(img_list)):

        plt.figure(i)
        plt.imshow(img_list[i].reshape(IMG_W, IMG_H), cmap = 'gray', aspect = 'auto')

    plt.show()



if __name__ == '__main__':

    train_data, test_data = load_MNIST_image_label(PATH)
    img_list = train_data[0]

    display_n_images_in_n_windows(img_list)

    '''
    x = img_list[5].reshape(IMG_H, IMG_W, 1)
    grid_shape = (20, 20)

    daga_list = augment_MNIST_image(train_data[1], x, nb_images = grid_shape[0] * grid_shape[1])

    show_effect_2(daga_list[0], grid_shape)
    '''














#
