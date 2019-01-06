import pickle
import keras_util

nb_class = 10
nb_class_list = list(range(nb_class))
(IMG_W, IMG_H, IMG_D) = (28, 28, 1)


PATH = '/Users/ichiroamitani/Documents/Software/GitHub/Siamese_network/image/'


if __name__ == '__main__':

    train_data, test_data = keras_util.load_MNIST_image_label(PATH)
    img_list   = train_data[0]
    label_list = train_data[1]

    for i in nb_class_list:

        print(i)
        x = img_list[i].reshape(IMG_H, IMG_W, 1)

        augmented_img_list = keras_util.augment_MNIST_image(label_list, x, nb_images = 2500)

        with open(PATH + str(i) + '.pickle', 'wb') as f:
            pickle.dump(augmented_img_list, f)
