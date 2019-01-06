from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt

nb_class = 10
nb_class_list = list(range(nb_class))

(IMG_W, IMG_H, IMG_D) = (28, 28, 1)
img_list = []

PATH = '/Users/ichiroamitani/Documents/Software/Python/Siamese_network-master/image/'

print("Image Read Started")
for i in nb_class_list:

    for filename in glob.glob(PATH + 'train/'  + str(i) + '/*.jpg'):
        img = Image.open(filename).convert('L')
        img = np.asarray(img)/255
        img_list.append(img)

print("Image Read Finished")


if __name__ == '__main__':

    x = img_list[0].reshape(1, IMG_H, IMG_W, 1)

    datagen = ImageDataGenerator(rotation_range = 90)
    g = datagen.flow(x, batch_size = 1)

    (r, c) = (3, 3)
    img = []
    for i in range(r * c):
        img.append(g.next())

    fig, axes = plt.subplots(r, c)
    k = 0
    for i in range(c):
        for j in range(r):
            axes[i, j].matshow(img[k].reshape(IMG_W, IMG_H), cmap = 'gray')
            axes[i, j].get_yaxis().set_visible(False)
            axes[i, j].get_xaxis().set_visible(False)
            k = k + 1

    plt.show()

    # datagen = ImageDataGenerator(width_shift_range = 0.2)
    datagen = ImageDataGenerator(height_shift_range = 0.2)
    g = datagen.flow(x, batch_size = 1)

    (r, c) = (3, 3)
    img = []
    for i in range(r * c):
        img.append(g.next())

    fig, axes = plt.subplots(r, c)
    k = 0
    for i in range(c):
        for j in range(r):
            axes[i, j].matshow(img[k].reshape(IMG_W, IMG_H), cmap = 'gray')
            axes[i, j].get_yaxis().set_visible(False)
            axes[i, j].get_xaxis().set_visible(False)
            k = k + 1

    plt.show()


    datagen = ImageDataGenerator(shear_range = 20) # 0.78 = pi/4
    g = datagen.flow(x, batch_size = 1)

    (r, c) = (3, 3)
    img = []
    for i in range(r * c):
        img.append(g.next())

    fig, axes = plt.subplots(r, c)
    k = 0
    for i in range(c):
        for j in range(r):
            axes[i, j].matshow(img[k].reshape(IMG_W, IMG_H), cmap = 'gray')
            axes[i, j].get_yaxis().set_visible(False)
            axes[i, j].get_xaxis().set_visible(False)
            k = k + 1

    plt.show()



#
#
#
