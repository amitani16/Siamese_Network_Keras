'''
http://aidiary.hatenablog.com/entry/20161212/1481549365

keras.preprocessing.image.ImageDataGenerator(
    featurewise_center            = False,
    samplewise_center             = False,
    featurewise_std_normalization = False,
    samplewise_std_normalization  = False,
    zca_whitening                 = False,
    rotation_range                = 0.,
    width_shift_range             = 0.,
    height_shift_range            = 0.,
    shear_range                   = 0.,
    zoom_range                    = 0.,
    channel_shift_range           = 0.,
    fill_mode                     = 'nearest',
    cval                          = 0.,
    horizontal_flip               = False,
    vertical_flip                 = False,
    rescale                       = None,
    dim_ordering                  = K.image_dim_ordering())
'''
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

    def show_effect(data_generator, x):
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


    # datagen = ImageDataGenerator(rotation_range = 90)
    # show_effect(datagen, x)
    # datagen = ImageDataGenerator(width_shift_range = 0.5)
    # show_effect(datagen, x)
    # datagen = ImageDataGenerator(height_shift_range = 0.5)
    # show_effect(datagen, x)
    # datagen = ImageDataGenerator(shear_range = 20) # 0.78 = pi/4
    # show_effect(datagen, x)
    datagen = ImageDataGenerator(zoom_range = 0.5)
    show_effect(datagen, x)
    datagen = ImageDataGenerator(horizontal_flip = True)
    show_effect(datagen, x)
    datagen = ImageDataGenerator(vertical_flip = True)
    show_effect(datagen, x)
#
#
#
