# from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

    fig, axes = plt.subplots(2, 5)
    # print(type(axes))
    k = 0
    for i in range(2):
        for j in range(5):
            axes[i, j].matshow(img_list[k].reshape(IMG_W, IMG_H), cmap = 'gray')
            axes[i, j].get_yaxis().set_visible(False)
            axes[i, j].get_xaxis().set_visible(False)
            k = k + 1

    # plt.xticks([])
    # plt.yticks([])

    plt.show()




#
#
#
