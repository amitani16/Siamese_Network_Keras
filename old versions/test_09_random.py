import numpy as np
import glob
from PIL import Image
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

print("Image read started")
for i in nb_class_list:

    for filename in glob.glob(PATH + 'train/'  + str(i) + '/*.jpg'):
        img = Image.open(filename).resize( (IMG_W, IMG_H) ).convert('L') # original is 28x28. so no need
        img = ( np.asarray(img) ).reshape(IMG_W * IMG_H)
        train_img_list.append(img)
        train_label_list.append(e[i])

    for filename in glob.glob(PATH + 'test/' + str(i) + '/*.jpg'):
        img = Image.open(filename).resize( (IMG_W, IMG_H) ).convert('L')
        img = ( np.asarray(img) ).reshape(IMG_W * IMG_H)
        test_img_list.append(img)
        test_label_list.append(e[i])
print("Image read finished")


def get_pair_train_data(label_list, img_list, replace = False):

    labels = np.random.choice(nb_class, size = 2, replace = replace)
    label_pair = [ label_list[labels[0]], label_list[labels[1]], ]
    img_pair   = [ img_list[labels[0]],   img_list[labels[1]], ]

    return label_pair, img_pair


train_label_pair, train_img_pair = get_pair_train_data(train_label_list, train_img_list)
print(train_label_pair)
img_0 = train_img_pair[0].reshape(IMG_W, IMG_H)
img_1 = train_img_pair[1].reshape(IMG_W, IMG_H)

img = np.concatenate((img_0, img_1), axis = 1)
plt.imshow(img)
plt.show()





#
#
#
