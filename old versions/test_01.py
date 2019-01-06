import numpy as np

dummy_0 = np.asarray([1.0, 0.0] * 2)
dummy_0 = dummy_0.reshape(2, 2)

#img_0_dummy = np.asarray(list(rep_0_img)  * 2)
#img_0_dummy = img_0_dummy.reshape(2, 784)
#distance_labels = [np.sum(x) for x in dummy_0]

for x in dummy_0:
    print(x)
    print(np.sum(x))
