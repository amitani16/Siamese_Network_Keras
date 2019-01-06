import numpy as np

dummy_0 = np.asarray([1.0, 0.0, 2.0] * 2) # [1, 0, 1, 0]
print(dummy_0)
'''[1. 0. 2. 1. 0. 2.]'''

dummy_0 = dummy_0.reshape(2, 3) # [[1, 0], [1, 0]] (row, col)
print(dummy_0)
'''
[[1. 0. 2.]
 [1. 0. 2.]]
'''
distance_labels_0 = [np.sum(x) for x in dummy_0]
print(distance_labels_0)
'''[3.0, 3.0]'''
