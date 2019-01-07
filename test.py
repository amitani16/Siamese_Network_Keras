import numpy as np

a = np.zeros(10)
b = np.ones(5)

print(a)
print(b)

c = np.concatenate([a, b, b])

print(c)
