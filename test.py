import numpy as np
import matplotlib.pyplot as plt
x1 = np.arange(0.0, 2*np.pi, 0.1)
x2 = np.arange(0.0, 2*np.pi, 0.1)

plt.figure(1)
plt.subplot(221)
plt.plot(x1, np.sin(x1), 'r--', x2, np.sin(x2), 'k')
plt.subplot(224)
plt.plot(x2, np.sin(2*np.pi*x2), 'r--')

plt.figure(2)
plt.plot(x1, np.cos(x1), 'r--')

plt.show()
