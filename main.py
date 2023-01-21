import numpy as np
import matplotlib.pyplot as plt

x_new = np.linspace(0, 1, 100)
y = np.exp(x_new)

plt.plot(x_new, y)
plt.show()