import matplotlib.pyplot as plt
import numpy as np
from package import otro, regression

add = regression.add(10, 12)

t = np.arange(0.0, 50.0, 0.01)
s = np.full(t.size, add)

plt.plot(t, s)
plt.title('Awesome!')
plt.savefig('img/foo.png')
