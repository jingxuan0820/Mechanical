
from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt


n = 100
h = n// 2
print(h)
dimen = 2
print(np.ones((h,dimen)))


data[:h,:] = data[:h,:] - 3*np.ones((h,dimen))