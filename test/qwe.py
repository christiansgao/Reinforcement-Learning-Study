import numpy as np
from skopt import gp_minimize

def f(x):
    return (np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) *
            np.random.randn() * 0.1)
#mnist_data = MnistLoader.read_mnist()
#mnist_data = mnist_data[0][1]

res = gp_minimize(f, [(-2.0, 2.0)])


print(res)
#blurred_face = ndimage.gaussian_filter(mnist_data, sigma=3)
