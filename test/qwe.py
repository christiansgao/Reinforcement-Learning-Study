import numpy as np
import sklearn.metrics as sk_metrics
from scipy import ndimage
from hashlearner.helper.mnist import MnistLoader

mnist_data = MnistLoader.read_mnist()
mnist_data = mnist_data[0][1]


a = np.array([[1, 2, 3, 1],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0]])
k = np.array([[1,1,1],[1,1,1],[1,1,1]])
test = ndimage.convolve(a, k, mode='constant', cval=0.0)
print(test)

blurred_face = ndimage.gaussian_filter(mnist_data, sigma=3)
pass