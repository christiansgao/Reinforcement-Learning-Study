
import numpy as np
from scipy import ndimage
from hashlearner.helper.mnist import MnistLoader

EDGE_KERNEL_1 = np.array([[1,0,-1],[0,0,0],[-1,0,1]])
EDGE_KERNEL_2 = np.array([[0,1,0],[1,-2,1],[0,1,0]])
EDGE_KERNEL_3 = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
SHARPEN = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])


def gaussian(mnist_image, sigma=2):
    return ndimage.gaussian_filter(mnist_image, sigma=sigma)

def edge(mnist_image):
    sx = ndimage.sobel(mnist_image, axis=0, mode='constant')
    sy = ndimage.sobel(mnist_image, axis=1, mode='constant')
    sob = np.hypot(sx, sy)
    return sob

def convolve_filter(mnist_image, kernel=SHARPEN):
    convolved_image = ndimage.convolve(mnist_image, kernel, mode='constant', cval=0.0)
    return convolved_image

def main():
    mnist_image = MnistLoader.get_example()
    MnistLoader.show(mnist_image)
    test = convolve_filter(mnist_image, kernel=EDGE_KERNEL_1)
    MnistLoader.show(test)
    pass

if __name__ == "__main__":
    main()



