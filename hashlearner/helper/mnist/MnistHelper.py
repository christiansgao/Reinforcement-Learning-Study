import numpy as np
import itertools


def convolve(image=None, kernel_dim=(3, 3)):
    view_shape = tuple(np.subtract(image.shape, kernel_dim) + 1) + kernel_dim
    strides = image.strides + image.strides

    sub_matrices = np.lib.stride_tricks.as_strided(image, view_shape, strides)
    sub_matrices_list = list(map(list, sub_matrices))
    convolved_matrixs = list(itertools.chain(*sub_matrices_list))

    return convolved_matrixs
