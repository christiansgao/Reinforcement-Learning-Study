import numpy as np
import itertools
import warnings
from skimage.transform import rescale


def convolve(image=None, kernel_dim=(3, 3)):
    view_shape = tuple(np.subtract(image.shape, kernel_dim) + 1) + kernel_dim
    strides = image.strides + image.strides

    sub_matrices = np.lib.stride_tricks.as_strided(image, view_shape, strides)
    sub_matrices_list = list(map(list, sub_matrices))
    convolved_matrixs = list(itertools.chain(*sub_matrices_list))

    return convolved_matrixs

def down_scale_images(images: list, ratio: float):
    if ratio == 1:
        return images
    images_rescaled = [down_scale_image(image, ratio) for image in images]
    return images_rescaled

def down_scale_image(image: np.ndarray, ratio: int):
    with warnings.catch_warnings(record=True) as w:
        return rescale(image, ratio, preserve_range = True)

def binarize_image(image: np.ndarray, threshold: int):
    binarized_image = image.copy()
    binarized_image[binarized_image <= threshold] = 0
    binarized_image[binarized_image > threshold] = 1
    return binarized_image

def binarize_images(images: list, threshold: int):
    binarized_images = [binarize_image(image, threshold) for image in images]
    return binarized_images

def _batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def batch(iterable, n=1):
        return list(_batch(iterable, n))

def rank_nodes(nodes, training_set, test_set, n=1):
    pass