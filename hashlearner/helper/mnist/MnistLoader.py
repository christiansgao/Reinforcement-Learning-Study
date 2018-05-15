import os
import struct
import numpy as np

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

class MnistImage:

    def __init__(self, label, image):
        self.labels = label
        self.images = image


def read_mnist(dataset = "training", path = "/Users/christiangao/Documents/MAS/thesis/Reinforcement-Learning-Study/data/"):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    #path = os.path.abspath(path)
    #path = "/home/christian/Documents/Reinforcement-Learning-Study/data/"

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (str(lbl[idx]), img[idx])

    # Create an iterator which returns each image in turn
    images = [get_img(i) for i in range(len(lbl))]
    return images

def read_mnist_objs(dataset = "training", path = "../../data/"):
    '''
    :type dataset:
    :type path:
    :rtype: list of MnistImage
    '''
    raw_images = read_mnist(dataset, path)
    images = [MnistImage(raw_images[0],raw_images[1]) for raw_image in raw_images]

    return images


def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

def get_example():
    return read_mnist()[0][1]

def main():
    training = read_mnist("training")
    image = training[0][1]
    image = image[1:20,1:20]
    show(image)

if __name__ == "__main__":
    main()

