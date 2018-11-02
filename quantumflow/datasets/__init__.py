
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
Standard QuantumFlow datasets.

>>> import quantumflow as qf
>>> graphs = qf.datasets.load_stdgraphs(10)
"""

from typing import Tuple, List

import numpy as np

import networkx as nx
from PIL import Image


def load_stdgraphs(size: int) -> List[nx.Graph]:
    """Load standard graph validation sets

    For each size (from 6 to 32 graph nodes) the dataset consists of
    100 graphs drawn from the Erdős-Rényi ensemble with edge
    probability 50%.
    """
    from pkg_resources import resource_stream

    if size < 6 or size > 32:
        raise ValueError('Size out of range.')

    filename = 'datasets/data/graph{}er100.g6'.format(size)
    fdata = resource_stream('quantumflow', filename)
    return nx.read_graph6(fdata)


_MNIST_BORDER = 5


def load_mnist(size: int = None,
               border: int = _MNIST_BORDER,
               blank_corners: bool = False,
               nums: List[int] = None) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Download and rescale the MNIST database of handwritten digits

    MNIST is a dataset of 60,000 28x28 grayscale images handwritten digits,
    along with a test set of 10,000 images. We use Keras to download and
    access the dataset. The first invocation of this method may take a while
    as the dataset has to be downloaded and cached.

    If size is None, then we return the original MNIST data.
    For rescaled MNIST, we chop off the border, downsample to the
    desired size with Lanczos resampling, and then (optionally) zero out the
    corner pixels.

    Returns (x_train, y_train, x_test, y_test)

    x_train ndarray of shape (60000, size, size)
    y_train ndarray of shape (60000,)
    x_test ndarray of shape (10000, size, size)
    y_test ndarray of shape (10000,)
    """
    # DOCME: Fix up formatting above,
    # DOCME: Explain nums argument

    # JIT import since keras startup is slow
    from keras.datasets import mnist

    def _filter_mnist(x: np.ndarray, y: np.ndarray, nums: List[int] = None) \
            -> Tuple[np.ndarray, np.ndarray]:
        xt = []
        yt = []
        items = len(y)
        for n in range(items):
            if nums is not None and y[n] in nums:
                xt.append(x[n])
                yt.append(y[n])
        xt = np.stack(xt)
        yt = np.stack(yt)
        return xt, yt

    def _rescale(imgarray: np.ndarray, size: int) -> np.ndarray:
        N = imgarray.shape[0]

        # Chop off border
        imgarray = imgarray[:, border:-border, border:-border]

        rescaled = np.zeros(shape=(N, size, size), dtype=np.float)
        for n in range(0, N):
            img = Image.fromarray(imgarray[n])
            img = img.resize((size, size), Image.LANCZOS)
            rsc = np.asarray(img).reshape((size, size))
            rsc = 256.*rsc/rsc.max()
            rescaled[n] = rsc

        return rescaled.astype(dtype=np.uint8)

    def _blank_corners(imgarray: np.ndarray) -> None:
        # Zero out corners
        sz = imgarray.shape[1]
        corner = (sz//2)-1
        for x in range(0, corner):
            for y in range(0, corner-x):
                imgarray[:, x, y] = 0
                imgarray[:, -(1+x), y] = 0
                imgarray[:, -(1+x), -(1+y)] = 0
                imgarray[:, x, -(1+y)] = 0

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if nums:
        x_train, y_train = _filter_mnist(x_train, y_train, nums)
        x_test, y_test = _filter_mnist(x_test, y_test, nums)

    if size:
        x_train = _rescale(x_train, size)
        x_test = _rescale(x_test, size)

    if blank_corners:
        _blank_corners(x_train)
        _blank_corners(x_test)

    return x_train, y_train, x_test, y_test
