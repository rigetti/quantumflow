#!/usr/bin/env python

# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
Download and rescale the MNIST database of handwritten digits
"""

import argparse
import random

import numpy as np
from PIL import Image

import quantumflow as qf
from quantumflow.datasets import _MNIST_BORDER


# ---------- Command Line Interface ----------
def _cli():

    parser = argparse.ArgumentParser(description='Download and rescale MNIST')

    parser.add_argument('--version', action='version',
                        version=qf.__version__)

    parser.add_argument('size', type=int, action='store',
                        help='Rescale to size x size pixels')

    parser.add_argument('--samples', type=int, action='store', default=0,
                        help='Save this number of example rescaled images.')

    parser.add_argument('--corners', action='store_true', default=False,
                        help='Blank out corners of images')

    parser.add_argument('--border', type=int, action='store',
                        default=_MNIST_BORDER,
                        help='Size of border to remove before rescaling')

    opts = vars(parser.parse_args())
    size = opts.pop('size')
    samples = opts.pop('samples')
    corners = opts.pop('corners')
    border = opts.pop('border')

    print('Loading MNIST...')

    (x_train, y_train, _, _) = qf.datasets.load_mnist()

    (x_train_rescaled, y_train, x_test_rescaled, y_test) \
        = qf.datasets.load_mnist(size, border, corners)

    # Save datafile
    outfile = 'mnist_{}x{}.npz'.format(size, size)
    np.savez(outfile, x_train=x_train_rescaled,
             y_train=y_train, x_test=x_test_rescaled, y_test=y_test)

    # Save a few examples as pngs so we can check this all worked
    output_size = (28*4, 28*4)
    for n in random.sample(range(60000), k=samples):

        digit = y_train[n]

        img = Image.fromarray(x_train[n])
        img = img.resize(output_size)
        img.save('mnist_{}_{}.png'.format(n, digit))

        img = Image.fromarray(x_train_rescaled[n])
        img = img.resize(output_size)
        img.save('mnist_{}_{}_{}x{}.png'.format(n, digit, size, size))


if __name__ == "__main__":
    _cli()
