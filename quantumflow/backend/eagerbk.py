
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Tensorflow eager mode backend for QuantumFlow
"""

# FIXME: use "with device" rather than .gpu()?

import math
import os

import tensorflow as tf
from tensorflow.contrib.eager.python import tfe
tfe.enable_eager_execution()

from tensorflow import transpose, minimum                       # noqa: F401
from tensorflow import exp, cos, sin, reshape                   # noqa: F401
from tensorflow import conj, real, imag, sqrt, matmul, trace    # noqa: F401
from tensorflow import abs as absolute                          # noqa: F401
from tensorflow import diag_part as diag                        # noqa: F401

from .tensorflowbk import (                                     # noqa: F401
    rank, sum, ccast, CTYPE, FTYPE, TENSOR, BKTensor, TensorLike, inner,
    outer, gpu_available, set_random_seed, cis, arccos, getitem, size,
    productdiag, EINSUM_SUBSCRIPTS, einsum, tensormul)


DEFAULT_DEVICE = 'gpu' if gpu_available() else 'cpu'
DEVICE = os.getenv('QUANTUMFLOW_DEVICE', DEFAULT_DEVICE)
assert DEVICE in {'cpu', 'gpu'}


TL = tf


# The effective maximum size is 28 qubits for a 16 GiB GPU (probably 29 qubits
# for 24 GiB)
MAX_QUBITS = 32


def evaluate(tensor: BKTensor) -> TensorLike:
    """Return the value of a tensor"""
    return tensor.numpy()


def fcast(value: float) -> TensorLike:
    """Cast to float tensor"""
    newvalue = tf.cast(value, FTYPE)
    if DEVICE == 'gpu':
        newvalue = newvalue.gpu()  # Why is this needed?  # pragma: no cover
    return newvalue


def astensor(array: TensorLike) -> BKTensor:
    """Convert to product tensor"""
    tensor = tf.convert_to_tensor(array, dtype=CTYPE)
    if DEVICE == 'gpu':
        tensor = tensor.gpu()  # pragma: no cover

    # size = np.prod(np.array(tensor.get_shape().as_list()))
    N = int(math.log2(size(tensor)))
    tensor = tf.reshape(tensor, ([2]*N))

    return tensor


def astensorproduct(array: TensorLike) -> BKTensor:
    """Returns: array as a product tensor"""
    return astensor(array)
