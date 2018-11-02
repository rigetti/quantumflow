
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Unit tests for quantumflow.backend
"""

import numpy as np

import quantumflow.backend as bk
from . import ALMOST_ZERO

if bk.BACKEND == 'tensorflow':
    import tensorflow as tf
    tf.InteractiveSession()


def test_import():

    # Backend
    assert bk.TL
    assert bk.MAX_QUBITS
    assert bk.gpu_available
    assert bk.set_random_seed

    # Conversions
    assert bk.TENSOR
    assert bk.CTYPE
    assert bk.FTYPE
    assert bk.TensorLike
    assert bk.BKTensor
    assert bk.ccast
    assert bk.fcast
    assert bk.astensor
    assert bk.evaluate

    # Math
    assert bk.conj
    assert bk.absolute
    assert bk.minimum
    assert bk.arccos
    assert bk.exp
    assert bk.cos
    assert bk.sin
    assert bk.real
    assert bk.cis

    # Tensor
    assert bk.diag
    assert bk.reshape
    assert bk.sum
    assert bk.matmul
    assert bk.transpose
    assert bk.inner
    assert bk.outer


def test_gpu_available():
    bk.gpu_available()


def test_inner():
    v0 = np.random.normal(size=[2, 2, 2, 2]) \
        + 1.0j * np.random.normal(size=[2, 2, 2, 2])
    v1 = np.random.normal(size=[2, 2, 2, 2]) \
        + 1.0j * np.random.normal(size=[2, 2, 2, 2])
    res = np.vdot(v0, v1)

    bkres = bk.evaluate(bk.inner(bk.astensor(v0), bk.astensor(v1)))

    print(bkres)
    assert np.abs(res-bkres) == ALMOST_ZERO


def test_outer():
    s0 = np.random.normal(size=[2, 2]) + 1.0j * np.random.normal(size=[2, 2])
    s1 = np.random.normal(size=[2, 2, 2]) \
        + 1.0j * np.random.normal(size=[2, 2, 2])

    res = bk.astensorproduct(bk.outer(bk.astensor(s0), bk.astensor(s1)))
    assert bk.rank(res) == 5

    res2 = np.outer(s0, s1).reshape([2]*5)
    assert np.allclose(bk.evaluate(res), res2)


def test_absolute():
    t = bk.astensor([-2.25 + 4.75j, -3.25 + 5.75j])
    t = bk.absolute(t)

    assert np.allclose(bk.evaluate(t), [5.25594902, 6.60492229])

    t = bk.astensor([-2.25 + 4.75j])
    t = bk.absolute(t)
    print(bk.evaluate(t))
    assert np.allclose(bk.evaluate(t), [5.25594902])


def test_random_seed():
    # Also tested indirectly by test_config::test_seed
    # But that doesn't get captured by coverage tool
    bk.set_random_seed(42)


def test_size():
    """size(tensor) should return the number of elements"""
    t = bk.astensor([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]])
    assert bk.size(t) == 16


def test_real_imag():
    tensor = bk.astensor([1.0 + 2.0j, 0.5 - 0.2j])
    t = bk.real(tensor)
    t = bk.evaluate(t)
    assert np.allclose(bk.evaluate(bk.real(tensor)), [1.0, 0.5])
    assert np.allclose(bk.evaluate(bk.imag(tensor)), [2.0, -0.2])


def test_trace():
    tensor = bk.astensor(np.asarray([[1, 0, 0, 0],
                                     [0, -1, 0, 0],
                                     [0, 0, 2.7, 1],
                                     [0, 0, 1, 0.3j]]))
    tensor = bk.reshape(tensor, (4, 4))  # FIXME astensor should not reshape
    tr = bk.evaluate(bk.trace(tensor))
    print(tr)

    assert tr - (2.7+0.3j) == ALMOST_ZERO


def test_productdiag():
    t = bk.astensorproduct([[0., 0., 0., 6.],
                            [0., 1., 0., 0.],
                            [0., 0., 2., 1.],
                            [4., 0., 1., 3.]])
    print(t.shape)
    print(bk.productdiag(t).shape)
    assert np.allclose(bk.evaluate(bk.productdiag(t)), [[0, 1], [2, 3]])


# fin
