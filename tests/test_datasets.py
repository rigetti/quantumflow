
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import pytest

import quantumflow as qf


def test_load_stdgraphs():
    graphs = qf.datasets.load_stdgraphs(6)
    assert len(graphs) == 100


def test_load_mnist():
    (x_train, y_train, x_test, y_test) = \
        qf.datasets.load_mnist(size=4, blank_corners=True, nums=[0, 1])

    assert x_train.shape == (12665, 4, 4)
    assert y_train.shape == (12665,)
    assert y_test.shape == (2115,)
    assert x_test.shape == (2115, 4, 4)


def test_exceptions():
    with pytest.raises(ValueError):
        qf.datasets.load_stdgraphs(1000)
