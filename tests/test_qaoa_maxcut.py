
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Unittests for qaoa_maxcut.py"""

from . import tensorflow_only


@tensorflow_only
def test_maxcut_qaoa():
    """Test maxcut_qaoa"""
    # Late import so tox tests will run without installing tensorflow
    from examples.qaoa_maxcut import maxcut_qaoa

    ratio, opt_beta, opt_gamma = maxcut_qaoa([[0, 1], [1, 2], [1, 3]])
    assert ratio > 0.95
    assert opt_beta is not None
    assert opt_gamma is not None
