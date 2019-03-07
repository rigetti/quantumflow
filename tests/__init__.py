
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Unit tests for QuantumFlow
"""
import sys
import pytest
import shutil

from quantumflow.config import TOLERANCE
import quantumflow.backend as bk

ALMOST_ZERO = pytest.approx(0.0, abs=TOLERANCE)
ALMOST_ONE = pytest.approx(1.0)

REPS = 16   # Repetitions

skip_tensorflow = pytest.mark.skipif(
    bk.BACKEND == 'tensorflow',
    reason="Unsupported backend")

tensorflow_only = pytest.mark.skipif(
    bk.BACKEND != 'tensorflow',
    reason="Unsupported backend")

tensorflow2_only = pytest.mark.skipif(
    bk.BACKEND != 'tensorflow2',
    reason="Unsupported backend")

eager_only = pytest.mark.skipif(
    bk.BACKEND != 'eager',
    reason="Unsupported backend")

skip_windows = pytest.mark.skipif(
    sys.platform == 'win32',
    reason="Does not run on windows")

skip_torch = pytest.mark.skipif(
    bk.BACKEND == 'torch',
    reason="Unsupported backend")

skip_unless_pdflatex = pytest.mark.skipif(
    shutil.which('pdflatex') is None or shutil.which('pdftocairo') is None,
    reason='Necessary external dependencies not installed')
