
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Unit tests for quantumflow tools.
"""

import subprocess


def test_benchmark_main():
    rval = subprocess.call(['tools/benchmark.py', '2'])
    assert rval == 0

# TODO: test other scripts
