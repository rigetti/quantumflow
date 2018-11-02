
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Unit tests for quantumflow.meta
"""


import io
import subprocess

from quantumflow import meta


def test_print_versions():
    out = io.StringIO()
    meta.print_versions(out)
    print(out)


def test_print_versions_main():
    rval = subprocess.call(['python', '-m', 'quantumflow.meta'])
    assert rval == 0
