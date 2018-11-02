
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow Configuration
"""

import os
import random
import logging

# TODO:  Device environment variables. Document
# DOCME: docstrings


_PREFIX = 'QUANTUMFLOW_'  # Environment variable prefix


# ==== Version number ====
try:
    from quantumflow.version import version
except ImportError:                           # pragma: no cover
    # package is not installed
    version = "?.?.?"


# ==== Logging Level ====
# Set logging level (CRITICAL, ERROR, WARNING (default), INFO, DEBUG)
# https://docs.python.org/3.6/library/logging.html

logging.getLogger('quantumflow').addHandler(logging.StreamHandler())
_LOGLEVEL = os.getenv(_PREFIX + 'LOG', None)
if _LOGLEVEL is not None:
    logging.getLogger('quantumflow').setLevel(_LOGLEVEL)


# ==== Tensor Library Backend ====

DEFAULT_BACKEND = 'numpy'
BACKENDS = ('tensorflow', 'eager', 'torch', 'numpy')

# Environment variable override
BACKEND = os.getenv(_PREFIX + 'BACKEND', DEFAULT_BACKEND)
if BACKEND not in BACKENDS:  # pragma: no cover
    raise ValueError('Unknown backend: QF_BACKEND={}'.format(BACKEND))
logging.getLogger(__name__).info("QuantumFlow Backend: %s", BACKEND)


# ==== TOLERANCE ====
TOLERANCE = 1e-6
"""Tolerance used in various floating point comparisons"""


# ==== Random Seed ====
_ENVSEED = os.getenv(_PREFIX + 'SEED', None)
SEED = int(_ENVSEED) if _ENVSEED is not None else None
if SEED is not None:
    random.seed(SEED)  # pragma: no cover
