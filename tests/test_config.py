
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import os
# from quantumflow.config import *

from . import skip_windows


@skip_windows
def test_random_seed():
    code = ('QUANTUMFLOW_SEED=42 python -c "import quantumflow as qf;'
            ' assert qf.config.SEED == 42"')
    assert os.system(code) == 0


@skip_windows
def test_logger():

    cmd = ('QUANTUMFLOW_LOG=INFO python -c \''
           'import quantumflow as qf;'
           'import logging;'
           'logger = logging.getLogger("quantumflow");'
           'assert logger.level == 20\'')

    print(cmd)

    assert os.system(cmd) == 0
