"""
QuantumFlow: prototype simulator of gate-based quantum computers.
"""

from quantumflow.config import version as __version__       # noqa: F401
from quantumflow import backend                             # noqa: F401
from quantumflow.cbits import *                             # noqa: F401,F403
from quantumflow.qubits import *                            # noqa: F401,F403
from quantumflow.states import *                            # noqa: F401,F403
from quantumflow.ops import *                               # noqa: F401,F403
from quantumflow.stdops import *                            # noqa: F401,F403
from quantumflow.gates import *                             # noqa: F401,F403
from quantumflow.stdgates import *                          # noqa: F401,F403
from quantumflow.channels import *                          # noqa: F401,F403
from quantumflow.circuits import *                          # noqa: F401,F403
from quantumflow.programs import *                          # noqa: F401,F403
from quantumflow.parser import *                            # noqa: F401,F403
from quantumflow.decompositions import *                    # noqa: F401,F403
from quantumflow.measures import *                          # noqa: F401,F403
from quantumflow.dagcircuit import *                        # noqa: F401,F403
from quantumflow.visualization import *                     # noqa: F401,F403
import quantumflow.forest                                   # noqa: F401
import quantumflow.datasets                                 # noqa: F401
# Fin GEC 2018
