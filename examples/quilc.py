#!/usr/bin/env python

# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: Example use of `quilc` to compile to Rigetti native gate set.
"""

import quantumflow as qf

# Construct a rippler adder circuit to add two N-qubit registers
N = 2
circ = qf.addition_circuit(list(range(N)),          # first register
                           list(range(N, 2 * N)),   # second register
                           [2 * N, 2 * N + 1])       # carry bits

# Render pre-compilation circuit
qf.circuit_to_image(circ).show()

# Compile circuit
compiler = qf.forest.get_compiler(circ.qubit_nb)
prog = qf.forest.circuit_to_pyquil(circ)
with qf.pyquil.local_qvm():
    prog = compiler.quil_to_native_quil(prog)

# Render post-compilation circuit
circ = qf.forest.pyquil_to_circuit(prog)
qf.circuit_to_image(circ).show()
