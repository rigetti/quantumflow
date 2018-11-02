#!/usr/bin/env python

# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Standard QuantumFlow Benchmark.

A simple benchmark of QuantumFlow performance, measured in GOPS (Gate
Operations Per Second).
"""

import argparse
import random
import timeit

import quantumflow as qf

__version__ = qf.__version__
__description__ = 'Simple benchmark of QuantumFlow performance' + \
                    'measured in GOPS (Gate Operations Per Second)'


GATES = 100
REPS = 1
QUBITS = 16


def benchmark(N, gates):
    """Create and run a circuit with N qubits and given number of gates"""
    qubits = list(range(0, N))
    ket = qf.zero_state(N)

    for n in range(0, N):
        ket = qf.H(n).run(ket)

    for _ in range(0, (gates-N)//3):
        qubit0, qubit1 = random.sample(qubits, 2)
        ket = qf.X(qubit0).run(ket)
        ket = qf.T(qubit1).run(ket)
        ket = qf.CNOT(qubit0, qubit1).run(ket)

    return ket.vec.tensor


def benchmark_gops(N, gates, reps):
    """Return benchmark performance in GOPS (Gate operations per second)"""
    t = timeit.timeit(lambda: benchmark(N, gates), number=reps)
    gops = (GATES*REPS)/t
    gops = int((gops * 100) + 0.5) / 100.0
    return gops


# ---------- Command Line Interface ----------
def _cli():

    parser = argparse.ArgumentParser(description=__description__)

    parser.add_argument('--version', action='version', version=__version__)

    parser.add_argument('qubits', nargs='+', type=int, default=[QUBITS])

    # Run command
    opts = vars(parser.parse_args())
    qubits = opts.pop('qubits')

    print("# N\tGOPS")
    for N in qubits:
        gops = benchmark_gops(N, GATES, REPS)
        print(N, '\t', gops)


if __name__ == '__main__':
    _cli()
