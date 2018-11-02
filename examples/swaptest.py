#!/usr/bin/env python

# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QauntumFlow Examples
"""

import quantumflow as qf
import quantumflow.backend as bk


def swap_test(ket, q0: qf.Qubit, q1: qf.Qubit, q2: qf.Qubit) -> bk.BKTensor:
    """
    Apply a Swap Test to measure fidelity between qubits q1 and q2.

    Qubit q0 is an ancilla, which should be zero state initially. The qubits
    cannot be initially entangled.
    """
    circ = qf.Circuit()
    circ += qf.H(q0)
    circ += qf.CSWAP(q0, q1, q2)
    circ += qf.H(q0)
    circ += qf.P0(q0)    # Measure
    ket = circ.run(ket)

    fid = bk.fcast(2)*ket.norm() - bk.fcast(1.0)  # return fidelity
    return fid


if __name__ == "__main__":
    def main():
        """CLI"""
        print(swap_test.__doc__)

        print('Randomly generating two 1-qubit states...')

        ket0 = qf.zero_state([0])
        ket1 = qf.random_state([1])
        ket2 = qf.random_state([2])

        ket = qf.join_states(ket0, ket1, ket2)

        fid = qf.state_fidelity(ket1, ket2.relabel([1]))
        st_fid = swap_test(ket, 0, 1, 2)

        print('Fidelity:               ', qf.asarray(fid))
        print('Fidelity from swap test:', qf.asarray(st_fid))
    main()
