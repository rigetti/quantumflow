#!/usr/bin/env python

# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow Examples
"""


import quantumflow as qf


def prepare_w4():
    """
    Prepare a 4-qubit W state using sqrt(iswaps) and local gates
    """
    circ = qf.Circuit()
    circ += qf.X(1)

    circ += qf.ISWAP(1, 2) ** 0.5
    circ += qf.S(2)
    circ += qf.Z(2)

    circ += qf.ISWAP(2, 3) ** 0.5
    circ += qf.S(3)
    circ += qf.Z(3)

    circ += qf.ISWAP(0, 1) ** 0.5
    circ += qf.S(0)
    circ += qf.Z(0)

    ket = circ.run()

    return ket


if __name__ == "__main__":
    def main():
        """CLI"""
        print(prepare_w4.__doc__)
        print('states amplitudes')
        ket = prepare_w4()
        qf.print_state(ket)
    main()
