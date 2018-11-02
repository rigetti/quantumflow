#!/usr/bin/env python

# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow examples
"""

import quantumflow as qf


def prepare_w16():
    """
    Prepare a 16-qubit W state using sqrt(iswaps) and local gates,
    respecting linear topology

    """
    ket = qf.zero_state(16)
    circ = w16_circuit()
    ket = circ.run(ket)
    return ket


def w16_circuit() -> qf.Circuit:
    """
    Return a circuit that prepares the the 16-qubit W state using\
    sqrt(iswaps) and local gates, respecting linear topology
    """

    gates = [
        qf.X(7),

        qf.ISWAP(7, 8) ** 0.5,
        qf.S(8),
        qf.Z(8),

        qf.SWAP(7, 6),
        qf.SWAP(6, 5),
        qf.SWAP(5, 4),

        qf.SWAP(8, 9),
        qf.SWAP(9, 10),
        qf.SWAP(10, 11),


        qf.ISWAP(4, 3) ** 0.5,
        qf.S(3),
        qf.Z(3),

        qf.ISWAP(11, 12) ** 0.5,
        qf.S(12),
        qf.Z(12),

        qf.SWAP(3, 2),
        qf.SWAP(4, 5),
        qf.SWAP(11, 10),
        qf.SWAP(12, 13),


        qf.ISWAP(2, 1) ** 0.5,
        qf.S(1),
        qf.Z(1),

        qf.ISWAP(5, 6) ** 0.5,
        qf.S(6),
        qf.Z(6),

        qf.ISWAP(10, 9) ** 0.5,
        qf.S(9),
        qf.Z(9),

        qf.ISWAP(13, 14) ** 0.5,
        qf.S(14),
        qf.Z(14),


        qf.ISWAP(1, 0) ** 0.5,
        qf.S(0),
        qf.Z(0),

        qf.ISWAP(2, 3) ** 0.5,
        qf.S(3),
        qf.Z(3),

        qf.ISWAP(5, 4) ** 0.5,
        qf.S(4),
        qf.Z(4),

        qf.ISWAP(6, 7) ** 0.5,
        qf.S(7),
        qf.Z(7),

        qf.ISWAP(9, 8) ** 0.5,
        qf.S(8),
        qf.Z(8),

        qf.ISWAP(10, 11) ** 0.5,
        qf.S(11),
        qf.Z(11),

        qf.ISWAP(13, 12) ** 0.5,
        qf.S(12),
        qf.Z(12),

        qf.ISWAP(14, 15) ** 0.5,
        qf.S(15),
        qf.Z(15),
        ]
    circ = qf.Circuit(gates)

    return circ


if __name__ == "__main__":
    def main():
        """CLI"""
        print(prepare_w16.__doc__)
        print('states           : probabilities')
        ket = prepare_w16()
        qf.print_probabilities(ket)
    main()
