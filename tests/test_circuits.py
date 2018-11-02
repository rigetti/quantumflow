# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Unittests for quantumflow.circuits"""

from math import pi
import numpy as np

import pytest

import quantumflow as qf
from quantumflow.utils import bitlist_to_int, int_to_bitlist

from . import ALMOST_ZERO


def true_ket():
    # Adapted from referenceQVM
    wf_true = np.array(
        [0.00167784 + 1.00210180e-05*1j, 0.50000000 - 4.99997185e-01*1j,
         0.50000000 - 4.99997185e-01*1j, 0.00167784 + 1.00210180e-05*1j])
    return qf.State(wf_true.reshape((2, 2)))


def test_asgate():
    circ = qf.zyz_circuit(0.1, 2.2, 0.5, 0)
    print(">>>>", circ, len(circ.elements))
    assert qf.gates_close(circ.asgate(), qf.ZYZ(0.1, 2.2, 0.5))


def test_str():
    circ = qf.zyz_circuit(0.1, 2.2, 0.5, [0])
    print(circ)
    # TODO Expand


def test_name():
    assert qf.Circuit().name == 'CIRCUIT'


def test_qaoa_circuit():
    circ = qf.Circuit()
    circ += qf.RY(pi/2, 0)
    circ += qf.RX(pi, 0)
    circ += qf.RY(pi/2, 1)
    circ += qf.RX(pi, 1)
    circ += qf.CNOT(0, 1)
    circ += qf.RX(-pi/2, 1)
    circ += qf.RY(4.71572463191, 1)
    circ += qf.RX(pi/2, 1)
    circ += qf.CNOT(0, 1)
    circ += qf.RX(-2*2.74973750579, 0)
    circ += qf.RX(-2*2.74973750579, 1)

    ket = qf.zero_state(2)
    ket = circ.run(ket)

    assert qf.states_close(ket, true_ket())


def test_extend():
    circ = qf.Circuit()
    circ1 = qf.Circuit()
    circ2 = qf.Circuit()
    circ1 += qf.RY(pi/2, 0)
    circ1 += qf.RX(pi, 0)
    circ1 += qf.RY(pi/2, 1)
    circ1 += qf.RX(pi, 1)
    circ1 += qf.CNOT(0, 1)
    circ2 += qf.RX(-pi/2, 1)
    circ2 += qf.RY(4.71572463191, 1)
    circ2 += qf.RX(pi/2, 1)
    circ2 += qf.CNOT(0, 1)
    circ2 += qf.RX(-2*2.74973750579, 0)
    circ2 += qf.RX(-2*2.74973750579, 1)
    circ.extend(circ1)
    circ.extend(circ2)

    ket = qf.zero_state(2)
    ket = circ.run(ket)

    assert qf.states_close(ket, true_ket())


def test_qaoa_circuit_turns():
    circ = qf.Circuit()
    circ += qf.TY(1/2, 0)
    circ += qf.TX(1, 0)
    circ += qf.TY(1/2, 1)
    circ += qf.TX(1, 1)
    circ += qf.CNOT(0, 1)
    circ += qf.TX(-1/2, 1)
    circ += qf.TY(4.71572463191/pi, 1)
    circ += qf.TX(1/2, 1)
    circ += qf.CNOT(0, 1)
    circ += qf.TX(-2*2.74973750579/pi, 0)
    circ += qf.TX(-2*2.74973750579/pi, 1)

    ket = qf.zero_state(2)
    ket = circ.run(ket)

    assert qf.states_close(ket, true_ket())


def test_circuit_wires():
    circ = qf.Circuit()
    circ += qf.TY(1/2, 0)
    circ += qf.TX(1, 10)
    circ += qf.TY(1/2, 1)
    circ += qf.TX(1, 1)
    circ += qf.CNOT(0, 4)

    bits = circ.qubits
    assert bits == (0, 1, 4, 10)


def test_inverse():
    # Random circuit
    circ = qf.Circuit()
    circ += qf.TY(1/2, 0)
    circ += qf.H(0)
    circ += qf.TY(1/2, 1)
    circ += qf.TX(1.23123, 1)
    circ += qf.CNOT(0, 1)
    circ += qf.TX(-1/2, 1)
    circ += qf.TY(4.71572463191/pi, 1)
    circ += qf.CNOT(0, 1)
    circ += qf.TX(-2*2.74973750579/pi, 0)
    circ += qf.TX(-2*2.74973750579/pi, 1)

    circ_inv = circ.H

    ket = circ.run()
    qf.print_state(ket)

    ket = circ_inv.run(ket)
    qf.print_state(ket)

    print(ket.qubits)
    print(true_ket().qubits)
    assert qf.states_close(ket, qf.zero_state(2))

    ket = qf.zero_state(2)
    circ.extend(circ_inv)
    ket = circ.run(ket)
    assert qf.states_close(ket, qf.zero_state(2))


def test_implicit_state():
    circ = qf.Circuit()
    circ += qf.TY(1/2, 0)
    circ += qf.H(0)
    circ += qf.TY(1/2, 1)

    ket = circ.run()    # Implicit state
    assert len(ket.qubits) == 2

    circ += qf.TY(1/2, 'namedqubit')
    with pytest.raises(TypeError):
        # Should fail because qubits aren't sortable, so no standard ordering
        circ.run()    # Implicit state


def test_elements():
    circ = qf.Circuit()
    circ1 = qf.Circuit()
    circ2 = qf.Circuit()
    circ1 += qf.RY(pi/2, 0)
    circ1 += qf.RX(pi, 0)
    circ1 += qf.RY(pi/2, 1)
    circ1 += qf.RX(pi, 1)
    circ1 += qf.CNOT(0, 1)
    circ2 += qf.RX(-pi/2, 1)
    circ2 += qf.RY(4.71572463191, 1)
    circ2 += qf.RX(pi/2, 1)
    circ2 += qf.CNOT(0, 1)
    circ2 += qf.RX(-2*2.74973750579, 0)
    circ2 += qf.RX(-2*2.74973750579, 1)
    circ += circ1
    circ.extend(circ2)

    gates = list(circ.elements)
    assert len(gates) == 11
    assert circ.size() == 11
    assert gates[4].name == 'CNOT'


def test_qft():
    circ = qf.Circuit()
    circ += qf.X(2)
    circ.extend(qf.qft_circuit([0, 1, 2]))

    ket = qf.zero_state(3)
    ket = circ.run(ket)

    true_qft = qf.State([0.35355339+0.j, 0.25000000+0.25j,
                         0.00000000+0.35355339j, -0.25000000+0.25j,
                         -0.35355339+0.j, -0.25000000-0.25j,
                         0.00000000-0.35355339j, 0.25000000-0.25j])

    assert qf.states_close(ket, true_qft)


def test_create():
    gen = [qf.H(i) for i in range(8)]

    circ1 = qf.Circuit(list(gen))
    circ1.run(qf.zero_state(8))

    circ2 = qf.Circuit(gen)
    circ2.run(qf.zero_state(8))

    circ3 = qf.Circuit(qf.H(i) for i in range(8))
    circ3.run(qf.zero_state(8))


def test_add():
    circ = qf.Circuit()
    circ += qf.H(0)
    circ += qf.H(1)
    circ += qf.H(1)

    assert len(list(circ)) == 3

    circ = circ + circ
    assert len(list(circ.elements)) == 6

    circ += circ
    assert len(list(circ.elements)) == 12


def test_ccnot_circuit():
    ket0 = qf.random_state(3)
    ket1 = qf.CCNOT(0, 1, 2).run(ket0)
    ket2 = qf.ccnot_circuit([0, 1, 2]).run(ket0)
    assert qf.states_close(ket1, ket2)

    with pytest.raises(ValueError):
        qf.ccnot_circuit([0, 1, 2, 3])


def test_ccnot_circuit_evolve():
    rho0 = qf.random_state(3).asdensity()
    rho1 = qf.CCNOT(0, 1, 2).evolve(rho0)
    rho2 = qf.ccnot_circuit([0, 1, 2]).evolve(rho0)
    assert qf.densities_close(rho1, rho2)

    qf.ccnot_circuit([0, 1, 2]).evolve()


def test_circuit_aschannel():
    rho0 = qf.random_state(3).asdensity()
    rho1 = qf.CCNOT(0, 1, 2).evolve(rho0)

    chan = qf.ccnot_circuit([0, 1, 2]).aschannel()
    rho2 = chan.evolve(rho0)

    assert qf.densities_close(rho1, rho2)


def test_control_circuit():
    ccnot = qf.control_circuit([0, 1], qf.X(2))
    ket0 = qf.random_state(3)
    ket1 = qf.CCNOT(0, 1, 2).run(ket0)
    ket2 = ccnot.run(ket0)
    assert qf.states_close(ket1, ket2)


def test_phase_estimation_circuit():
    N = 8
    phase = 1/4
    gate = qf.RZ(-4*np.pi*phase, N)
    circ = qf.phase_estimation_circuit(gate, range(N))
    res = circ.run().measure()[0:N]
    est_phase = bitlist_to_int(res) / 2**N
    assert phase-est_phase == ALMOST_ZERO

    phase = 12/256
    gate = qf.RZ(-4*np.pi*phase, N)
    circ = qf.phase_estimation_circuit(gate, range(N))
    res = circ.run().measure()[0:N]
    est_phase = bitlist_to_int(res) / 2**N
    assert phase-est_phase == ALMOST_ZERO

    gate = qf.ZZ(-4*phase, N, N+1)
    circ = qf.phase_estimation_circuit(gate, range(N))
    res = circ.run().measure()[0:N]
    est_phase = bitlist_to_int(res) / 2**N
    assert phase-est_phase == ALMOST_ZERO

    with pytest.raises(ValueError):
        # Gate and output qubits overlap
        circ = qf.phase_estimation_circuit(gate, range(N+1))


def test_addition_circuit():
    # Two bit addition circuit
    circ = qf.addition_circuit([0, 1], [2, 3], [4, 5])

    for c0 in range(0, 2):
        for a0 in range(0, 4):
            for a1 in range(0, 4):
                expected = a0 + a1 + c0
                b0 = int_to_bitlist(a0, 2)
                b1 = int_to_bitlist(a1, 2)
                bc = [c0]
                bits = tuple(b0 + b1 + bc + [0])

                state = np.zeros(shape=[2]*6)
                state[bits] = 1
                ket = qf.State(state)

                ket = circ.run(ket)
                bits = ket.measure()
                res = bits[[5, 2, 3]]
                res = bitlist_to_int(res)

                print(c0, a0, a1, expected, res)
                assert res == expected

    # Three bit addition circuit
    circ = qf.addition_circuit([0, 1, 2], [3, 4, 5], [6, 7])
    for c0 in range(0, 2):
        for a0 in range(0, 8):
            for a1 in range(0, 8):
                expected = a0 + a1 + c0
                b0 = int_to_bitlist(a0, 3)
                b1 = int_to_bitlist(a1, 3)
                bc = [c0]
                bits = tuple(b0 + b1 + bc + [0])

                state = np.zeros(shape=[2]*8)
                state[bits] = 1
                ket = qf.State(state)

                ket = circ.run(ket)
                bits = ket.measure()
                res = bits[[7, 3, 4, 5]]
                res = bitlist_to_int(res)

                print(c0, a0, a1, expected, res)
                assert res == expected

    with pytest.raises(ValueError):
        qf.addition_circuit([0, 1, 2], [3, 4, 5, 6], [7, 8])

    with pytest.raises(ValueError):
        qf.addition_circuit([0, 1, 2], [3, 4, 5], [6, 7, 8])


def test_ghz_circuit():
    N = 12
    qubits = list(range(N))
    circ = qf.ghz_circuit(qubits)
    circ.run()


def test_zyz_circuit():
    gate0 = qf.zyz_circuit(0.1, 0.3, 0.2, 0).asgate()
    gate1 = qf.ZYZ(0.1, 0.3, 0.2, q0=0)
    assert qf.gates_close(gate0, gate1)


def test_map_gate():
    circ = qf.map_gate(qf.X(), [[0], [1], [2]])
    assert circ.elements[1].qubits[0] == 1

    circ = qf.map_gate(qf.CNOT(), [[0, 1], [1, 2]])
    assert circ.elements[1].qubits == (1, 2)


def test_count():
    circ = qf.Circuit()
    circ += qf.H(0)
    circ += qf.H(1)
    circ += qf.H(1)
    op_count = qf.count_operations(circ)
    assert op_count == {qf.H: 3}

    circ = qf.addition_circuit([0, 1, 2], [3, 4, 5], [6, 7])
    op_count = qf.count_operations(circ)
    assert op_count == {qf.CNOT: 13, qf.CCNOT: 6}
