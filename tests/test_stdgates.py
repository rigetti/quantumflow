
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Unit tests for quantumflow.stdgates
"""

import random
from math import pi
import numpy as np

import quantumflow as qf
from . import ALMOST_ONE

from . import REPS


def test_I():
    assert np.allclose(qf.I().vec.asarray(), np.eye(2))
    assert np.allclose(qf.I(0, 1).vec.asarray().reshape((4, 4)), np.eye(4))


def test_unitary_1qubit():
    assert qf.almost_unitary(qf.X())
    assert qf.almost_unitary(qf.Y())
    assert qf.almost_unitary(qf.Z())
    assert qf.almost_unitary(qf.H())
    assert qf.almost_unitary(qf.S())
    assert qf.almost_unitary(qf.T())


def test_unitary_2qubit():
    assert qf.almost_unitary(qf.CZ())
    assert qf.almost_unitary(qf.CNOT())
    assert qf.almost_unitary(qf.SWAP())
    assert qf.almost_unitary(qf.ISWAP())


def test_unitary_3qubit():
    assert qf.almost_unitary(qf.CCNOT())
    assert qf.almost_unitary(qf.CNOT())
    assert qf.almost_unitary(qf.CSWAP())


def test_parametric_gates1():
    for _ in range(REPS):
        theta = random.uniform(-4*pi, +4*pi)
        assert qf.almost_unitary(qf.RX(theta))
        assert qf.almost_unitary(qf.RY(theta))
        assert qf.almost_unitary(qf.RZ(theta))

    for _ in range(REPS):
        theta = random.uniform(-4*pi, +4*pi)
        assert qf.almost_unitary(qf.TX(theta))
        assert qf.almost_unitary(qf.TY(theta))
        assert qf.almost_unitary(qf.TZ(theta))

    for _ in range(REPS):
        theta = random.uniform(-4*pi, +4*pi)
        assert qf.almost_unitary(qf.CPHASE00(theta))
        assert qf.almost_unitary(qf.CPHASE01(theta))
        assert qf.almost_unitary(qf.CPHASE10(theta))
        assert qf.almost_unitary(qf.CPHASE(theta))
        assert qf.almost_unitary(qf.PSWAP(theta))

    assert qf.gates_close(qf.I(), qf.I())
    assert qf.gates_close(qf.RX(pi), qf.X())
    assert qf.gates_close(qf.RY(pi), qf.Y())
    assert qf.gates_close(qf.RZ(pi), qf.Z())


def test_cnot():
    # three cnots same as one swap
    gate = qf.identity_gate(2)
    gate = qf.CNOT(1, 0) @ gate
    gate = qf.CNOT(0, 1) @ gate
    gate = qf.CNOT(1, 0) @ gate
    res = qf.asarray(qf.inner_product(gate.vec, qf.SWAP().vec))
    assert abs(res)/4 == ALMOST_ONE


def test_CZ():
    ket = qf.zero_state(2)
    ket = qf.CZ(0, 1).run(ket)
    assert ket.vec.asarray()[0, 0] == ALMOST_ONE

    ket = qf.X(0).run(ket)
    ket = qf.X(1).run(ket)
    ket = qf.CZ(0, 1).run(ket)
    assert -ket.vec.asarray()[1, 1] == ALMOST_ONE


def test_cnot_reverse():
    # Hadamards reverse control on CNOT
    gate0 = qf.identity_gate(2)
    gate0 = qf.H(0) @ gate0
    gate0 = qf.H(1) @ gate0
    gate0 = qf.CNOT(1, 0) @ gate0
    gate0 = qf.H(0) @ gate0
    gate0 = qf.H(1) @ gate0

    assert qf.gates_close(qf.CNOT(), gate0)


def test_ccnot():
    ket = qf.zero_state(3)
    ket = qf.CCNOT(0, 1, 2).run(ket)
    assert ket.vec.asarray()[0, 0, 0] == ALMOST_ONE

    ket = qf.X(1).run(ket)
    ket = qf.CCNOT(0, 1, 2).run(ket)
    assert ket.vec.asarray()[0, 1, 0] == ALMOST_ONE

    ket = qf.X(0).run(ket)
    ket = qf.CCNOT(0, 1, 2).run(ket)
    assert ket.vec.asarray()[1, 1, 1] == ALMOST_ONE


def test_cswap():
    ket = qf.zero_state(3)
    ket = qf.X(1).run(ket)
    ket = qf.CSWAP(0, 1, 2).run(ket)
    assert ket.vec.asarray()[0, 1, 0] == ALMOST_ONE

    ket = qf.X(0).run(ket)
    ket = qf.CSWAP(0, 1, 2).run(ket)
    assert ket.vec.asarray()[1, 0, 1] == ALMOST_ONE


def test_phase():
    gate = qf.T(0) @ qf.T(0)

    assert qf.gates_close(gate, gate)

    assert qf.gates_close(gate, qf.S())
    assert qf.gates_close(qf.S(), qf.PHASE(pi/2))
    assert qf.gates_close(qf.T(), qf.PHASE(pi/4))

    # PHASE and RZ are the same up to a global phase.
    for _ in range(REPS):
        theta = random.uniform(-4*pi, +4*pi)
        assert qf.gates_close(qf.RZ(theta), qf.PHASE(theta))

    # Causes a rounding error that can result in NANs if not corrected for.
    theta = -2.5700302313621375
    assert qf.gates_close(qf.RZ(theta), qf.PHASE(theta))


def test_hadamard():
    gate = qf.I()
    gate = qf.RZ(pi/2, 0) @ gate
    gate = qf.RX(pi/2, 0) @ gate
    gate = qf.RZ(pi/2, 0) @ gate

    res = qf.asarray(qf.inner_product(gate.vec, qf.H().vec))

    assert abs(res)/2 == ALMOST_ONE


def test_piswap():
    for _ in range(REPS):
        theta = random.uniform(-4*pi, +4*pi)
        assert qf.almost_unitary(qf.PISWAP(theta))

    for _ in range(REPS):
        theta = random.uniform(0, + pi)

    assert qf.gates_close(qf.PISWAP(0), qf.identity_gate(2))

    assert qf.gates_close(qf.PISWAP(pi/4), qf.ISWAP())


def test_pswap():
    for _ in range(REPS):
        theta = random.uniform(-4*pi, +4*pi)
        assert qf.almost_unitary(qf.PSWAP(theta))

    assert qf.gates_close(qf.SWAP(), qf.PSWAP(0))

    assert qf.gates_close(qf.ISWAP(), qf.PSWAP(pi/2))


def test_cphase_gates():
    for _ in range(REPS):
        theta = random.uniform(-4*pi, +4*pi)

        gate11 = qf.control_gate(0, qf.PHASE(theta, 1))
        assert qf.gates_close(gate11, qf.CPHASE(theta, 0, 1))

        gate01 = qf.conditional_gate(0, qf.PHASE(theta, 1), qf.I(1))
        assert qf.gates_close(gate01, qf.CPHASE01(theta))

        gate00 = qf.identity_gate(2)
        gate00 = qf.X(0) @ gate00
        gate00 = qf.X(1) @ gate00
        gate00 = gate11 @ gate00
        gate00 = qf.X(0) @ gate00
        gate00 = qf.X(1) @ gate00
        assert qf.gates_close(gate00, qf.CPHASE00(theta))

        gate10 = qf.identity_gate(2)
        gate10 = qf.X(0) @ gate10
        gate10 = qf.X(1) @ gate10
        gate10 = gate01 @ gate10
        gate10 = qf.X(0) @ gate10
        gate10 = qf.X(1) @ gate10
        assert qf.gates_close(gate10, qf.CPHASE10(theta))


def test_parametric_TX_TY_TZ():
    gate = qf.I()
    gate = qf.TZ(1/2) @ gate
    gate = qf.TX(1/2) @ gate
    gate = qf.TZ(1/2) @ gate

    assert qf.gates_close(gate, qf.H())


def test_parametric_Y():
    pseudo_hadamard = qf.TY(1.5)
    inv_pseudo_hadamard = qf.TY(0.5)
    gate = pseudo_hadamard @ inv_pseudo_hadamard
    assert qf.gates_close(gate, qf.I())


def test_parametric_Z():
    assert qf.gates_close(qf.TZ(1/4), qf.T())
    assert qf.gates_close(qf.TZ(1/2), qf.S())
    assert qf.gates_close(qf.TZ(1.0), qf.Z())


def test_inverse_self():
    # These gates are their own inverse
    gate_names = ['I', 'X', 'Y', 'Z', 'CNOT', 'SWAP', 'CCNOT',
                  'CSWAP', 'CZ']

    for name in gate_names:
        gate = qf.STDGATES[name]()
        inv = gate.H
        assert type(gate) == type(inv)

        inv = qf.Gate(gate.tensor).H
        assert qf.gates_close(gate, inv)


def test_inverse_1qubit():
    # These gate pairs are inverses of each other
    gate_names = [('S', 'S_H'), ('T', 'T_H')]

    for name0, name1 in gate_names:
        gate0 = qf.STDGATES[name0]()
        gate1 = qf.STDGATES[name1]()
        assert qf.gates_close(gate0, gate1.H)
        assert qf.gates_close(gate0.H, gate1)

        assert qf.gates_close(gate0, gate1**-1)
        assert qf.gates_close(gate0**-1, gate1)


def test_inverse_parametric_1qubit():
    gates = [qf.PHASE, qf.RX, qf.RY, qf.RZ]

    for gate in gates:
        for _ in range(REPS):
            theta = random.uniform(-4*pi, +4*pi)
            g = gate(theta)
            inv = g.H
            assert qf.gates_close(qf.I(), g @ inv)
            assert type(g) == type(inv)


def test_inverse_parametric_2qubit():
    gates = [qf.CPHASE00, qf.CPHASE01, qf.CPHASE10, qf.CPHASE, qf.PSWAP]

    for gate in gates:
        for _ in range(REPS):
            theta = random.uniform(-4*pi, +4*pi)
            g = gate(theta)
            inv = g.H
            assert qf.gates_close(qf.identity_gate(2), g @ inv)
            assert type(g) == type(inv)


def test_inverse_tgates_1qubit():
    gates = qf.TX, qf.TY, qf.TZ

    for gate in gates:
        for _ in range(REPS):
            t = random.uniform(-2, +2)
            g = gate(t)
            inv = g.H
            assert qf.gates_close(qf.I(), g @ inv)
            assert type(g) == type(inv)


def test_inverse_tgates_2qubit():
    gates = [qf.PISWAP]

    for gate in gates:
        for _ in range(REPS):
            t = random.uniform(-2, +2)
            g = gate(t)
            inv = g.H
            assert qf.gates_close(qf.identity_gate(2), g @ inv)
            assert type(g) == type(inv)


def test_CAN():
    t1 = random.uniform(-2, +2)
    t2 = random.uniform(-2, +2)
    t3 = random.uniform(-2, +2)
    gate = qf.CAN(t1, t2, t3)
    assert qf.almost_unitary(gate)
    inv = gate.H
    assert type(gate) == type(inv)
    assert qf.gates_close(qf.identity_gate(2), inv @ gate)


def test_EXCH():
    t = random.uniform(-2, +2)
    gate = qf.EXCH(t)
    assert qf.almost_unitary(gate)
    inv = gate.H
    assert type(gate) == type(inv)
    assert qf.gates_close(qf.identity_gate(2), inv @ gate)

    gate1 = qf.CANONICAL(t, t, t)
    assert qf.gates_close(gate, gate1)


def test_ZYZ():
    t0 = random.uniform(-2, +2)
    t1 = random.uniform(-2, +2)
    t2 = random.uniform(-2, +2)

    gate = qf.ZYZ(t0, t1, t2, 0)
    assert qf.almost_unitary(gate)
    inv = gate.H
    assert type(gate) == type(inv)
    assert qf.gates_close(qf.I(0), inv @ gate)


def test_XX_YY_ZZ():
    gates = [qf.XX, qf.YY, qf.ZZ]
    for gate_class in gates:
        t = random.uniform(-2, +2)
        gate = gate_class(t)
        assert qf.almost_unitary(gate)
        inv = gate.H
        assert type(gate) == type(inv)
        assert qf.gates_close(qf.identity_gate(2), gate @ inv)


def test_CH():
    # Construct a controlled Hadamard gate
    gate = qf.identity_gate(2)

    gate = qf.S(1).H @ gate
    gate = qf.H(1) @ gate
    gate = qf.T(1).H @ gate
    gate = qf.CNOT(0, 1) @ gate
    gate = qf.T(1) @ gate
    gate = qf.H(1) @ gate
    gate = qf.S(1) @ gate

    # Do nothing
    ket = qf.zero_state(2)
    ket = gate.run(ket)
    assert qf.states_close(ket, qf.zero_state(2))

    # Do nothing
    ket = qf.zero_state(2)
    ket = qf.X(0).run(ket)
    ket = gate.run(ket)

    ket = qf.H(1).run(ket)

    ket = qf.X(0).run(ket)
    assert qf.states_close(ket, qf.zero_state(2))


def test_pseudo_hadamard():
    # 1-qubit pseudo-Hadamard gates turn a cnot into a CZ
    gate = qf.identity_gate(2)
    gate = qf.TY(3/2, 1).H @ gate
    gate = qf.CNOT(0, 1) @ gate
    gate = qf.TY(3/2, 1) @ gate

    assert qf.gates_close(gate, qf.CZ())


def test_RN():
    for _ in range(REPS):
        theta = random.uniform(-2*pi, +2*pi)
        nx = random.uniform(0, 1)
        ny = random.uniform(0, 1)
        nz = random.uniform(0, 1)
        L = np.sqrt(nx**2 + ny**2 + nz**2)
        nx /= L
        ny /= L
        nz /= L
        gate = qf.RN(theta, nx, ny, nz)
        assert qf.almost_unitary(gate)

        gate2 = qf.RN(-theta, nx, ny, nz)
        assert qf.gates_close(gate.H, gate2)
        assert qf.gates_close(gate**-1, gate2)

    theta = 1.23

    gate = qf.RN(theta, 1, 0, 0)
    assert qf.gates_close(gate, qf.RX(theta))

    gate = qf.RN(theta, 0, 1, 0)
    assert qf.gates_close(gate, qf.RY(theta))

    gate = qf.RN(theta, 0, 0, 1)
    assert qf.gates_close(gate, qf.RZ(theta))

    gate = qf.RN(pi, np.sqrt(2), 0, np.sqrt(2))
    assert qf.gates_close(gate, qf.H())


def test_qaoa_circuit():
    # Kudos: Adapted from reference QVM
    wf_true = np.array(
        [0.00167784 + 1.00210180e-05*1j, 0.50000000 - 4.99997185e-01*1j,
         0.50000000 - 4.99997185e-01*1j, 0.00167784 + 1.00210180e-05*1j])
    ket_true = qf.State(wf_true.reshape((2, 2)))

    ket = qf.zero_state(2)
    ket = qf.RY(pi/2, 0).run(ket)
    ket = qf.RX(pi, 0).run(ket)
    ket = qf.RY(pi/2, 1).run(ket)
    ket = qf.RX(pi, 1).run(ket)
    ket = qf.CNOT(0, 1).run(ket)
    ket = qf.RX(-pi/2, 1).run(ket)
    ket = qf.RY(4.71572463191, 1).run(ket)
    ket = qf.RX(pi/2, 1).run(ket)
    ket = qf.CNOT(0, 1).run(ket)
    ket = qf.RX(-2*2.74973750579, 0).run(ket)
    ket = qf.RX(-2*2.74973750579, 1).run(ket)

    assert qf.states_close(ket, ket_true)


def test_gatepow():
    gates = [qf.I(), qf.X(), qf.Y(), qf.Z(), qf.H(), qf.S(), qf.T(),
             qf.PHASE(0.1), qf.RX(0.2), qf.RY(0.3), qf.RZ(0.4), qf.CZ(),
             qf.CNOT(), qf.SWAP(), qf.ISWAP(), qf.CPHASE00(0.5),
             qf.CPHASE01(0.6), qf.CPHASE10(0.6), qf.CPHASE(0.7),
             qf.PSWAP(0.15), qf.CCNOT(), qf.CSWAP(), qf.TX(2.7), qf.TY(1.2),
             qf.TZ(0.3), qf.ZYZ(3.5, 0.9, 2.1), qf.CANONICAL(0.1, 0.2, 7.4),
             qf.XX(1.8), qf.YY(0.9), qf.ZZ(0.45), qf.PISWAP(0.2),
             qf.EXCH(0.1), qf.TH(0.3)
             ]

    for gate in gates:
        assert qf.gates_close(gate.H, gate ** -1)

    for gate in gates:
        sqrt_gate = gate ** (1/2)
        two_gate = sqrt_gate @ sqrt_gate
        assert qf.gates_close(gate, two_gate)

    for gate in gates:
        gate0 = gate ** 0.3
        gate1 = gate ** 0.7
        gate2 = gate0 @ gate1
        assert qf.gates_close(gate, gate2)

    for K in range(1, 5):
        gate = qf.random_gate(K)  # FIXME: Throw error on K=0
        sqrt_gate = gate ** 0.5
        two_gate = sqrt_gate @ sqrt_gate
        assert qf.gates_close(gate, two_gate)

    for gate in gates:
        rgate = qf.Gate((gate**0.5).tensor)
        tgate = rgate @ rgate
        assert qf.gates_close(gate, tgate)


def test_qubit_qaoa_circuit():
    # Adapted from reference QVM
    wf_true = np.array(
        [0.00167784 + 1.00210180e-05*1j, 0.50000000 - 4.99997185e-01*1j,
         0.50000000 - 4.99997185e-01*1j, 0.00167784 + 1.00210180e-05*1j])
    ket_true = qf.State(wf_true.reshape((2, 2)))

    ket = qf.zero_state(2)
    ket = qf.RY(pi/2, 0).run(ket)
    ket = qf.RX(pi, 0).run(ket)
    ket = qf.RY(pi/2, 1).run(ket)
    ket = qf.RX(pi, 1).run(ket)
    ket = qf.CNOT(0, 1).run(ket)
    ket = qf.RX(-pi/2, 1).run(ket)
    ket = qf.RY(4.71572463191, 1).run(ket)
    ket = qf.RX(pi/2, 1).run(ket)
    ket = qf.CNOT(0, 1).run(ket)
    ket = qf.RX(-2*2.74973750579, 0).run(ket)
    ket = qf.RX(-2*2.74973750579, 1).run(ket)

    assert qf.states_close(ket, ket_true)


# fin
