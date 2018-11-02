
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Unit tests for quantumflow.tensors
"""

import random
from math import pi

import numpy as np

import pytest

import quantumflow as qf
from . import ALMOST_ZERO, ALMOST_ONE

REPS = 4


def test_getitem():
    ket = qf.ghz_state(6)
    assert qf.asarray(ket.vec[0, 1, 0, 0, 0, 0]) == ALMOST_ZERO
    assert qf.asarray(ket.vec[1, 1, 1, 1, 1, 1]) != 0.0


def test_rank():
    data = [1]*256

    assert qf.QubitVector(data, range(1)).rank == 8
    assert qf.QubitVector(data, range(2)).rank == 4
    assert qf.QubitVector(data, range(4)).rank == 2
    assert qf.QubitVector(data, range(8)).rank == 1


def test_trace():
    data = [1]*256

    r8 = qf.QubitVector(data, range(1))
    r4 = qf.QubitVector(data, range(2))
    r2 = qf.QubitVector(data, range(4))
    r1 = qf.QubitVector(data, range(8))

    assert np.isclose(qf.asarray(r8.trace()), 16)
    assert np.isclose(qf.asarray(r4.trace()), 16)
    assert np.isclose(qf.asarray(r2.trace()), 16)

    with pytest.raises(ValueError):
        r1.trace()


def test_partial_trace():
    data = [1]*(2**16)

    r8 = qf.QubitVector(data, range(2))
    r4 = qf.QubitVector(data, range(4))
    r2 = qf.QubitVector(data, range(8))

    tr2 = r2.partial_trace([1])
    assert tr2.qubits == (0, 2, 3, 4, 5, 6, 7)
    assert tr2.rank == 2

    tr2 = r2.partial_trace([2, 3])
    assert tr2.qubits == (0, 1, 4, 5, 6, 7)
    assert tr2.rank == 2

    tr4 = r4.partial_trace([0])
    assert tr4.qubits == (1, 2, 3)
    assert tr4.rank == 4

    tr8 = r8.partial_trace([1])
    assert tr8.qubits == (0, )
    assert tr8.rank == 8

    with pytest.raises(ValueError):
        r2.partial_trace(range(8))

    chan012 = qf.identity_gate(3).aschannel()
    assert np.isclose(qf.asarray(chan012.trace()), 64)   # 2**(2**3)

    chan02 = chan012.partial_trace([1])
    assert np.isclose(qf.asarray(chan02.trace()), 32)    # TODO: Checkme
    chan2 = chan012.partial_trace([0, 1])

    assert np.isclose(qf.asarray(chan2.trace()), 16)     # TODO: checkme

    # partial traced channels should be identities still, upto normalization
    assert qf.channels_close(chan2, qf.I(2).aschannel())
    # TODO: Channel.normalize()

    with pytest.raises(ValueError):
        qf.zero_state(4).vec.partial_trace([1, 2])


def test_inner_product():
    # also tested via test_gate_angle

    for _ in range(REPS):
        theta = random.uniform(-4*pi, +4*pi)

        hs = qf.asarray(qf.inner_product(qf.RX(theta).vec, qf.RX(theta).vec))
        print('RX({}), hilbert_schmidt = {}'.format(theta, hs))
        assert hs/2 == ALMOST_ONE

        hs = qf.asarray(qf.inner_product(qf.RZ(theta).vec, qf.RZ(theta).vec))
        print('RZ({}), hilbert_schmidt = {}'.format(theta, hs))
        assert hs/2 == ALMOST_ONE

        hs = qf.asarray(qf.inner_product(qf.RY(theta).vec, qf.RY(theta).vec))
        print('RY({}), hilbert_schmidt = {}'.format(theta, hs))
        assert hs/2 == ALMOST_ONE

        hs = qf.asarray(qf.inner_product(qf.PSWAP(theta).vec,
                                         qf.PSWAP(theta).vec))
        print('PSWAP({}), hilbert_schmidt = {}'.format(theta, hs))
        assert hs/4 == ALMOST_ONE

    with pytest.raises(ValueError):
        qf.inner_product(qf.zero_state(0).vec, qf.X(0).vec)

    with pytest.raises(ValueError):
        qf.inner_product(qf.CNOT(0, 1).vec, qf.X(0).vec)


def test_fubini_study_angle():

    for _ in range(REPS):
        theta = random.uniform(-pi, +pi)

        ang = qf.asarray(qf.fubini_study_angle(qf.I().vec,
                                               qf.RX(theta).vec))
        assert 2 * ang / abs(theta) == ALMOST_ONE

        ang = qf.asarray(qf.fubini_study_angle(qf.I().vec,
                                               qf.RY(theta).vec))
        assert 2 * ang / abs(theta) == ALMOST_ONE

        ang = qf.asarray(qf.fubini_study_angle(qf.I().vec,
                                               qf.RZ(theta).vec))
        assert 2 * ang / abs(theta) == ALMOST_ONE

        ang = qf.asarray(qf.fubini_study_angle(qf.SWAP().vec,
                                               qf.PSWAP(theta).vec))
        assert 2 * ang / abs(theta) == ALMOST_ONE

        ang = qf.asarray(qf.fubini_study_angle(qf.I().vec,
                                               qf.PHASE(theta).vec))
        assert 2 * ang / abs(theta) == ALMOST_ONE

    for n in range(1, 6):
        eye = qf.identity_gate(n)
        assert qf.asarray(qf.fubini_study_angle(eye.vec, eye.vec)) \
            == ALMOST_ZERO

    with pytest.raises(ValueError):
        qf.fubini_study_angle(qf.random_gate(1).vec,
                              qf.random_gate(2).vec)


def test_fubini_study_angle_states():
    # The state angle is half angle in Bloch sphere
    angle1 = 0.1324
    ket1 = qf.zero_state(1)
    ket2 = qf.RX(angle1, 0).run(ket1)
    angle2 = qf.asarray(qf.fubini_study_angle(ket1.vec, ket2.vec))
    assert angle1 - angle2 * 2 == ALMOST_ZERO


def test_vectors_not_close():
    assert not qf.vectors_close(qf.random_gate(1).vec, qf.random_gate(2).vec)
    assert not qf.vectors_close(qf.zero_state(1).vec, qf.random_gate(1).vec)
    assert not qf.vectors_close(qf.X(0).vec, qf.X(1).vec)


def test_outer_product():
    with pytest.raises(ValueError):
        ket = qf.zero_state(1)
        gate = qf.I(2)
        qf.outer_product(ket.vec, gate.vec)

    with pytest.raises(ValueError):
        ket0 = qf.zero_state([0, 1, 2])
        ket1 = qf.zero_state([2, 3, 4])
        qf.outer_product(ket0.vec, ket1.vec)


# fin
