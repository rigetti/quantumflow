
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Unit tests for quantumflow States
"""


import io

import numpy as np

import pytest
import quantumflow as qf
from quantumflow import backend as bk

from . import ALMOST_ZERO, ALMOST_ONE, REPS, skip_torch


# Test States

def test_zeros():
    ket = qf.zero_state(4)
    vec = ket.vec.asarray()
    assert vec[0, 0, 0, 0] == ALMOST_ONE
    assert vec[0, 0, 1, 0] == ALMOST_ZERO


def test_w_state():
    vec = qf.w_state(4).vec.asarray()
    assert vec[0, 0, 0, 0] == ALMOST_ZERO
    assert vec[0, 0, 1, 0]*2. == ALMOST_ONE


def test_ghz_state():
    vec = qf.ghz_state(4).vec.asarray()
    assert vec[0, 0, 0, 1] == ALMOST_ZERO
    assert 2.*vec[0, 0, 0, 0]**2. == ALMOST_ONE


def test_random_state():
    state = qf.random_state(4).vec.asarray()
    assert state.shape == (2,)*4


def test_state_bits():
    for n in range(1, 6):
        assert qf.zero_state(n).qubit_nb == n


def test_state_labels():
    # Quil labeling convention
    N = 4
    qubits = range(N-1, -1, -1)
    ket = qf.zero_state(qubits)
    ket = qf.X(0).run(ket)
    ket = qf.X(1).run(ket)
    assert ket.vec.asarray()[0, 0, 1, 1] == ALMOST_ONE

    ket = ket.relabel([0, 1, 3, 4])
    assert ket.vec.asarray()[0, 0, 1, 1] == ALMOST_ONE

    ket = ket.permute([4, 3, 0, 1])


def test_probability():
    state = qf.w_state(3)
    qf.print_state(state)
    prob = state.probabilities()

    qf.print_probabilities(state)
    assert qf.asarray(prob).sum() == ALMOST_ONE


def test_states_close():
    ket0 = qf.w_state(4)
    ket1 = qf.w_state(3)
    ket2 = qf.w_state(4)

    assert qf.states_close(ket0, ket2)
    assert not qf.states_close(ket0, ket1)
    assert qf.states_close(ket2, ket2)


def test_str():
    ket = qf.random_state(10)
    s = str(ket)
    assert s[-3:] == '...'


def test_print_state():
    f = io.StringIO()
    state = qf.w_state(5)
    qf.print_state(state, file=f)
    print(f.getvalue())


def test_print_probabilities():
    f = io.StringIO()
    state = qf.w_state(5)
    qf.print_probabilities(state, file=f)
    print(f.getvalue())


def test_measure():
    ket = qf.zero_state(2)
    res = ket.measure()
    assert np.allclose(res, [0, 0])

    ket = qf.X(0).run(ket)
    res = ket.measure()
    assert np.allclose(res, [1, 0])

    ket = qf.H(0).run(ket)
    ket = qf.CNOT(0, 1).run(ket)
    for _ in range(REPS):
        res = ket.measure()
        assert res[0] == res[1]  # Both qubits measured in same state


def test_sample():
    ket = qf.zero_state(2)
    ket = qf.H(0).run(ket)
    ket = qf.CNOT(0, 1).run(ket)

    samples = ket.sample(10)
    assert samples.sum() == 10


def test_expectation():
    ket = qf.zero_state(1)
    ket = qf.H(0).run(ket)

    m = ket.expectation([0.4, 0.6])
    assert qf.asarray(m) - 0.5 == ALMOST_ZERO

    m = ket.expectation([0.4, 0.6], 10)


def test_random_density():
    rho = qf.random_density(4)
    assert list(rho.vec.asarray().shape) == [2]*8


def test_density():
    ket = qf.random_state(3)
    matrix = bk.outer(ket.tensor, bk.conj(ket.tensor))
    qf.Density(matrix)
    rho = qf.Density(matrix, [0, 1, 2])

    with pytest.raises(ValueError):
        qf.Density(matrix, [0, 1, 2, 3])

    assert rho.asdensity() is rho

    rho = rho.relabel([10, 11, 12]).permute([12, 11, 10])
    assert rho.qubits == (12, 11, 10)


def test_state_to_density():
    density = qf.ghz_state(4).asdensity()
    assert list(density.vec.asarray().shape) == [2]*8

    prob = qf.asarray(density.probabilities())
    assert prob[0, 0, 0, 0] - 0.5 == ALMOST_ZERO
    assert prob[0, 1, 0, 0] == ALMOST_ZERO
    assert prob[1, 1, 1, 1] - 0.5 == ALMOST_ZERO

    ket = qf.random_state(3)
    density = ket.asdensity()
    ket_prob = qf.asarray(ket.probabilities())
    density_prob = qf.asarray(density.probabilities())

    for index, prob in np.ndenumerate(ket_prob):
        assert prob - density_prob[index] == ALMOST_ZERO


@skip_torch  # FIXME: Currently broken in torch backend
def test_density_trace():
    rho = qf.random_density(3)
    assert qf.asarray(rho.trace()) == ALMOST_ONE

    rho = qf.Density(np.eye(8))
    assert np.isclose(qf.asarray(rho.trace()), 8)

    rho = rho.normalize()
    assert np.isclose(qf.asarray(rho.trace()), 1)


def test_mixed_density():
    rho = qf.mixed_density(4)
    assert qf.asarray(rho.trace()) == ALMOST_ONE


def test_join_densities():
    rho0 = qf.zero_state([0]).asdensity()
    rho1 = qf.zero_state([1]).asdensity()
    rho01 = qf.join_densities(rho0, rho1)
    assert rho01.qubits == (0, 1)


def test_memory():
    ket0 = qf.zero_state(1)
    assert ket0.memory == {}

    ro = qf.Register('ro')
    ket1 = ket0.update({ro[1]: 1})
    assert ket0.memory == {}
    assert ket1.memory == {ro[1]: 1}

    assert ket1.cbit_nb == 1
    assert ket1.cbits == (ro[1],)

    ket2 = qf.H(0).run(ket1)
    assert ket2.memory == ket1.memory

    ket3 = ket2.update({ro[1]: 0, ro[0]: 0})
    assert ket3.memory == {ro[0]: 0, ro[1]: 0}
    assert ket3.cbits == (ro[0], ro[1])


def test_density_memory():
    rho0 = qf.zero_state(1).asdensity()
    assert rho0.memory == {}

    ro = qf.Register('ro')
    rho1 = rho0.update({ro[1]: 1})
    assert rho0.memory == {}
    assert rho1.memory == {ro[1]: 1}

    assert rho1.cbit_nb == 1
    assert rho1.cbits == (ro[1],)

    rho2 = qf.H(0).aschannel().evolve(rho1)
    assert rho2.memory == rho2.memory

    rho3 = qf.X(0).evolve(rho1)
    assert rho3.memory == rho2.memory


# fin
