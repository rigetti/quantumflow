
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Unit tests for quantumflow.channels
"""
# Kudos: Tests adapted from density branch of reference-qvm by Nick Rubin

from functools import reduce
from operator import add

import numpy as np
from numpy import pi

import pytest

import quantumflow as qf
import quantumflow.backend as bk

from . import ALMOST_ZERO, ALMOST_ONE


def test_transpose_map():
    # The transpose map is a superoperator that transposes a 1-qubit
    # density matrix. Not physical.
    # quant-ph/0202124

    ops = [qf.Gate(np.asarray([[1, 0], [0, 0]])),
           qf.Gate(np.asarray([[0, 0], [0, 1]])),
           qf.Gate(np.asarray([[0, 1], [1, 0]]) / np.sqrt(2)),
           qf.Gate(np.asarray([[0, 1], [-1, 0]]) / np.sqrt(2))]

    kraus = qf.Kraus(ops, weights=(1, 1, 1, -1))
    rho0 = qf.random_density(1)
    rho1 = kraus.evolve(rho0)

    op0 = qf.asarray(rho0.asoperator())
    op1 = qf.asarray(rho1.asoperator())
    assert np.allclose(op0.T, op1)

    # The Choi matrix should be same as SWAP operator
    choi = kraus.aschannel().choi()
    choi = qf.asarray(choi)
    assert np.allclose(choi, qf.asarray(qf.SWAP(0, 2).asoperator()))


def test_random_density():
    rho = qf.random_density(4)
    assert list(rho.vec.asarray().shape) == [2]*8


def test_density():
    ket = qf.random_state(3)
    matrix = bk.outer(ket.tensor, bk.conj(ket.tensor))
    qf.Density(matrix)
    qf.Density(matrix, [0, 1, 2])

    with pytest.raises(ValueError):
        qf.Density(matrix, [0, 1, 2, 3])


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


def test_purity():
    density = qf.ghz_state(4).asdensity()
    assert qf.asarray(qf.purity(density)) == ALMOST_ONE

    for _ in range(10):
        rho = qf.random_density(4)
        purity = np.real(qf.asarray(qf.purity(rho)))
        assert purity < 1.0
        assert purity >= 0.0


def test_stdkraus_creation():
    qf.Damping(0.1, 0)
    qf.Depolarizing(0.1, 0)
    qf.Dephasing(0.1, 0)


def test_stdchannels_creation():
    qf.Damping(0.1, 0).aschannel()
    qf.Depolarizing(0.1, 0).aschannel()
    qf.Dephasing(0.1, 0).aschannel()


def test_identity():
    chan = qf.identity_gate(1).aschannel()
    rho = qf.random_density(2)
    after = chan.evolve(rho)
    assert qf.densities_close(rho, after)

    assert chan.name == 'Channel'


def test_channel_chi():
    chan = qf.identity_gate(3).aschannel()
    chi = qf.asarray(chan.chi())
    assert chi.shape == (64, 64)


def test_channel_add():
    chan1 = qf.identity_gate(1).aschannel()
    chan1 *= 0.5

    chan2 = qf.X().aschannel()
    chan2 *= 0.5

    chan = chan1 + chan2
    assert chan is not None

    chan3 = qf.X(3).aschannel()
    with pytest.raises(ValueError):
        chan = chan1 + chan3

    with pytest.raises(NotImplementedError):
        chan = chan1 + 2.0
        assert chan is not None


def test_sample_coin():
    chan = qf.H(0).aschannel()
    rho = qf.zero_state(1).asdensity()
    rho = chan.evolve(rho)
    prob = qf.asarray(rho.probabilities())
    assert np.allclose(prob, [[0.5, 0.5]])


def test_sample_bell():
    rho = qf.zero_state(2).asdensity()
    chan = qf.H(0).aschannel()
    rho = chan.evolve(rho)
    chan = qf.CNOT(0, 1).aschannel()    # TODO: chanmul
    rho = chan.evolve(rho)
    prob = qf.asarray(rho.probabilities())

    assert np.allclose(prob, [[0.5, 0], [0, 0.5]])


def test_biased_coin():
    # sample from a 75% head and 25% tails coin
    rho = qf.zero_state(1).asdensity()
    chan = qf.RX(np.pi/3, 0).aschannel()
    rho = chan.evolve(rho)
    prob = qf.asarray(rho.probabilities())
    assert np.allclose(prob, [0.75, 0.25])


def test_measurement():
    rho = qf.zero_state(2).asdensity()
    chan = qf.H(0).aschannel()
    rho = chan.evolve(rho)
    rho = qf.Kraus([qf.P0(0), qf.P1(0)]).aschannel().evolve(rho)
    K = qf.Kraus([qf.P0(1), qf.P1(1)])
    chan = K.aschannel()

    rho = qf.Kraus([qf.P0(1), qf.P1(1)]).aschannel().evolve(rho)
    prob = qf.asarray(rho.probabilities())
    assert np.allclose(prob, [[0.5, 0], [0.5, 0]])
    assert prob[0, 0]*2 == ALMOST_ONE
    assert prob[1, 0]*2 == ALMOST_ONE


def test_qaoa():
    ket_true = [0.00167784 + 1.00210180e-05*1j, 0.5 - 4.99997185e-01*1j,
                0.5 - 4.99997185e-01*1j, 0.00167784 + 1.00210180e-05*1j]
    rho_true = qf.State(ket_true).asdensity()

    rho = qf.zero_state(2).asdensity()
    rho = qf.RY(pi/2, 0).aschannel().evolve(rho)
    rho = qf.RX(pi, 0).aschannel().evolve(rho)
    rho = qf.RY(pi/2, 1).aschannel().evolve(rho)
    rho = qf.RX(pi, 1).aschannel().evolve(rho)
    rho = qf.CNOT(0, 1).aschannel().evolve(rho)
    rho = qf.RX(-pi/2, 1).aschannel().evolve(rho)
    rho = qf.RY(4.71572463191, 1).aschannel().evolve(rho)
    rho = qf.RX(pi/2, 1).aschannel().evolve(rho)
    rho = qf.CNOT(0, 1).aschannel().evolve(rho)
    rho = qf.RX(-2*2.74973750579, 0).aschannel().evolve(rho)
    rho = qf.RX(-2*2.74973750579, 1).aschannel().evolve(rho)
    assert qf.densities_close(rho, rho_true)


def test_amplitude_damping():
    rho = qf.zero_state(1).asdensity()
    p = 1.0 - np.exp(-(50)/15000)
    chan = qf.Damping(p, 0).aschannel()
    rho1 = chan.evolve(rho)
    assert qf.densities_close(rho, rho1)

    rho2 = qf.X(0).aschannel().evolve(rho1)
    rho3 = chan.evolve(rho2)

    expected = qf.Density([[0.00332778+0.j, 0.00000000+0.j],
                           [0.00000000+0.j, 0.99667222+0.j]])
    assert qf.densities_close(expected, rho3)


def test_depolarizing():
    p = 1.0 - np.exp(-1/20)

    chan = qf.Depolarizing(p, 0).aschannel()
    rho0 = qf.Density([[0.5, 0], [0, 0.5]])
    assert qf.densities_close(rho0, chan.evolve(rho0))

    rho0 = qf.random_density(1)
    rho1 = chan.evolve(rho0)
    pr0 = np.real(qf.asarray(qf.purity(rho0)))
    pr1 = np.real(qf.asarray(qf.purity(rho1)))
    assert pr0 > pr1

    # Test data extracted from refereneqvm
    rho2 = qf.Density([[0.43328691, 0.48979689],
                       [0.48979689, 0.56671309]])
    rho_test = qf.Density([[0.43762509+0.j, 0.45794666+0.j],
                           [0.45794666+0.j, 0.56237491+0.j]])
    assert qf.densities_close(chan.evolve(rho2), rho_test)

    ket0 = qf.random_state(1)
    qf.Depolarizing(p, 0).run(ket0)

    rho1b = qf.Depolarizing(p, 0).evolve(rho0)
    assert qf.densities_close(rho1, rho1b)


def test_kruas_qubits():
    rho = qf.Kraus([qf.P0(0), qf.P1(1)])
    assert rho.qubits == (0, 1)
    assert rho.qubit_nb == 2


def test_kraus_evolve():
    rho = qf.zero_state(1).asdensity()
    p = 1 - np.exp(-50/15000)
    kraus = qf.Damping(p, 0)
    rho1 = kraus.evolve(rho)
    assert qf.densities_close(rho, rho1)

    rho2 = qf.X(0).aschannel().evolve(rho1)
    rho3 = kraus.evolve(rho2)

    expected = qf.Density([[0.00332778+0.j, 0.00000000+0.j],
                           [0.00000000+0.j, 0.99667222+0.j]])

    assert qf.densities_close(expected, rho3)


def test_kraus_run():
    ket0 = qf.zero_state(['a'])
    ket0 = qf.X('a').run(ket0)
    p = 1.0 - np.exp(-2000/15000)

    kraus = qf.Damping(p, 'a')

    reps = 1000
    results = [kraus.run(ket0).asdensity().asoperator()
               for _ in range(reps)]
    matrix = reduce(add, results) / reps
    rho_kraus = qf.Density(matrix, ['a'])

    rho0 = ket0.asdensity()
    chan = kraus.aschannel()
    rho_chan = chan.evolve(rho0)

    # If this fails occasionally consider increasing tolerance
    # Can't be very tolerant due to stochastic dynamics

    assert qf.densities_close(rho_chan, rho_kraus, tolerance=0.05)


def test_channel_adjoint():
    kraus0 = qf.Damping(0.1, 0)
    chan0 = kraus0.aschannel()
    chan1 = chan0.H.H
    assert qf.channels_close(chan0, chan1)

    chan2 = kraus0.H.aschannel()
    assert qf.channels_close(chan2, chan0.H)

    # 2 qubit hermitian channel
    chan3 = qf.CZ(0, 1).aschannel()
    chan4 = chan3.H
    assert qf.channels_close(chan3, chan4)


def test_kraus_qubits():
    kraus = qf.Kraus([qf.X(1), qf.Y(0)])
    assert kraus.qubits == (0, 1)

    kraus = qf.Kraus([qf.X('a'), qf.Y('b')])
    assert len(kraus.qubits) == 2
    assert kraus.qubit_nb == 2


def test_chan_qubits():
    chan = qf.Kraus([qf.X(1), qf.Y(0)]).aschannel()
    assert chan.qubits == (0, 1)
    assert chan.qubit_nb == 2


def test_chan_permute():
    chan0 = qf.CNOT(0, 1).aschannel()
    chan1 = qf.CNOT(1, 0).aschannel()

    assert not qf.channels_close(chan0, chan1)

    chan2 = chan1.permute([0, 1])
    assert chan2.qubits == (0, 1)
    assert qf.channels_close(chan1, chan2)

    chan3 = chan1.relabel([0, 1])
    assert qf.channels_close(chan0, chan3)


def test_channel_errors():
    chan = qf.CNOT(0, 1).aschannel()
    with pytest.raises(TypeError):
        chan.run(qf.zero_state(2))

    with pytest.raises(TypeError):
        chan.asgate()

    assert chan.aschannel() is chan

    with pytest.raises(NotImplementedError):
        chan @ 123


def test_kraus_errors():
    kraus = qf.Kraus([qf.X(1), qf.Y(0)])

    with pytest.raises(TypeError):
        kraus.asgate()


def test_kraus_complete():
    kraus = qf.Kraus([qf.X(1)])
    assert qf.kraus_iscomplete(kraus)

    kraus = qf.Damping(0.1, 0)
    assert qf.kraus_iscomplete(kraus)

    assert qf.kraus_iscomplete(qf.Damping(0.1, 0))
    assert qf.kraus_iscomplete(qf.Dephasing(0.1, 0))
    assert qf.kraus_iscomplete(qf.Depolarizing(0.1, 0))


def test_askraus():

    def _roundtrip(kraus):
        assert qf.kraus_iscomplete(kraus)

        chan0 = kraus.aschannel()
        kraus1 = qf.channel_to_kraus(chan0)
        assert qf.kraus_iscomplete(kraus1)

        chan1 = kraus1.aschannel()
        assert qf.channels_close(chan0, chan1)

    p = 1 - np.exp(-50/15000)
    _roundtrip(qf.Kraus([qf.X(1)]))
    _roundtrip(qf.Damping(p, 0))
    _roundtrip(qf.Depolarizing(0.9, 1))


def test_channel_trace():
    chan = qf.I(0).aschannel()
    assert np.isclose(qf.asarray(chan.trace()), 4)


def test_channel_join():
    chan0 = qf.H(0).aschannel()
    chan1 = qf.X(1).aschannel()
    chan01 = qf.join_channels(chan0, chan1)
    assert chan01.qubits == (0, 1)


# fin
