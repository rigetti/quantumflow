
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Unit tests for quantumflow measures
"""

# TODO: unit tests for measures are currently scattered in other tests.

import numpy as np

import pytest

import quantumflow as qf
from quantumflow import backend as bk

from . import ALMOST_ZERO, ALMOST_ONE


def test_state_angle():
    ket0 = qf.random_state(1)
    ket1 = qf.random_state(1)
    qf.state_angle(ket0, ket1)

    assert not qf.states_close(ket0, ket1)
    assert qf.states_close(ket0, ket0)


def test_density_angle():
    rho0 = qf.random_density(1)
    rho1 = qf.random_density(1)
    qf.density_angle(rho0, rho1)

    assert not qf.densities_close(rho0, rho1)
    assert qf.densities_close(rho0, rho0)


def test_gate_angle():
    gate0 = qf.random_gate(1)
    gate1 = qf.random_gate(1)
    qf.gate_angle(gate0, gate1)

    assert not qf.gates_close(gate0, gate1)
    assert qf.gates_close(gate0, gate0)


def test_channel_angle():
    chan0 = qf.X(0).aschannel()
    chan1 = qf.Y(0).aschannel()
    qf.channel_angle(chan0, chan1)

    assert not qf.channels_close(chan0, chan1)
    assert qf.channels_close(chan0, chan0)


def test_fidelity():
    rho0 = qf.random_density(4)
    rho1 = qf.random_density(4)

    fid = qf.fidelity(rho0, rho1)
    print('FID', fid)
    assert 0.0 <= fid <= 1.0

    rho2 = qf.random_density([3, 2, 1, 0])
    fid = qf.fidelity(rho0, rho2)
    assert 0.0 <= fid <= 1.0

    fid = qf.fidelity(rho0, rho0)
    print('FID', fid)
    assert fid == ALMOST_ONE

    ket0 = qf.random_state(3)
    ket1 = qf.random_state(3)
    fid0 = qf.state_fidelity(ket0, ket1)

    rho0 = ket0.asdensity()
    rho1 = ket1.asdensity()
    fid1 = qf.fidelity(rho0, rho1)

    assert qf.asarray(fid1 - fid0) == ALMOST_ZERO

    fid2 = bk.cos(qf.fubini_study_angle(ket0.vec, ket1.vec))**2
    assert qf.asarray(fid2 - fid0) == ALMOST_ZERO


def test_purity():
    density = qf.ghz_state(4).asdensity()
    assert qf.asarray(qf.purity(density)) == ALMOST_ONE

    for _ in range(10):
        density = qf.random_density(4)
        purity = np.real(qf.asarray(qf.purity(density)))
        assert purity < 1.0
        assert purity >= 0.0

    rho = qf.Density(np.diag([0.9, 0.1]))
    assert np.isclose(qf.asarray(qf.purity(rho)), 0.82)   # Kudos: Josh Combes


def test_bures_distance():
    rho = qf.random_density(4)
    assert qf.bures_distance(rho, rho) == ALMOST_ZERO

    rho1 = qf.random_density(4)
    qf.bures_distance(rho, rho1)

    # TODO: Check distance of known special case


def test_bures_angle():
    rho = qf.random_density(4)
    assert qf.bures_angle(rho, rho) == ALMOST_ZERO

    rho1 = qf.random_density(4)
    qf.bures_angle(rho, rho1)

    ket0 = qf.random_state(4)
    ket1 = qf.random_state(4)
    rho0 = ket0.asdensity()
    rho1 = ket1.asdensity()

    ang0 = qf.fubini_study_angle(ket0.vec, ket1.vec)
    ang1 = qf.bures_angle(rho0, rho1)

    assert np.isclose(ang0, ang1)


def test_entropy():
    N = 4
    rho0 = qf.mixed_density(N)
    ent = qf.entropy(rho0, base=2)
    assert np.isclose(ent, N)

    # Entropy invariant to unitary evolution
    chan = qf.random_gate(N).aschannel()
    rho1 = chan.evolve(rho0)
    ent = qf.entropy(rho1, base=2)
    assert np.isclose(ent, N)


def test_mutual_info():
    rho0 = qf.mixed_density(4)
    info0 = qf.mutual_info(rho0, qubits0=[0, 1], qubits1=[2, 3])

    # Information invariant to local unitary evolution
    chan = qf.random_gate(2).aschannel()
    rho1 = chan.evolve(rho0)
    info1 = qf.mutual_info(rho1, qubits0=[0, 1], qubits1=[2, 3])

    assert np.isclose(info0, info1)

    info2 = qf.mutual_info(rho1, qubits0=[0, 1])
    assert np.isclose(info0, info2)


def test_diamond_norm():
    # Test cases borrowed from qutip,
    # https://github.com/qutip/qutip/blob/master/qutip/tests/test_metrics.py
    # which were in turn  generated using QuantumUtils for MATLAB
    # (https://goo.gl/oWXhO9)

    chan0 = qf.I(0).aschannel()
    chan1 = qf.X(0).aschannel()
    dn = qf.diamond_norm(chan0, chan1)
    assert np.isclose(2.0, dn, rtol=0.0001)

    turns_dnorm = [[1.000000e-03, 3.141591e-03],
                   [3.100000e-03, 9.738899e-03],
                   [1.000000e-02, 3.141463e-02],
                   [3.100000e-02, 9.735089e-02],
                   [1.000000e-01, 3.128689e-01],
                   [3.100000e-01, 9.358596e-01]]

    for turns, target in turns_dnorm:
        chan0 = qf.TX(0).aschannel()
        chan1 = qf.TX(turns).aschannel()

        dn = qf.diamond_norm(chan0, chan1)
        assert np.isclose(target, dn, rtol=0.0001)

    hadamard_mixtures = [[1.000000e-03, 2.000000e-03],
                         [3.100000e-03, 6.200000e-03],
                         [1.000000e-02, 2.000000e-02],
                         [3.100000e-02, 6.200000e-02],
                         [1.000000e-01, 2.000000e-01],
                         [3.100000e-01, 6.200000e-01]]

    for p, target in hadamard_mixtures:
        # FIXME: implement __rmul__ for channels
        chan0 = qf.I(0).aschannel() * (1 - p) + qf.H(0).aschannel() * p
        chan1 = qf.I(0).aschannel()

        dn = qf.diamond_norm(chan0, chan1)
        assert np.isclose(dn, target, rtol=0.0001)

    chan0 = qf.TY(0.5, 0).aschannel()
    chan1 = qf.I(0).aschannel()
    dn = qf.diamond_norm(chan0, chan1)
    assert np.isclose(dn, np.sqrt(2), rtol=0.0001)

    chan0 = qf.CNOT(0, 1).aschannel()
    chan1 = qf.CNOT(1, 0).aschannel()
    qf.diamond_norm(chan0, chan1)


def test_diamond_norm_err():
    with pytest.raises(ValueError):
        chan0 = qf.I(0).aschannel()
        chan1 = qf.I(1).aschannel()
        qf.diamond_norm(chan0, chan1)

# fin
