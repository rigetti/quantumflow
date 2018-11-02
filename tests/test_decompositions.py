
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Unittests for QuantumFlow Gate Decompositions
"""

import numpy as np
from numpy import pi
import scipy.stats

import pytest

import quantumflow as qf
from quantumflow.decompositions import _eig_complex_symmetric

from . import REPS, skip_tensorflow


def test_bloch_decomposition():
    theta = 1.23

    gate0 = qf.RN(theta, 1, 0, 0)
    gate1 = qf.bloch_decomposition(gate0).elements[0]
    assert qf.gates_close(gate0, gate1)

    gate0 = qf.RN(theta, 0, 1, 0)
    gate1 = qf.bloch_decomposition(gate0).elements[0]
    assert qf.gates_close(gate0, gate1)

    gate0 = qf.RN(theta, 0, 0, 1)
    gate1 = qf.bloch_decomposition(gate0).elements[0]
    assert qf.gates_close(gate0, gate1)

    gate0 = qf.RN(pi, np.sqrt(2), 0, np.sqrt(2))
    gate1 = qf.bloch_decomposition(gate0).elements[0]
    assert qf.gates_close(gate0, gate1)

    for _ in range(REPS):
        gate0 = qf.random_gate(qubits=[0])
        gate1 = qf.bloch_decomposition(gate0).elements[0]
        assert qf.gates_close(gate0, gate1)

    gate0 = qf.I(0)
    gate1 = qf.bloch_decomposition(gate0).elements[0]
    assert qf.gates_close(gate0, gate1)


def test_bloch_decomp_errors():
    # Wrong number of qubits
    with pytest.raises(ValueError):
        qf.bloch_decomposition(qf.CNOT())


def test_zyz_decomp_errors():
    # Wrong number of qubits
    with pytest.raises(ValueError):
        qf.zyz_decomposition(qf.CNOT())


def test_zyz_decomposition():
    gate0 = qf.random_gate(1)
    circ1 = qf.zyz_decomposition(gate0)
    gate1 = circ1.asgate()
    assert qf.gates_close(gate0, gate1)


def test_kronecker_decomposition():
    for _ in range(REPS):
        left = qf.random_gate(1).vec.asarray()
        right = qf.random_gate(1).vec.asarray()
        both = np.kron(left, right)
        gate0 = qf.Gate(both, qubits=[0, 1])
        circ = qf.kronecker_decomposition(gate0)
        gate1 = circ.asgate()

        assert qf.gates_close(gate0, gate1)

    circ0 = qf.Circuit()
    circ0 += qf.Z(0)
    circ0 += qf.H(1)
    gate0 = circ.asgate()
    circ1 = qf.kronecker_decomposition(gate0)
    gate1 = circ1.asgate()

    assert qf.gates_close(gate0, gate1)


def test_kronecker_decomp_errors():
    # Wrong number of qubits
    with pytest.raises(ValueError):
        qf.kronecker_decomposition(qf.X(0))

    # Not kronecker product
    with pytest.raises(ValueError):
        qf.kronecker_decomposition(qf.CNOT(0, 1))


@skip_tensorflow  # FIXME: Runs slowly in tensorflow, for unclear reasons
def test_canonical_decomposition():
    for tt1 in range(0, 10):
        for tt2 in range(tt1):
            for tt3 in range(tt2):
                t1, t2, t3 = tt1/20, tt2/20, tt3/20
                if t3 == 0 and t1 > 0.5:
                    continue
                coords = np.asarray((t1, t2, t3))

                print('b')
                circ0 = qf.Circuit()
                circ0 += qf.ZYZ(0.2, 0.2, 0.2, q0=0)
                circ0 += qf.ZYZ(0.3, 0.3, 0.3, q0=1)
                circ0 += qf.CANONICAL(t1, t2, t3, 0, 1)
                circ0 += qf.ZYZ(0.15, 0.2, 0.3, q0=0)
                circ0 += qf.ZYZ(0.15, 0.22, 0.3, q0=1)
                gate0 = circ0.asgate()
                print('c')

                circ1 = qf.canonical_decomposition(gate0)
                assert qf.gates_close(gate0, circ1.asgate())
                print('d')

                print(circ1)
                canon = circ1.elements[6]
                new_coords = np.asarray([canon.params[n] for n in
                                         ['tx', 'ty', 'tz']])
                assert np.allclose(coords, np.asarray(new_coords))

                coords2 = qf.canonical_coords(gate0)
                assert np.allclose(coords, np.asarray(coords2))
                print('>')
                print()


@skip_tensorflow
def test_canonical_decomp_sandwich():
    for _ in range(REPS):
        # Random CZ sandwich
        circ0 = qf.Circuit()
        circ0 += qf.random_gate([0])
        circ0 += qf.random_gate([1])
        circ0 += qf.CZ(0, 1)
        circ0 += qf.TY(0.4, 0)
        circ0 += qf.TY(0.25, 1)
        circ0 += qf.CZ(0, 1)
        circ0 += qf.random_gate([0])
        circ0 += qf.random_gate([1])

        gate0 = circ0.asgate()

        circ1 = qf.canonical_decomposition(gate0)
        gate1 = circ1.asgate()

        assert qf.gates_close(gate0, gate1)
        assert qf.almost_unitary(gate0)


@skip_tensorflow
def test_canonical_decomp_random():
    for _ in range(REPS*2):
        gate0 = qf.random_gate([0, 1])
        gate1 = qf.canonical_decomposition(gate0).asgate()
        assert qf.gates_close(gate0, gate1)


def test_canonical_decomp_errors():
    # Wrong number of qubits
    with pytest.raises(ValueError):
        qf.canonical_decomposition(qf.X())


@skip_tensorflow
def test_decomp_stdgates():
    gate0 = qf.I(0, 1)
    gate1 = qf.canonical_decomposition(gate0).asgate()
    assert qf.gates_close(gate0, gate1)

    gate0 = qf.CNOT(0, 1)
    gate1 = qf.canonical_decomposition(gate0).asgate()
    assert qf.gates_close(gate0, gate1)

    gate0 = qf.SWAP(0, 1)
    gate1 = qf.canonical_decomposition(gate0).asgate()
    assert qf.gates_close(gate0, gate1)

    gate0 = qf.ISWAP(0, 1)
    gate1 = qf.canonical_decomposition(gate0).asgate()
    assert qf.gates_close(gate0, gate1)

    gate0 = qf.CNOT(0, 1) ** 0.5
    gate1 = qf.canonical_decomposition(gate0).asgate()
    assert qf.gates_close(gate0, gate1)

    gate0 = qf.SWAP(0, 1) ** 0.5
    gate1 = qf.canonical_decomposition(gate0).asgate()
    assert qf.gates_close(gate0, gate1)

    gate0 = qf.ISWAP(0, 1) ** 0.5
    gate1 = qf.canonical_decomposition(gate0).asgate()
    assert qf.gates_close(gate0, gate1)


def test_decomp_sqrtswap_sandwich():
    circ0 = qf.Circuit()
    circ0 += qf.CANONICAL(1/4, 1/4, 1/4, 0, 1)
    circ0 += qf.random_gate([0])
    circ0 += qf.random_gate([1])
    circ0 += qf.CANONICAL(1/4, 1/4, 1/4, 0, 1)

    gate0 = circ0.asgate()
    circ1 = qf.canonical_decomposition(gate0)
    gate1 = circ1.asgate()
    assert qf.gates_close(gate0, gate1)


def test_eig_complex_symmetric():
    samples = 1000
    for _ in range(samples):

        # Build a random symmetric complex matrix
        orthoganal = scipy.stats.special_ortho_group.rvs(4)
        eigvals = (np.random.normal(size=(4,))
                   + 1j * np.random.normal(size=(4,)))/np.sqrt(2.0)
        M = orthoganal @ np.diag(eigvals)  @ orthoganal.T

        eigvals, eigvecs = _eig_complex_symmetric(M)
        assert np.allclose(M, eigvecs @ np.diag(eigvals) @ eigvecs.T)


def test_eigcs_errors():
    with pytest.raises(np.linalg.LinAlgError):
        _eig_complex_symmetric(np.random.normal(size=(4, 4)))


# Fin
