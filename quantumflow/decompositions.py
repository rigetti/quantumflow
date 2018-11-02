
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow Gate Decompositions
"""

from typing import Sequence, Tuple
import itertools

import numpy as np
from numpy import pi

from .qubits import asarray
from .config import TOLERANCE
from .gates import Gate
from .measures import gates_close
from .stdgates import RN, CANONICAL, TZ, TY
from .circuits import Circuit

__all__ = ['bloch_decomposition',
           'zyz_decomposition',
           'kronecker_decomposition',
           'canonical_decomposition',
           'canonical_coords']


def bloch_decomposition(gate: Gate) -> Circuit:
    """
    Converts a 1-qubit gate into a RN gate, a 1-qubit rotation of angle theta
    about axis (nx, ny, nz) in the Bloch sphere.

    Returns:
        A Circuit containing a single RN gate
    """
    if gate.qubit_nb != 1:
        raise ValueError('Expected 1-qubit gate')

    U = asarray(gate.asoperator())
    U /= np.linalg.det(U) ** (1/2)

    nx = - U[0, 1].imag
    ny = - U[0, 1].real
    nz = - U[0, 0].imag
    N = np.sqrt(nx**2 + ny**2 + nz**2)
    if N == 0:  # Identity
        nx, ny, nz = 1, 1, 1
    else:
        nx /= N
        ny /= N
        nz /= N
    sin_halftheta = N
    cos_halftheta = U[0, 0].real
    theta = 2 * np.arctan2(sin_halftheta, cos_halftheta)

    # We return a Circuit (rather than just a gate) to keep the
    # interface of decomposition routines uniform.
    return Circuit([RN(theta, nx, ny, nz, *gate.qubits)])


# DOCME TESTME
def zyz_decomposition(gate: Gate) -> Circuit:
    """
    Returns the Euler Z-Y-Z decomposition of a local 1-qubit gate.
    """
    if gate.qubit_nb != 1:
        raise ValueError('Expected 1-qubit gate')

    q, = gate.qubits

    U = asarray(gate.asoperator())
    U /= np.linalg.det(U) ** (1/2)    # SU(2)

    if abs(U[0, 0]) > abs(U[1, 0]):
        theta1 = 2 * np.arccos(min(abs(U[0, 0]), 1))
    else:
        theta1 = 2 * np.arcsin(min(abs(U[1, 0]), 1))

    cos_halftheta1 = np.cos(theta1/2)
    if not np.isclose(cos_halftheta1, 0.0):
        phase = U[1, 1] / cos_halftheta1
        theta0_plus_theta2 = 2 * np.arctan2(np.imag(phase), np.real(phase))
    else:
        theta0_plus_theta2 = 0.0

    sin_halftheta1 = np.sin(theta1/2)
    if not np.isclose(sin_halftheta1, 0.0):
        phase = U[1, 0] / sin_halftheta1
        theta0_sub_theta2 = 2 * np.arctan2(np.imag(phase), np.real(phase))
    else:
        theta0_sub_theta2 = 0.0

    theta0 = (theta0_plus_theta2 + theta0_sub_theta2) / 2
    theta2 = (theta0_plus_theta2 - theta0_sub_theta2) / 2

    t0 = theta0/np.pi
    t1 = theta1/np.pi
    t2 = theta2/np.pi

    circ1 = Circuit()
    circ1 += TZ(t2, q)
    circ1 += TY(t1, q)
    circ1 += TZ(t0, q)

    return circ1


def kronecker_decomposition(gate: Gate) -> Circuit:
    """
    Decompose a 2-qubit unitary composed of two 1-qubit local gates.

    Uses the "Nearest Kronecker Product" algorithm. Will give erratic
    results if the gate is not the direct product of two 1-qubit gates.
    """
    # An alternative approach would be to take partial traces, but
    # this approach appears to be more robust.

    if gate.qubit_nb != 2:
        raise ValueError('Expected 2-qubit gate')

    U = asarray(gate.asoperator())
    rank = 2**gate.qubit_nb
    U /= np.linalg.det(U) ** (1/rank)

    R = np.stack([U[0:2, 0:2].reshape(4),
                  U[0:2, 2:4].reshape(4),
                  U[2:4, 0:2].reshape(4),
                  U[2:4, 2:4].reshape(4)])
    u, s, vh = np.linalg.svd(R)
    v = vh.transpose()
    A = (np.sqrt(s[0]) * u[:, 0]).reshape(2, 2)
    B = (np.sqrt(s[0]) * v[:, 0]).reshape(2, 2)

    q0, q1 = gate.qubits
    g0 = Gate(A, qubits=[q0])
    g1 = Gate(B, qubits=[q1])

    if not gates_close(gate, Circuit([g0, g1]).asgate()):
        raise ValueError("Gate cannot be decomposed into two 1-qubit gates")

    circ = Circuit()
    circ += zyz_decomposition(g0)
    circ += zyz_decomposition(g1)

    assert gates_close(gate, circ.asgate())  # Sanity check

    return circ


def canonical_coords(gate: Gate) -> Sequence[float]:
    """Returns the canonical coordinates of a 2-qubit gate"""
    circ = canonical_decomposition(gate)
    gate = circ.elements[6]  # type: ignore
    params = [gate.params[key] for key in ('tx', 'ty', 'tz')]
    return params


def canonical_decomposition(gate: Gate) -> Circuit:
    """Decompose a 2-qubit gate by removing local 1-qubit gates to leave
    the non-local canonical two-qubit gate. [1]_ [2]_ [3]_ [4]_

    Returns: A Circuit of 5 gates: two initial 1-qubit gates; a CANONICAL
    gate, with coordinates in the Weyl chamber; two final 1-qubit gates

    The canonical coordinates can be found in circ.elements[2].params

    More or less follows the algorithm outlined in [2]_.

    .. [1] A geometric theory of non-local two-qubit operations, J. Zhang,
        J. Vala, K. B. Whaley, S. Sastry quant-ph/0291120
    .. [2] An analytical decomposition protocol for optimal implementation of
        two-qubit entangling gates. M. Blaauboer, R.L. de Visser,
        cond-mat/0609750
    .. [3] Metric structure of two-qubit gates, perfect entangles and quantum
        control, P. Watts, M. O'Conner, J. Vala, Entropy (2013)
    .. [4] Constructive Quantum Shannon Decomposition from Cartan Involutions
        B. Drury, P. Love, arXiv:0806.4015
    """

    # Implementation note: The canonical decomposition is easy. Constraining
    # canonical coordinates to the Weyl chamber is easy. But doing the
    # canonical decomposition with the canonical gate in the Weyl chamber
    # proved to be surprisingly tricky.

    # Unitary transform to Magic Basis of Bell states
    Q = np.asarray([[1, 0, 0, 1j],
                    [0, 1j, 1, 0],
                    [0, 1j, -1, 0],
                    [1, 0, 0, -1j]]) / np.sqrt(2)
    Q_H = Q.conj().T

    if gate.qubit_nb != 2:
        raise ValueError('Expected 2-qubit gate')

    U = asarray(gate.asoperator())
    rank = 2**gate.qubit_nb
    U /= np.linalg.det(U) ** (1/rank)  # U is in SU(4) so det U = 1

    U_mb = Q_H @ U @ Q   # Transform gate to Magic Basis [1, (eq. 17, 18)]
    M = U_mb.transpose() @ U_mb         # Construct M matrix [1, (eq. 22)]

    # Diagonalize symmetric complex matrix
    eigvals, eigvecs = _eig_complex_symmetric(M)

    lambdas = np.sqrt(eigvals)          # Eigenvalues of F
    # Lambdas only fixed up to a sign. So make sure det F = 1 as it should
    det_F = np.prod(lambdas)
    if det_F.real < 0:
        lambdas[0] *= -1

    coords, signs, perm = _constrain_to_weyl(lambdas)

    # Construct local and canonical gates in magic basis
    lambdas = (lambdas*signs)[perm]
    O2 = (np.diag(signs) @ eigvecs.transpose())[perm]
    F = np.diag(lambdas)
    O1 = U_mb @ O2.transpose() @ F.conj()

    # Sanity check: Make sure O1 and O2 are orthogonal
    assert np.allclose(np.eye(4), O2.transpose() @ O2)  # Sanity check
    assert np.allclose(np.eye(4), O1.transpose() @ O1)  # Sanity check

    # Sometimes O1 & O2 end up with det = -1, instead of +1 as they should.
    # We can commute a diagonal matrix through F to fix this up.
    neg = np.diag([-1, 1, 1, 1])
    if np.linalg.det(O2).real < 0:
        O2 = neg @ O2
        O1 = O1 @ neg

    # Transform gates back from magic basis
    K1 = Q @ O1 @ Q_H
    A = Q @ F @ Q_H
    K2 = Q @ O2 @ Q_H

    assert gates_close(Gate(U), Gate(K1 @ A @ K2))  # Sanity check
    canon = CANONICAL(coords[0], coords[1], coords[2], 0, 1)

    # Sanity check
    assert gates_close(Gate(A, qubits=gate.qubits), canon, tolerance=1e-4)

    # Decompose local gates into the two component 1-qubit gates
    gateK1 = Gate(K1, qubits=gate.qubits)
    circK1 = kronecker_decomposition(gateK1)
    assert gates_close(gateK1, circK1.asgate())  # Sanity check

    gateK2 = Gate(K2, qubits=gate.qubits)
    circK2 = kronecker_decomposition(gateK2)
    assert gates_close(gateK2, circK2.asgate())  # Sanity check

    # Build and return circuit
    circ = Circuit()
    circ += circK2
    circ += canon
    circ += circK1

    return circ


def _eig_complex_symmetric(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Diagonalize a complex symmetric  matrix. The eigenvalues are
    complex, and the eigenvectors form an orthogonal matrix.

    Returns:
        eigenvalues, eigenvectors
    """
    if not np.allclose(M, M.transpose()):
        raise np.linalg.LinAlgError('Not a symmetric matrix')

    # The matrix of eigenvectors should be orthogonal.
    # But the standard 'eig' method will fail to return an orthogonal
    # eigenvector matrix when the eigenvalues are degenerate. However,
    # both the real and
    # imaginary part of M must be symmetric with the same orthogonal
    # matrix of eigenvectors. But either the real or imaginary part could
    # vanish. So we use a randomized algorithm where we diagonalize a
    # random linear combination of real and imaginary parts to find the
    # eigenvectors, taking advantage of the 'eigh' subroutine for
    # diagonalizing symmetric matrices.
    # This can fail if we're very unlucky with our random coefficient, so we
    # give the algorithm a few chances to succeed.

    # Empirically, never seems to fail on randomly sampled complex
    # symmetric 4x4 matrices.
    # If failure rate is less than 1 in a million, then 16 rounds
    # will have overall failure rate less than 1 in a googol.
    # However, cannot (yet) guarantee that there aren't special cases
    # which have much higher failure rates.

    # GEC 2018

    max_attempts = 16
    for _ in range(max_attempts):
        c = np.random.uniform(0, 1)
        matrix = c * M.real + (1-c) * M.imag
        _, eigvecs = np.linalg.eigh(matrix)
        eigvecs = np.array(eigvecs, dtype=complex)
        eigvals = np.diag(eigvecs.transpose() @ M @ eigvecs)

        # Finish if we got a correct answer.
        reconstructed = eigvecs @ np.diag(eigvals) @ eigvecs.transpose()
        if np.allclose(M, reconstructed):
            return eigvals, eigvecs

    # Should never happen. Hopefully.
    raise np.linalg.LinAlgError(
        'Cannot diagonalize complex symmetric matrix.')  # pragma: no cover


def _lambdas_to_coords(lambdas: Sequence[float]) -> np.ndarray:
    # [2, eq.11], but using [1]s coordinates.
    l1, l2, _, l4 = lambdas
    c1 = np.real(1j * np.log(l1 * l2))
    c2 = np.real(1j * np.log(l2 * l4))
    c3 = np.real(1j * np.log(l1 * l4))
    coords = np.asarray((c1, c2, c3))/pi

    coords[np.abs(coords-1) < TOLERANCE] = -1
    if all(coords < 0):
        coords += 1

    # If we're close to the boundary, floating point errors can conspire
    # to make it seem that we're never on the inside
    # Fix: If near boundary, reset to boundary

    # Left
    if np.abs(coords[0] - coords[1]) < TOLERANCE:
        coords[1] = coords[0]

    # Front
    if np.abs(coords[1] - coords[2]) < TOLERANCE:
        coords[2] = coords[1]

    # Right
    if np.abs(coords[0]-coords[1]-1/2) < TOLERANCE:
        coords[1] = coords[0]-1/2

    # Base
    coords[np.abs(coords) < TOLERANCE] = 0

    return coords


def _constrain_to_weyl(lambdas: Sequence[float]) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    for permutation in itertools.permutations(range(4)):
        for signs in ([1, 1, 1, 1], [1, 1, -1, -1],
                      [-1, 1, -1, 1], [1, -1, -1, 1]):
            signed_lambdas = lambdas * np.asarray(signs)
            perm = list(permutation)
            lambas_perm = signed_lambdas[perm]

            coords = _lambdas_to_coords(lambas_perm)

            if _in_weyl(*coords):
                return coords, np.asarray(signs), perm

    # Should never get here
    assert False                # pragma: no cover
    return None, None, None     # pragma: no cover


def _in_weyl(tx: float, ty: float, tz: float) -> bool:
    # Note 'tz>0' in second term. This takes care of symmetry across base
    # when tz==0
    return (1/2 >= tx >= ty >= tz >= 0) or (1/2 >= (1-tx) >= ty >= tz > 0)
