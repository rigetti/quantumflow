
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: Measures on vectors, states, and quantum operations
"""
import numpy as np

from scipy.linalg import sqrtm  # matrix square root
import scipy.stats
import cvxpy as cvx             # FIXME add to requirements, meta

from . import backend as bk
from .config import TOLERANCE
from .qubits import Qubits, asarray
from .qubits import vectors_close, fubini_study_angle
from .states import State, Density
from .ops import Gate, Channel

__all__ = ['state_fidelity', 'state_angle', 'states_close',
           'purity', 'fidelity', 'bures_distance', 'bures_angle',
           'density_angle', 'densities_close', 'entropy', 'mutual_info',
           'gate_angle', 'channel_angle', 'gates_close', 'channels_close',
           'diamond_norm']


# -- Measures on pure states ---

def state_fidelity(state0: State, state1: State) -> bk.BKTensor:
    """Return the quantum fidelity between pure states."""
    assert state0.qubits == state1.qubits   # FIXME
    tensor = bk.absolute(bk.inner(state0.tensor, state1.tensor))**bk.fcast(2)
    return tensor


def state_angle(ket0: State, ket1: State) -> bk.BKTensor:
    """The Fubini-Study angle between states.

    Equal to the Burrs angle for pure states.
    """
    return fubini_study_angle(ket0.vec, ket1.vec)


def states_close(state0: State, state1: State,
                 tolerance: float = TOLERANCE) -> bool:
    """Returns True if states are almost identical.

    Closeness is measured with the metric Fubini-Study angle.
    """
    return vectors_close(state0.vec, state1.vec, tolerance)


# -- Measures on density matrices ---


def purity(rho: Density) -> bk.BKTensor:
    """
    Calculate the purity of a mixed quantum state.

    Purity, defined as tr(rho^2), has an upper bound of 1 for a pure state,
    and a lower bound of 1/D (where D is the Hilbert space dimension) for a
    competently mixed state.

    Two closely related measures are the linear entropy, 1- purity, and the
    participation ratio, 1/purity.
    """
    tensor = rho.tensor
    N = rho.qubit_nb
    matrix = bk.reshape(tensor, [2**N, 2**N])
    return bk.trace(bk.matmul(matrix, matrix))


def fidelity(rho0: Density, rho1: Density) -> float:
    """Return the fidelity F(rho0, rho1) between two mixed quantum states.

    Note: Fidelity cannot be calculated entirely within the tensor backend.
    """
    assert rho0.qubit_nb == rho1.qubit_nb   # FIXME

    rho1 = rho1.permute(rho0.qubits)
    op0 = asarray(rho0.asoperator())
    op1 = asarray(rho1.asoperator())

    fid = np.real((np.trace(sqrtm(sqrtm(op0) @ op1 @ sqrtm(op0)))) ** 2)
    fid = min(fid, 1.0)
    fid = max(fid, 0.0)     # Correct for rounding errors

    return fid


# DOCME
def bures_distance(rho0: Density, rho1: Density) -> float:
    """Return the Bures distance between mixed quantum states

    Note: Bures distance cannot be calculated within the tensor backend.
    """
    fid = fidelity(rho0, rho1)
    op0 = asarray(rho0.asoperator())
    op1 = asarray(rho1.asoperator())
    tr0 = np.trace(op0)
    tr1 = np.trace(op1)

    return np.sqrt(tr0 + tr1 - 2.*np.sqrt(fid))


# DOCME
def bures_angle(rho0: Density, rho1: Density) -> float:
    """Return the Bures angle between mixed quantum states

    Note: Bures angle cannot be calculated within the tensor backend.
    """
    return np.arccos(np.sqrt(fidelity(rho0, rho1)))


def density_angle(rho0: Density, rho1: Density) -> bk.BKTensor:
    """The Fubini-Study angle between density matrices"""
    return fubini_study_angle(rho0.vec, rho1.vec)


def densities_close(rho0: Density, rho1: Density,
                    tolerance: float = TOLERANCE) -> bool:
    """Returns True if densities are almost identical.

    Closeness is measured with the metric Fubini-Study angle.
    """
    return vectors_close(rho0.vec, rho1.vec, tolerance)


# TESTME
def entropy(rho: Density, base: float = None) -> float:
    """
    Returns the von-Neumann entropy of a mixed quantum state.

    Args:
        rho:    A density matrix
        base:   Optional logarithm base. Default is base e, and entropy is
                measures in nats. For bits set base to 2.

    Returns:
        The von-Neumann entropy of rho
    """
    op = asarray(rho.asoperator())
    probs = np.linalg.eigvalsh(op)
    probs = np.maximum(probs, 0.0)  # Compensate for floating point errors
    return scipy.stats.entropy(probs, base=base)


# TESTME
def mutual_info(rho: Density,
                qubits0: Qubits,
                qubits1: Qubits = None,
                base: float = None) -> float:
    """Compute the bipartite von-Neumann mutual information of a mixed
    quantum state.

    Args:
        rho:    A density matrix of the complete system
        qubits0: Qubits of system 0
        qubits1: Qubits of system 1. If none, taken to be all remaining qubits
        base:   Optional logarithm base. Default is base e

    Returns:
        The bipartite von-Neumann mutual information.
    """
    if qubits1 is None:
        qubits1 = tuple(set(rho.qubits) - set(qubits0))

    rho0 = rho.partial_trace(qubits1)
    rho1 = rho.partial_trace(qubits0)

    ent = entropy(rho, base)
    ent0 = entropy(rho0, base)
    ent1 = entropy(rho1, base)

    return ent0 + ent1 - ent


# Measures on gates

def gate_angle(gate0: Gate, gate1: Gate) -> bk.BKTensor:
    """The Fubini-Study angle between gates"""
    return fubini_study_angle(gate0.vec, gate1.vec)


def gates_close(gate0: Gate, gate1: Gate,
                tolerance: float = TOLERANCE) -> bool:
    """Returns: True if gates are almost identical.

    Closeness is measured with the gate angle.
    """
    return vectors_close(gate0.vec, gate1.vec, tolerance)


# Measures on channels

def channel_angle(chan0: Channel, chan1: Channel) -> bk.BKTensor:
    """The Fubini-Study angle between channels"""
    return fubini_study_angle(chan0.vec, chan1.vec)


def channels_close(chan0: Channel, chan1: Channel,
                   tolerance: float = TOLERANCE) -> bool:
    """Returns: True if channels are almost identical.

    Closeness is measured with the channel angle.
    """
    return vectors_close(chan0.vec, chan1.vec, tolerance)


def diamond_norm(chan0: Channel, chan1: Channel) -> float:
    """Return the diamond norm between two completely positive
    trace-preserving (CPTP) superoperators.

    The calculation uses the simplified semidefinite program of Watrous
    [arXiv:0901.4709](http://arxiv.org/abs/0901.4709)
    [J. Watrous, [Theory of Computing 5, 11, pp. 217-238
    (2009)](http://theoryofcomputing.org/articles/v005a011/)]
    """
    # Kudos: Based on MatLab code written by Marcus P. da Silva
    # (https://github.com/BBN-Q/matlab-diamond-norm/)

    if set(chan0.qubits) != set(chan1.qubits):
        raise ValueError('Channels must operate on same qubits')

    if chan0.qubits != chan1.qubits:
        chan1 = chan1.permute(chan0.qubits)

    N = chan0.qubit_nb
    dim = 2**N

    choi0 = asarray(chan0.choi())
    choi1 = asarray(chan1.choi())

    delta_choi = choi0 - choi1

    # Density matrix must be Hermitian, positive semidefinite, trace 1
    rho = cvx.Variable([dim, dim], complex=True)
    constraints = [rho == rho.H]
    constraints += [rho >> 0]
    constraints += [cvx.trace(rho) == 1]

    # W must be Hermitian, positive semidefinite
    W = cvx.Variable([dim**2, dim**2], complex=True)
    constraints += [W == W.H]
    constraints += [W >> 0]

    constraints += [(W - cvx.kron(np.eye(dim), rho)) << 0]

    J = cvx.Parameter([dim**2, dim**2], complex=True)
    objective = cvx.Maximize(cvx.real(cvx.trace(J.H * W)))

    prob = cvx.Problem(objective, constraints)

    J.value = delta_choi
    prob.solve()

    dnorm = prob.value * 2

    return dnorm

# fin
