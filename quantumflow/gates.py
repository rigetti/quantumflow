
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow Gates and actions on gates.
"""

from typing import TextIO, Union
from functools import reduce
import numpy as np
import scipy.stats

from .config import TOLERANCE
from .qubits import Qubit, Qubits, qubits_count_tuple, asarray
from .qubits import outer_product
from .ops import Gate

__all__ = ['identity_gate',
           'random_gate',
           'join_gates',
           'control_gate',
           'conditional_gate',
           'P0', 'P1',
           'almost_unitary',
           'almost_identity',
           'almost_hermitian',
           'print_gate']


class I(Gate):                                              # noqa: E742
    r"""
    The identity gate.

    This gate can take a variable number of qubits.

    .. math::
        I() \equiv \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}
    """
    def __init__(self, *qubits: Qubit) -> None:
        if not qubits:
            qubits = (0,)
        N = len(qubits)
        tensor = np.eye(2**N)
        super().__init__(qubits=qubits, tensor=tensor)

    @property
    def H(self) -> Gate:
        return self  # Hermitian

    def __pow__(self, t: float) -> Gate:
        return self


def identity_gate(qubits: Union[int, Qubits]) -> Gate:
    """Returns the K-qubit identity gate"""
    _, qubits = qubits_count_tuple(qubits)
    return I(*qubits)


def join_gates(*gates: Gate) -> Gate:
    """Direct product of two gates. Qubit count is the sum of each gate's
    bit count."""
    vectors = [gate.vec for gate in gates]
    vec = reduce(outer_product, vectors)
    return Gate(vec.tensor, vec.qubits)


def control_gate(control: Qubit, gate: Gate) -> Gate:
    """Return a controlled unitary gate. Given a gate acting on K qubits,
    return a new gate on K+1 qubits prepended with a control bit. """

    if control in gate.qubits:
        raise ValueError('Gate and control qubits overlap')

    qubits = [control, *gate.qubits]
    gate_tensor = join_gates(P0(control), identity_gate(gate.qubits)).tensor \
        + join_gates(P1(control), gate).tensor
    controlled_gate = Gate(qubits=qubits, tensor=gate_tensor)

    return controlled_gate


def conditional_gate(control: Qubit, gate0: Gate, gate1: Gate) -> Gate:
    """Return a conditional unitary gate. Do gate0 on bit 1 if bit 0 is zero,
    else do gate1 on 1"""
    assert gate0.qubits == gate1.qubits  # FIXME

    tensor = join_gates(P0(control), gate0).tensor
    tensor += join_gates(P1(control), gate1).tensor
    gate = Gate(tensor=tensor, qubits=[control, *gate0.qubits])
    return gate


def almost_unitary(gate: Gate) -> bool:
    """Return true if gate tensor is (almost) unitary"""
    res = (gate @ gate.H).asoperator()
    N = gate.qubit_nb
    return np.allclose(asarray(res), np.eye(2**N), atol=TOLERANCE)


def almost_identity(gate: Gate) -> bool:
    """Return true if gate tensor is (almost) the identity"""
    N = gate.qubit_nb
    return np.allclose(asarray(gate.asoperator()), np.eye(2**N))


def almost_hermitian(gate: Gate) -> bool:
    """Return true if gate tensor is (almost) Hermitian"""
    return np.allclose(asarray(gate.asoperator()),
                       asarray(gate.H.asoperator()))


def print_gate(gate: Gate, ndigits: int = 2,
               file: TextIO = None) -> None:
    """Pretty print a gate tensor

    Args:
        gate:
        ndigits:
        file: Stream to which to write. Defaults to stdout
    """
    N = gate.qubit_nb
    gate_tensor = gate.vec.asarray()
    lines = []
    for index, amplitude in np.ndenumerate(gate_tensor):
        ket = "".join([str(n) for n in index[0:N]])
        bra = "".join([str(index[n]) for n in range(N, 2*N)])
        if round(abs(amplitude)**2, ndigits) > 0.0:
            lines.append('{} -> {} : {}'.format(bra, ket, amplitude))
    lines.sort(key=lambda x: int(x[0:N]))
    print('\n'.join(lines), file=file)


class P0(Gate):
    r"""Project qubit to zero.

    A non-unitary gate that represents the effect of a measurement. The norm
    of the resultant state is multiplied by the probability of observing 0.
    """
    def __init__(self, q0: Qubit = 0) -> None:
        operator = [[1, 0], [0, 0]]
        super().__init__(operator, qubits=[q0])


class P1(Gate):
    r"""Project qubit to one.

    A non-unitary gate that represents the effect of a measurement. The norm
    of the resultant state is multiplied by the probability of observing 1.
    """
    def __init__(self, q0: Qubit = 0) -> None:
        operator = [[0, 0], [0, 1]]
        super().__init__(operator, qubits=[q0])


def random_gate(qubits: Union[int, Qubits]) -> Gate:
    r"""Returns a random unitary gate on K qubits.

    Ref:
        "How to generate random matrices from the classical compact groups"
        Francesco Mezzadri, math-ph/0609050
    """
    N, qubits = qubits_count_tuple(qubits)
    unitary = scipy.stats.unitary_group.rvs(2**N)
    return Gate(unitary, qubits=qubits, name='RAND{}'.format(N))
