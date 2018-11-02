
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Quantum Circuits
"""

from typing import Sequence, Iterator, Iterable, Dict, Type
from math import pi
from itertools import chain
from collections import defaultdict

from .qubits import Qubit, Qubits
from .states import State, Density, zero_state
from .ops import Operation, Gate, Channel
from .gates import control_gate, identity_gate
from .stdgates import H, CPHASE, SWAP, CNOT, T, X, TY, TZ, CCNOT

__all__ = ['Circuit',
           'count_operations',
           'map_gate',
           'qft_circuit',
           'reversal_circuit',
           'control_circuit',
           'ccnot_circuit',
           'zyz_circuit',
           'phase_estimation_circuit',
           'addition_circuit',
           'ghz_circuit']


class Circuit(Operation):
    """A quantum Circuit contains a sequences of circuit elements.
    These can be any quantum Operation, including other circuits.

    QuantumFlow's circuit can only contain Operations. They do not contain
    control flow of other classical computations(similar to pyquil's
    protoquil). For hybrid algorithms involving control flow and other
    classical processing use QuantumFlow's Program class.
    """
    def __init__(self, elements: Iterable[Operation] = None) -> None:
        if elements is None:
            elements = []
        self.elements = list(elements)

    def add(self, other: 'Circuit') -> 'Circuit':
        """Concatenate gates and return new circuit"""
        return Circuit(self.elements + other.elements)

    def extend(self, other: Operation) -> None:
        """Append gates from circuit to the end of this circuit"""
        if isinstance(other, Circuit):
            self.elements.extend(other.elements)
        else:
            self.elements.extend([other])

    def __add__(self, other: 'Circuit') -> 'Circuit':
        return self.add(other)

    def __iadd__(self, other: Operation) -> 'Circuit':
        self.extend(other)
        return self

    def __iter__(self) -> Iterator[Operation]:
        return self.elements.__iter__()

    # TESTME
    def size(self) -> int:
        """Return the number of operations in this circuit"""
        return len(self.elements)

    @property
    def qubits(self) -> Qubits:
        """Returns: Sorted list of qubits acted upon by this circuit

        Raises:
            TypeError: If qubits cannot be sorted into unique order.
        """
        qbs = [q for elem in self.elements for q in elem.qubits]    # gather
        qbs = list(set(qbs))                                        # unique
        qbs = sorted(qbs)                                           # sort
        return tuple(qbs)

    def run(self, ket: State = None) -> State:
        """
        Apply the action of this circuit upon a state.

        If no initial state provided an initial zero state will be created.
        """
        if ket is None:
            qubits = self.qubits
            ket = zero_state(qubits=qubits)

        for elem in self.elements:
            ket = elem.run(ket)
        return ket

    # DOCME
    def evolve(self, rho: Density = None) -> Density:
        if rho is None:
            qubits = self.qubits
            rho = zero_state(qubits=qubits).asdensity()

        for elem in self.elements:
            rho = elem.evolve(rho)
        return rho

    def asgate(self) -> Gate:
        """
        Return the action of this circuit as a gate
        """
        gate = identity_gate(self.qubits)
        for elem in self.elements:
            gate = elem.asgate() @ gate
        return gate

    # TESTME
    # DOCME
    def aschannel(self) -> Channel:
        chan = identity_gate(self.qubits).aschannel()
        for elem in self.elements:
            chan = elem.aschannel() @ chan
        return chan

    @property
    def H(self) -> 'Circuit':
        """Returns the Hermitian conjugate of this circuit.
        If all the subsidiary gates are unitary, returns the circuit inverse.
        """
        return Circuit([elem.H for elem in self.elements[::-1]])

    def __str__(self) -> str:
        return '\n'.join([str(elem) for elem in self.elements])

# End class Circuit


def count_operations(elements: Iterable[Operation]) \
        -> Dict[Type[Operation], int]:
    """Return a count of different operation types given a colelction of
    operations, such as a Circuit or DAGCircuit
    """
    op_count: Dict[Type[Operation], int] = defaultdict(int)
    for elem in elements:
        op_count[type(elem)] += 1
    return dict(op_count)


def map_gate(gate: Gate, args: Sequence[Qubits]) -> Circuit:
    """Applies the same gate all input qubits in the argument list.

    >>> circ = qf.map_gate(qf.H(), [[0], [1], [2]])
    >>> print(circ)
    H(0)
    H(1)
    H(2)

    """
    circ = Circuit()

    for qubits in args:
        circ += gate.relabel(qubits)

    return circ


def qft_circuit(qubits: Qubits) -> Circuit:
    """Returns the Quantum Fourier Transform circuit"""
    # Kudos: Adapted from Rigetti Grove, grove/qft/fourier.py

    N = len(qubits)
    circ = Circuit()
    for n0 in range(N):
        q0 = qubits[n0]
        circ += H(q0)
        for n1 in range(n0+1, N):
            q1 = qubits[n1]
            angle = pi / 2 ** (n1-n0)
            circ += CPHASE(angle, q1, q0)
    circ.extend(reversal_circuit(qubits))
    return circ


def reversal_circuit(qubits: Qubits) -> Circuit:
    """Returns a circuit to reverse qubits"""
    N = len(qubits)
    circ = Circuit()
    for n in range(N // 2):
        circ += SWAP(qubits[n], qubits[N-1-n])
    return circ


def control_circuit(controls: Qubits, gate: Gate) -> Circuit:
    """
    Returns a circuit for a target gate controlled by
    a collection of control qubits. [Barenco1995]_

    Uses a number of gates quadratic in the number of control qubits.

    .. [Barenco1995] A. Barenco, C. Bennett, R. Cleve (1995) Elementary Gates
        for Quantum Computation`<https://arxiv.org/abs/quant-ph/9503016>`_
        Sec 7.2
    """
    # Kudos: Adapted from Rigetti Grove's utility_programs.py
    # grove/utils/utility_programs.py::ControlledProgramBuilder

    circ = Circuit()
    if len(controls) == 1:
        q0 = controls[0]
        if isinstance(gate, X):
            circ += CNOT(q0, gate.qubits[0])
        else:
            cgate = control_gate(q0, gate)
            circ += cgate
    else:
        circ += control_circuit(controls[-1:], gate ** 0.5)
        circ += control_circuit(controls[0:-1], X(controls[-1]))
        circ += control_circuit(controls[-1:], gate ** -0.5)
        circ += control_circuit(controls[0:-1], X(controls[-1]))
        circ += control_circuit(controls[0:-1], gate ** 0.5)
    return circ


def ccnot_circuit(qubits: Qubits) -> Circuit:
    """Standard decomposition of CCNOT (Toffoli) gate into
    six CNOT gates (Plus Hadamard and T gates.) [Nielsen2000]_

    .. [Nielsen2000]
        M. A. Nielsen and I. L. Chuang, Quantum Computation and Quantum
        Information, Cambridge University Press (2000).
    """
    if len(qubits) != 3:
        raise ValueError('Expected 3 qubits')

    q0, q1, q2 = qubits

    circ = Circuit()
    circ += H(q2)
    circ += CNOT(q1, q2)
    circ += T(q2).H
    circ += CNOT(q0, q2)
    circ += T(q2)
    circ += CNOT(q1, q2)
    circ += T(q2).H
    circ += CNOT(q0, q2)
    circ += T(q1)
    circ += T(q2)
    circ += H(q2)
    circ += CNOT(q0, q1)
    circ += T(q0)
    circ += T(q1).H
    circ += CNOT(q0, q1)

    return circ


def zyz_circuit(t0: float, t1: float, t2: float, q0: Qubit) -> Circuit:
    """Circuit equivalent of 1-qubit ZYZ gate"""
    circ = Circuit()
    circ += TZ(t0, q0)
    circ += TY(t1, q0)
    circ += TZ(t2, q0)
    return circ


def phase_estimation_circuit(gate: Gate, outputs: Qubits) -> Circuit:
    """Returns a circuit for quantum phase estimation.

    The gate has an eigenvector with eigenvalue e^(i 2 pi phase). To
    run the circuit, the eigenvector should be set on the gate qubits,
    and the output qubits should be in the zero state. After evolution and
    measurement, the output qubits will be (approximately) a binary fraction
    representation of the phase.

    The output registers can be converted with the aid of the
    quantumflow.utils.bitlist_to_int() method.

    >>> import numpy as np
    >>> import quantumflow as qf
    >>> N = 8
    >>> phase = 1/4
    >>> gate = qf.RZ(-4*np.pi*phase, N)
    >>> circ = qf.phase_estimation_circuit(gate, range(N))
    >>> res = circ.run().measure()[0:N]
    >>> est_phase = int(''.join([str(d) for d in res]), 2) / 2**N # To float
    >>> print(phase, est_phase)
    0.25 0.25

    """
    circ = Circuit()
    circ += map_gate(H(), list(zip(outputs)))  # Hadamard on all output qubits

    for cq in reversed(outputs):
        cgate = control_gate(cq, gate)
        circ += cgate
        gate = gate @ gate

    circ += qft_circuit(outputs).H

    return circ


def addition_circuit(
        addend0: Qubits,
        addend1: Qubits,
        carry: Qubits) -> Circuit:
    """Returns a quantum circuit for ripple-carry addition. [Cuccaro2004]_

    Requires two carry qubit (input and output). The result is returned in
    addend1.

    .. [Cuccaro2004]
        A new quantum ripple-carry addition circuit, Steven A. Cuccaro,
        Thomas G. Draper, Samuel A. Kutin, David Petrie Moulton
        arXiv:quant-ph/0410184 (2004)
    """

    if len(addend0) != len(addend1):
        raise ValueError('Number of addend qubits must be equal')

    if len(carry) != 2:
        raise ValueError('Expected 2 carry qubits')

    def _maj(qubits: Qubits) -> Circuit:
        q0, q1, q2 = qubits
        circ = Circuit()
        circ += CNOT(q2, q1)
        circ += CNOT(q2, q0)
        circ += CCNOT(q0, q1, q2)
        return circ

    def _uma(qubits: Qubits) -> Circuit:
        q0, q1, q2 = qubits
        circ = Circuit()
        circ += CCNOT(q0, q1, q2)
        circ += CNOT(q2, q0)
        circ += CNOT(q0, q1)
        return circ

    qubits = [carry[0]] + list(chain.from_iterable(
        zip(reversed(addend1), reversed(addend0)))) + [carry[1]]

    circ = Circuit()

    for n in range(0, len(qubits)-3, 2):
        circ += _maj(qubits[n:n+3])

    circ += CNOT(qubits[-2], qubits[-1])

    for n in reversed(range(0, len(qubits)-3, 2)):
        circ += _uma(qubits[n:n+3])

    return circ


def ghz_circuit(qubits: Qubits) -> Circuit:
    """Returns a circuit that prepares a multi-qubit Bell state from the zero
    state.
    """
    circ = Circuit()

    circ += H(qubits[0])
    for q0 in range(0, len(qubits)-1):
        circ += CNOT(qubits[q0], qubits[q0+1])

    return circ


# Fin
