
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: Channels
"""

# Kudos: Kraus maps originally adapted from Nick Rubin's reference-qvm

from functools import reduce
from operator import add
from typing import Sequence

import numpy as np

from .qubits import Qubit, Qubits, outer_product, asarray
from .ops import Operation, Gate, Channel
from .states import State, Density
from .gates import almost_identity
from .stdgates import I, X, Y, Z

__all__ = ['Kraus',
           'UnitaryMixture',
           'Depolarizing',
           'Damping',
           'Dephasing',
           'join_channels',
           'channel_to_kraus',
           'kraus_iscomplete'
           ]


# DOCME
class Kraus(Operation):
    """A Kraus representation of a quantum channel"""
    # DOCME: operator-value-sum representation

    def __init__(self,
                 operators: Sequence[Gate],
                 weights: Sequence[float] = None) -> None:
        self.operators = operators

        if weights is None:
            weights = [1.]*len(operators)

        self.weights = tuple(weights)

    def asgate(self) -> Gate:
        """Not possible in general. (But see UnitaryMixture)

        Raises: TypeError
        """
        raise TypeError('Not possible in general')

    def aschannel(self) -> Channel:
        """Returns: Action of Kraus operators as a superoperator Channel"""
        qubits = self.qubits
        N = len(qubits)
        ident = Gate(np.eye(2**N), qubits=qubits).aschannel()

        channels = [op.aschannel() @ ident for op in self.operators]
        if self.weights is not None:
            channels = [c*w for c, w in zip(channels, self.weights)]
        channel = reduce(add, channels)
        return channel

    def run(self, ket: State) -> State:
        """Apply the action of this Kraus quantum operation upon a state"""
        res = [op.run(ket) for op in self.operators]
        probs = [asarray(ket.norm()) * w for ket, w in zip(res, self.weights)]
        probs = np.asarray(probs)
        probs /= np.sum(probs)
        newket = np.random.choice(res, p=probs)
        return newket.normalize()

    def evolve(self, rho: Density) -> Density:
        """Apply the action of this Kraus quantum operation upon a density"""
        qubits = rho.qubits
        results = [op.evolve(rho) for op in self.operators]
        tensors = [rho.tensor * w for rho, w in zip(results, self.weights)]
        tensor = reduce(add, tensors)
        return Density(tensor, qubits)

    @property
    def qubits(self) -> Qubits:
        """Returns: List of qubits acted upon by this Kraus operation

        The list of qubits is ordered if the qubits labels can be sorted,
        else the the order is indeterminate.

        Raises:
            TypeError: If qubits cannot be sorted into unique order.
        """
        qbs = [q for elem in self.operators for q in elem.qubits]   # gather
        qbs = list(set(qbs))                                        # unique
        qbs = sorted(qbs)                                           # sort
        return tuple(qbs)

    @property
    def H(self) -> 'Kraus':
        """Return the complex conjugate of this Kraus operation"""
        operators = [op.H for op in self.operators]
        return Kraus(operators, self.weights)

# End class Kraus


class UnitaryMixture(Kraus):
    """A Kraus channel which is a convex mixture of unitary dynamics."""
    # def __init__(self,
    #              operators: Sequence[Gate],
    #              weights: Sequence[float] = None) -> None:
    #     super().__init__(operators, weights)
    #     # TODO: Sanity check. operators unitary, weights unit

    def asgate(self) -> Gate:
        """Return one of the composite Kraus operators at random with
        the appropriate weights"""
        return np.random.choice(self.operators, p=self.weights)

    def run(self, ket: State) -> State:
        return self.asgate().run(ket)


class Depolarizing(UnitaryMixture):
    """A Kraus representation of a depolarizing channel on 1-qubit.

    Args:
        prob:   The one-step depolarizing probability.
        q0:     The qubit on which to act.
    """
    def __init__(self, prob: float, q0: Qubit) -> None:
        operators = [I(q0), X(q0), Y(q0), Z(q0)]
        weights = [1 - prob, prob/3.0, prob/3.0, prob/3.0]
        super().__init__(operators, weights)


class Damping(Kraus):
    """A Kraus representation of an amplitude-damping (spontaneous emission)
    channel on one qubit

    Args:
        prob:   The one-step damping probability.
        q0:     The qubit on which to act.
    """

    def __init__(self, prob: float, q0: Qubit) -> None:
        kraus0 = Gate([[1.0, 0.0], [0.0, np.sqrt(1 - prob)]], qubits=[q0])
        kraus1 = Gate([[0.0, np.sqrt(prob)], [0.0, 0.0]], qubits=[q0])
        super().__init__([kraus0, kraus1])


class Dephasing(UnitaryMixture):
    """A Kraus representation of a phase-damping quantum channel

    Args:
        prob:   The one-step damping probability.
        q0:     The qubit on which to act.
    """
    def __init__(self, prob: float, q0: Qubit) -> None:
        operators = [I(q0), Z(q0)]
        weights = [1 - prob/2, prob/2]
        super().__init__(operators, weights)


# TESTME
def join_channels(*channels: Channel) -> Channel:
    """Join two channels acting on different qubits into a single channel
    acting on all qubits"""
    vectors = [chan.vec for chan in channels]
    vec = reduce(outer_product, vectors)
    return Channel(vec.tensor, vec.qubits)


# TESTME
# DOCME
# FIXME
def channel_to_kraus(chan: Channel) -> 'Kraus':
    """Convert a channel superoperator into a Kraus operator representation
    of the same channel."""
    qubits = chan.qubits
    N = chan.qubit_nb

    choi = asarray(chan.choi())
    evals, evecs = np.linalg.eig(choi)
    evecs = np.transpose(evecs)

    assert np.allclose(evals.imag, 0.0)  # FIXME exception
    assert np.all(evals.real >= 0.0)     # FIXME exception

    values = np.sqrt(evals.real)

    ops = []
    for i in range(2**(2*N)):
        if not np.isclose(values[i], 0.0):
            mat = np.reshape(evecs[i], (2**N, 2**N))*values[i]
            g = Gate(mat, qubits)
            ops.append(g)

    return Kraus(ops)


# TESTME DOCME
def kraus_iscomplete(kraus: Kraus) -> bool:
    """Returns True if the collection of (weighted) Kraus operators are
    complete. (Which is necessary for a CPTP map to preserve trace)
    """
    qubits = kraus.qubits
    N = kraus.qubit_nb

    ident = Gate(np.eye(2**N), qubits)  # FIXME

    tensors = [(op.H @ op @ ident).asoperator() for op in kraus.operators]
    tensors = [t*w for t, w in zip(tensors, kraus.weights)]

    tensor = reduce(np.add, tensors)
    res = Gate(tensor, qubits)

    return almost_identity(res)

# Fin
