
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow States and actions on states.
"""

from math import sqrt
from typing import Union, TextIO, Sequence, Dict, Any
from functools import reduce
from collections import ChainMap, defaultdict

import numpy as np

from . import backend as bk
from .cbits import Addr
from .qubits import Qubits, QubitVector, qubits_count_tuple
from .qubits import outer_product

__all__ = ['State', 'ghz_state',
           'join_states', 'print_probabilities', 'print_state',
           'random_state', 'w_state', 'zero_state',
           'Density', 'mixed_density', 'random_density', 'join_densities']


class State:
    """The quantum state of a collection of qubits.

    Note that memory usage grows exponentially with the number of qubits.
    (16*2^N bytes for N qubits)

    """

    def __init__(self,
                 tensor: bk.TensorLike,
                 qubits: Qubits = None,
                 memory: Dict[Addr, Any] = None) -> None:  # DOCME TESTME
        """Create a new State from a tensor of qubit amplitudes

        Args:
            tensor: A vector or tensor of state amplitudes
            qubits: A sequence of qubit names.
                (Defaults to integer indices, e.g. [0, 1, 2] for 3 qubits)
            memory: Classical memory.
        """
        if qubits is None:
            tensor = bk.astensorproduct(tensor)
            bits = bk.rank(tensor)
            qubits = range(bits)

        self.vec = QubitVector(tensor, qubits)
        self._memory = memory if memory is not None else {}

    @property
    def tensor(self) -> bk.BKTensor:
        """Returns the tensor representation of state vector"""
        return self.vec.tensor

    @property
    def qubits(self) -> Qubits:
        """Return qubit labels of this state"""
        return self.vec.qubits

    @property
    def qubit_nb(self) -> int:
        """Return the total number of qubits"""
        return self.vec.qubit_nb

    def norm(self) -> bk.BKTensor:
        """Return the state vector norm"""
        return self.vec.norm()

    # DOCME TESTME
    @property
    def memory(self) -> dict:
        return defaultdict(int, self._memory)

    # DOCME TESTME
    def update(self, memory: Dict[Addr, Any]) -> 'State':
        mem = self.memory
        mem.update(memory)
        return State(self.tensor, self.qubits, mem)

    # DOCME TESTME
    @property
    def cbits(self) -> Sequence[Addr]:
        cbs = [addr for addr in self._memory if addr.dtype == 'BIT']
        cbs.sort()
        return tuple(cbs)

    # DOCME TESTME
    @property
    def cbit_nb(self) -> int:
        return len(self.cbits)

    def relabel(self, qubits: Qubits) -> 'State':
        """Return a copy of this state with new qubits"""
        return State(self.vec.tensor, qubits, self._memory)

    def permute(self, qubits: Qubits) -> 'State':
        """Return a copy of this state with qubit labels permuted"""
        vec = self.vec.permute(qubits)
        return State(vec.tensor, vec.qubits, self._memory)

    def normalize(self) -> 'State':
        """Normalize the state"""
        tensor = self.tensor / bk.ccast(bk.sqrt(self.norm()))
        return State(tensor, self.qubits, self._memory)

    def probabilities(self) -> bk.BKTensor:
        """
        Returns:
            The state probabilities
        """
        value = bk.absolute(self.tensor)
        return value * value

    def sample(self, trials: int) -> np.ndarray:
        """Measure the state in the computational basis the the given number
        of trials, and return the counts of each output configuration.
        """
        # TODO: Can we do this within backend?
        probs = np.real(bk.evaluate(self.probabilities()))
        res = np.random.multinomial(trials, probs.ravel())
        res = res.reshape(probs.shape)
        return res

    def expectation(self, diag_hermitian: bk.TensorLike,
                    trials: int = None) -> bk.BKTensor:
        """Return the expectation of a measurement. Since we can only measure
        our computer in the computational basis, we only require the diagonal
        of the Hermitian in that basis.

        If the number of trials is specified, we sample the given number of
        times. Else we return the exact expectation (as if we'd performed an
        infinite number of trials. )
        """
        if trials is None:
            probs = self.probabilities()
        else:
            probs = bk.real(bk.astensorproduct(self.sample(trials) / trials))

        diag_hermitian = bk.astensorproduct(diag_hermitian)
        return bk.sum(bk.real(diag_hermitian) * probs)

    def measure(self) -> np.ndarray:
        """Measure the state in the computational basis.

        Returns:
            A [2]*bits array of qubit states, either 0 or 1
        """
        # TODO: Can we do this within backend?
        probs = np.real(bk.evaluate(self.probabilities()))
        indices = np.asarray(list(np.ndindex(*[2] * self.qubit_nb)))
        res = np.random.choice(probs.size, p=probs.ravel())
        res = indices[res]
        return res

    def asdensity(self) -> 'Density':
        """Convert a pure state to a density matrix"""
        matrix = bk.outer(self.tensor, bk.conj(self.tensor))
        return Density(matrix, self.qubits, self._memory)

    def __str__(self) -> str:
        state = self.vec.asarray()
        s = []
        count = 0
        MAX_ELEMENTS = 64
        for index, amplitude in np.ndenumerate(state):
            if not np.isclose(amplitude, 0.0):
                ket = '|' + ''.join([str(n) for n in index]) + '>'
                s.append('({c.real:0.04g}{c.imag:+0.04g}i) {k}'
                         .format(c=amplitude, k=ket))
                count += 1
                if count > MAX_ELEMENTS:
                    s.append('...')
                    break
        return ' + '.join(s)

# End class State


def zero_state(qubits: Union[int, Qubits]) -> State:
    """Return the all-zero state on N qubits"""
    N, qubits = qubits_count_tuple(qubits)
    ket = np.zeros(shape=[2] * N)
    ket[(0,) * N] = 1
    return State(ket, qubits)


def w_state(qubits: Union[int, Qubits]) -> State:
    """Return a W state on N qubits"""
    N, qubits = qubits_count_tuple(qubits)
    ket = np.zeros(shape=[2] * N)
    for n in range(N):
        idx = np.zeros(shape=N, dtype=int)
        idx[n] += 1
        ket[tuple(idx)] = 1 / sqrt(N)
    return State(ket, qubits)


def ghz_state(qubits: Union[int, Qubits]) -> State:
    """Return a GHZ state on N qubits"""
    N, qubits = qubits_count_tuple(qubits)
    ket = np.zeros(shape=[2] * N)
    ket[(0, ) * N] = 1 / sqrt(2)
    ket[(1, ) * N] = 1 / sqrt(2)
    return State(ket, qubits)


def random_state(qubits: Union[int, Qubits]) -> State:
    """Return a random state from the space of N qubits"""
    N, qubits = qubits_count_tuple(qubits)
    ket = np.random.normal(size=([2] * N)) \
        + 1j * np.random.normal(size=([2] * N))
    return State(ket, qubits).normalize()


# == Actions on States ==


def join_states(*states: State) -> State:
    """Join two state vectors into a larger qubit state"""
    vectors = [ket.vec for ket in states]
    vec = reduce(outer_product, vectors)
    return State(vec.tensor, vec.qubits)


# = Output =

def print_state(state: State, file: TextIO = None) -> None:
    """Print a state vector"""
    state = state.vec.asarray()
    for index, amplitude in np.ndenumerate(state):
        ket = "".join([str(n) for n in index])
        print(ket, ":", amplitude, file=file)


# TODO: Should work for density also. Check
def print_probabilities(state: State, ndigits: int = 4,
                        file: TextIO = None) -> None:
    """
    Pretty print state probabilities.

    Args:
        state:
        ndigits: Number of digits of accuracy
        file: Output stream (Defaults to stdout)
    """
    prob = bk.evaluate(state.probabilities())
    for index, prob in np.ndenumerate(prob):
        prob = round(prob, ndigits)
        if prob == 0.0:
            continue
        ket = "".join([str(n) for n in index])
        print(ket, ":", prob, file=file)


# --  Mixed Quantum States --

class Density(State):
    """A density matrix representation of a mixed quantum state"""
    def __init__(self,
                 tensor: bk.TensorLike,
                 qubits: Qubits = None,
                 memory: Dict[Addr, Any] = None) -> None:
        if qubits is None:
            tensor = bk.astensorproduct(tensor)
            bits = bk.rank(tensor) // 2
            qubits = range(bits)

        super().__init__(tensor, qubits, memory)

    def trace(self) -> bk.BKTensor:
        """Return the trace of this density operator"""
        return self.vec.trace()

    # TESTME
    def partial_trace(self, qubits: Qubits) -> 'Density':
        """Return the partial trace over the specified qubits"""
        vec = self.vec.partial_trace(qubits)
        return Density(vec.tensor, vec.qubits, self._memory)

    def relabel(self, qubits: Qubits) -> 'Density':
        """Return a copy of this state with new qubits"""
        return Density(self.vec.tensor, qubits, self._memory)

    def permute(self, qubits: Qubits) -> 'Density':
        """Return a copy of this state with qubit labels permuted"""
        vec = self.vec.permute(qubits)
        return Density(vec.tensor, vec.qubits, self._memory)

    def normalize(self) -> 'Density':
        """Normalize state"""
        tensor = self.tensor / self.trace()
        return Density(tensor, self.qubits, self._memory)

    # TESTME
    def probabilities(self) -> bk.BKTensor:
        """Returns: The state probabilities """
        prob = bk.productdiag(self.tensor)
        return prob

    def asoperator(self) -> bk.BKTensor:
        """Return the density matrix as a square array"""
        return self.vec.flatten()

    def asdensity(self) -> 'Density':
        """Returns self"""
        return self

    # DOCME TESTME
    def update(self, memory: Dict[Addr, Any]) -> 'Density':
        mem = self.memory
        mem.update(memory)
        return Density(self.tensor, self.qubits, mem)


def mixed_density(qubits: Union[int, Qubits]) -> Density:
    """Returns the completely mixed density matrix"""
    N, qubits = qubits_count_tuple(qubits)
    matrix = np.eye(2**N) / 2**N
    return Density(matrix, qubits)


def random_density(qubits: Union[int, Qubits]) -> Density:
    """
    Returns: A randomly sampled Density from the Hilbert–Schmidt
                ensemble of quantum states

    Ref: "Induced measures in the space of mixed quantum states" Karol
        Zyczkowski, Hans-Juergen Sommers, J. Phys. A34, 7111-7125 (2001)
        https://arxiv.org/abs/quant-ph/0012101
    """
    N, qubits = qubits_count_tuple(qubits)
    size = (2**N, 2**N)
    ginibre_ensemble = (np.random.normal(size=size) +
                        1j * np.random.normal(size=size)) / np.sqrt(2.0)
    matrix = ginibre_ensemble @ np.transpose(np.conjugate(ginibre_ensemble))
    matrix /= np.trace(matrix)

    return Density(matrix, qubits=qubits)


# TESTME
def join_densities(*densities: Density) -> Density:
    """Join two mixed states into a larger qubit state"""
    vectors = [rho.vec for rho in densities]
    vec = reduce(outer_product, vectors)

    memory = dict(ChainMap(*[rho.memory for rho in densities]))  # TESTME
    return Density(vec.tensor, vec.qubits, memory)

# fin
