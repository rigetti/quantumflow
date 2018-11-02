

# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: Classical and hybrid quantum-classical operations. (That
are not Gate's or Channel's.
"""

# Callable and State imported for typing pragmas

from abc import ABCMeta  # Abstract Base Class
from typing import Callable, Union
from numbers import Number
import operator

import numpy as np

from .cbits import Addr
from .qubits import Qubit, Qubits, asarray, QubitVector
from .states import State, Density
from .ops import Operation, Gate, Channel
from .gates import P0, P1
from . import backend as bk


__all__ = ['Measure', 'Reset', 'Barrier', 'If',
           'Neg', 'Not',
           'And', 'Ior', 'Or', 'Xor',
           'Add', 'Mul', 'Sub', 'Div',
           'Move', 'Exchange',
           'TRUE', 'FALSE',
           'EQ', 'LT', 'GT', 'LE', 'GE', 'NE']


class Measure(Operation):
    """Measure a quantum bit and copy result to a classical bit"""
    def __init__(self, qubit: Qubit, cbit: Addr = None) -> None:
        self.qubit = qubit
        self.cbit = cbit

    def quil(self) -> str:
        if self.cbit is not None:
            return '{} {} {}'.format(self.name.upper(), self.qubit, self.cbit)
        return '{} {}'.format(self.name.upper(), self.qubit)

    @property
    def qubits(self) -> Qubits:
        return [self.qubit]

    def run(self, ket: State) -> State:
        prob_zero = asarray(P0(self.qubit).run(ket).norm())

        # generate random number to 'roll' for measurement
        if np.random.random() < prob_zero:
            ket = P0(self.qubit).run(ket).normalize()
            if self.cbit is not None:
                ket = ket.update({self.cbit: 0})
        else:  # measure one
            ket = P1(self.qubit).run(ket).normalize()
            if self.cbit is not None:
                ket = ket.update({self.cbit: 1})
        return ket

    def evolve(self, rho: Density) -> Density:
        p0 = P0(self.qubit).aschannel()
        p1 = P1(self.qubit).aschannel()

        prob_zero = asarray(p0.evolve(rho).norm())

        # generate random number to 'roll' for measurement
        if np.random.random() < prob_zero:
            rho = p0.evolve(rho).normalize()
            if self.cbit is not None:
                rho = rho.update({self.cbit: 0})
        else:  # measure one
            rho = p1.evolve(rho).normalize()
            if self.cbit is not None:
                rho = rho.update({self.cbit: 1})
        return rho


class Reset(Operation):
    r"""An operation that resets qubits to zero irrespective of the
    initial state.
    """
    def __init__(self, *qubits: Qubit) -> None:
        if not qubits:
            qubits = ()
        self._qubits = tuple(qubits)

        self.vec = QubitVector([[1, 1], [0, 0]], [0])

    @property
    def H(self) -> 'Reset':
        return self  # Hermitian

    def run(self, ket: State) -> State:
        if self.qubits:
            qubits = self.qubits
        else:
            qubits = ket.qubits

        indices = [ket.qubits.index(q) for q in qubits]
        ket_tensor = ket.tensor
        for idx in indices:
            ket_tensor = bk.tensormul(self.vec.tensor, ket_tensor, [idx])
        ket = State(ket_tensor, ket.qubits, ket.memory).normalize()
        return ket

    def evolve(self, rho: Density) -> Density:
        # TODO
        raise TypeError('Not yet implemented')

    def asgate(self) -> Gate:
        raise TypeError('Reset not convertible to Gate')

    def aschannel(self) -> Channel:
        raise TypeError('Reset not convertible to Channel')

    def quil(self) -> str:
        if self.qubits:
            return 'RESET ' + ' '.join([str(q) for q in self.qubits])
        return 'RESET'


# DOCME
class Barrier(Operation):
    """An operation that prevents reordering of operations across the barrier.
    Has no effect on the quantum state."""
    def __init__(self, *qubits: Qubit) -> None:
        self._qubits = qubits

    @property
    def H(self) -> 'Barrier':
        return self  # Hermitian

    def run(self, ket: State) -> State:
        return ket  # NOP

    def evolve(self, rho: Density) -> Density:
        return rho  # NOP

    def quil(self) -> str:
        return self.name.upper() + ' ' + ' '.join(str(q) for q in self.qubits)


# DOCME
class If(Operation):
    def __init__(self, elem: Operation, condition: Addr, value: bool = True) \
            -> None:
        self.element = elem
        self.value = value
        self.condition = condition

    def run(self, ket: State) -> State:
        print(ket.memory[self.condition], self.value)

        if ket.memory[self.condition] == self.value:
            ket = self.element.run(ket)
        return ket

    def evolve(self, rho: Density) -> Density:
        if rho.memory[self.condition] == self.value:
            rho = self.element.evolve(rho)
        return rho


# Classical operations

class Neg(Operation):
    """Negate value stored in classical memory."""
    def __init__(self, target: Addr) -> None:
        self.target = target
        self.addresses = [target]

    def run(self, ket: State) -> State:
        return ket.update({self.target: - ket.memory[self.target]})

    def quil(self) -> str:
        return '{} {}'.format(self.name.upper(), self.target)


class Not(Operation):
    """Take logical Not of a classical bit."""
    def __init__(self, target: Addr) -> None:
        self.target = target

    def run(self, ket: State) -> State:
        res = int(not ket.memory[self.target])
        return ket.update({self.target: res})

    def quil(self) -> str:
        return '{} {}'.format(self.name.upper(), self.target)


# Binary classical operations

class BinaryOP(Operation, metaclass=ABCMeta):
    _op: Callable

    """Abstract Base Class for operations between two classical addresses"""
    def __init__(self, target: Addr, source: Union[Addr, Number]) -> None:
        self.target = target
        self.source = source

    def _source(self, state: State) -> Union[Addr, Number]:
        if isinstance(self.source, Addr):
            return state.memory[self.source]
        return self.source

    def quil(self) -> str:
        return '{} {} {}'.format(self.name.upper(), self.target, self.source)

    def run(self, ket: State) -> State:
        target = ket.memory[self.target]
        if isinstance(self.source, Addr):
            source = ket.memory[self.source]
        else:
            source = self.source

        print(target, source)
        res = self._op(target, source)
        ket = ket.update({self.target: res})
        return ket

    def evolve(self, rho: Density) -> Density:
        res = self.run(rho)
        assert isinstance(res, Density)  # Make type checker happy
        return res


class And(BinaryOP):
    """Classical logical And of two addresses. Result placed in target"""
    _op = operator.and_


class Ior(BinaryOP):
    """Take logical inclusive-or of two classical bits, and place result
    in first bit."""
    _op = operator.or_


class Or(BinaryOP):
    """Take logical inclusive-or of two classical bits, and place result
    in first bit. (Deprecated in quil. Use Ior instead."""
    _op = operator.or_


class Xor(BinaryOP):
    """Take logical exclusive-or of two classical bits, and place result
    in first bit.
    """
    _op = operator.xor


class Add(BinaryOP):
    """Add two classical values, and place result in target."""
    _op = operator.add


class Sub(BinaryOP):
    """Add two classical values, and place result in target."""
    _op = operator.sub


class Mul(BinaryOP):
    """Add two classical values, and place result in target."""
    _op = operator.mul


class Div(BinaryOP):
    """Add two classical values, and place result in target."""
    _op = operator.truediv


class Move(BinaryOP):
    """Copy left classical bit to right classical bit"""
    def run(self, ket: State) -> State:
        return ket.update({self.target: self._source(ket)})


class Exchange(BinaryOP):
    """Exchange two classical bits"""
    def run(self, ket: State) -> State:
        assert isinstance(self.source, Addr)
        return ket.update({self.target: ket.memory[self.source],
                           self.source: ket.memory[self.target]})


# Comparisons

class Comparison(Operation, metaclass=ABCMeta):
    """Abstract Base Class for classical comparisons"""
    _op: Callable

    def __init__(self, target: Addr, left: Addr, right: Addr) -> None:
        self.target = target
        self.left = left
        self.right = right

    def quil(self) -> str:
        return '{} {} {} {}'.format(self.name, self.target,
                                    self.left, self.right)

    def run(self, ket: State) -> State:
        res = self._op(ket.memory[self.left], ket.memory[self.right])
        ket = ket.update({self.target: res})
        return ket


class EQ(Comparison):
    """Set target to boolean (left==right)"""
    _op = operator.eq


class GT(Comparison):
    """Set target to boolean (left>right)"""
    _op = operator.gt


class GE(Comparison):
    """Set target to boolean (left>=right)"""
    _op = operator.ge


class LT(Comparison):
    """Set target to boolean (left<right)"""
    _op = operator.lt


class LE(Comparison):
    """Set target to boolean (left<=right)"""
    _op = operator.le


class NE(Comparison):
    """Set target to boolean (left!=right)"""
    _op = operator.ne


class TRUE(Operation):
    """Set classical bit to one. (Deprecated in quil. Use Move)"""
    def __init__(self, addr: Addr) -> None:
        self.addr = addr

    def quil(self) -> str:
        return '{} {}'.format(self.name, self.addr)

    def run(self, ket: State) -> State:
        return ket.update({self.addr: 1})


class FALSE(Operation):
    """Set classical bit to zero. (Deprecated in quil. Use Move)"""
    def __init__(self, addr: Addr) -> None:
        # super().__init__()
        self.addr = addr

    def quil(self) -> str:
        return '{} {}'.format(self.name, self.addr)

    def run(self, ket: State) -> State:
        return ket.update({self.addr: 0})
