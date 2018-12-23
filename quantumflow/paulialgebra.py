# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: Module for working with the Pauli algebra.
"""

# Kudos: Adapted from PyQuil's paulis.py, original written by Nick Rubin

# TODO:
# pauli_close

from typing import Tuple, Any, Iterator, List
from operator import itemgetter, mul
from functools import reduce, total_ordering
from itertools import groupby, product
import heapq
from cmath import isclose  # type: ignore
from numbers import Complex

from quantumflow.qubits import Qubit, Qubits


__all__ = ['PauliTerm', 'Pauli', 'sX', 'sY', 'sZ', 'sI',
           'pauli_sum', 'pauli_product', 'pauli_pow', 'paulis_commute',
           'pauli_commuting_sets']

PauliTerm = Tuple[Tuple[Tuple[Qubit, str], ...], complex]

PAULI_OPS = ["X", "Y", "Z", "I"]

PAULI_PROD = {'ZZ': ('I', 1.0),
              'YY': ('I', 1.0),
              'XX': ('I', 1.0),
              'II': ('I', 1.0),
              'XY': ('Z', 1.0j),
              'XZ': ('Y', -1.0j),
              'YX': ('Z', -1.0j),
              'YZ': ('X', 1.0j),
              'ZX': ('Y', 1.0j),
              'ZY': ('X', -1.0j),
              'IX': ('X', 1.0),
              'IY': ('Y', 1.0),
              'IZ': ('Z', 1.0),
              'ZI': ('Z', 1.0),
              'YI': ('Y', 1.0),
              'XI': ('X', 1.0)}


@total_ordering
class Pauli:
    """
    An element of the Pauli algebra.

    An element of the Pauli algebra is a sequence of terms, such as

        Y(1) - 0.5 Z(1) X(2) Y(4)

    where X, Y, Z and I are the 1-qubit Pauli operators.

    """

    # Internally, each term is a tuple of a complex coefficient, and a sequence
    # of single qubit Pauli operators. (The coefficient goes last so that the
    # terms sort on the operators).
    #
    # PauliTerm = Tuple[Tuple[Tuple[Qubit, str], ...], complex]
    #
    # Each Pauli operator consists of a tuple of
    # qubits e.g. (0, 1, 3), a tuple of Pauli operators e.g. ('X', 'Y', 'Z').
    # Qubits and Pauli terms are kept in sorted order. This ensures that a
    # Pauli element has a unique representation, and makes summation and
    # simplification efficient. We use Tuples (and not lists) because they are
    # immutable and hashable.

    terms: Tuple[PauliTerm, ...]

    def __init__(self, terms: Tuple[PauliTerm, ...]) -> None:
        self.terms = terms

    @classmethod
    def term(cls, qubits: Qubits, ops: str,
             coefficient: complex = 1.0) -> 'Pauli':
        """
        Create an element of the Pauli algebra from a sequence of qubits
        and operators. Qubits must be unique and sortable
        """
        if not all(op in PAULI_OPS for op in ops):
            raise ValueError("Valid Pauli operators are I, X, Y, and Z")

        coeff = complex(coefficient)

        terms = ()  # type: Tuple[PauliTerm, ...]
        if isclose(coeff, 0.0):
            terms = ()
        else:
            qops = zip(qubits, ops)
            qops = filter(lambda x: x[1] != 'I', qops)
            terms = ((tuple(sorted(qops)), coeff),)

        return cls(terms)

    @classmethod
    def sigma(cls, qubit: Qubit, operator: str,
              coefficient: complex = 1.0) -> 'Pauli':
        """Returns a Pauli operator ('I', 'X', 'Y', or 'Z') acting
        on the given qubit"""
        if operator == 'I':
            return cls.scalar(coefficient)
        return cls.term([qubit], operator, coefficient)

    @classmethod
    def scalar(cls, coefficient: complex) -> 'Pauli':
        """Return a scalar multiple of the Pauli identity element."""
        return cls.term((), '', coefficient)

    def is_scalar(self) -> bool:
        """Returns true if this object is a scalar multiple of the Pauli
        identity element"""
        if len(self.terms) > 1:
            return False
        if len(self.terms) == 0:
            return True  # Zero element
        if self.terms[0][0] == ():
            return True
        return False

    @classmethod
    def identity(cls) -> 'Pauli':
        """Return the identity element of the Pauli algebra"""
        return cls.scalar(1.0)

    def is_identity(self) -> bool:
        """Returns True if this object is identity Pauli element."""

        if len(self) != 1:
            return False
        if self.terms[0][0] != ():
            return False
        return isclose(self.terms[0][1], 1.0)

    @classmethod
    def zero(cls) -> 'Pauli':
        """Return the zero element of the Pauli algebra"""
        return cls(())

    def is_zero(self) -> bool:
        """Return True if this object is the zero Pauli element."""
        return len(self.terms) == 0

    @property
    def qubits(self) -> Qubits:
        """Return a list of qubits acted upon by the Pauli element"""
        return list({q for term, _ in self.terms
                     for q, _ in term})  # type: ignore

    def __repr__(self) -> str:
        return 'Pauli(' + str(self.terms) + ')'

    def __str__(self) -> str:
        out = []
        for term in self.terms:
            out.append('+ {:+}'.format(term[1]))

            for q, op in term[0]:
                out.append(op+'('+str(q)+')')

        return ' '.join(out)

    def __iter__(self) -> Iterator[PauliTerm]:
        return iter(self.terms)

    def __len__(self) -> int:
        return len(self.terms)

    def __add__(self, other: Any) -> 'Pauli':
        if isinstance(other, Complex):
            other = Pauli.scalar(complex(other))
        return pauli_sum(self, other)

    def __radd__(self, other: Any) -> 'Pauli':
        return self.__add__(other)

    def __mul__(self, other: Any) -> 'Pauli':
        if isinstance(other, Complex):
            other = Pauli.scalar(complex(other))
        return pauli_product(self, other)

    def __rmul__(self, other: Any) -> 'Pauli':
        return self.__mul__(other)

    def __sub__(self, other: Any) -> 'Pauli':
        return self + -1. * other

    def __rsub__(self, other: Any) -> 'Pauli':
        return other + -1. * self

    def __neg__(self) -> 'Pauli':
        return self * -1

    def __pos__(self) -> 'Pauli':
        return self

    def __pow__(self, exponent: int) -> 'Pauli':
        return pauli_pow(self, exponent)

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Pauli):
            return NotImplemented
        return self.terms < other.terms

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Pauli):
            return NotImplemented
        return self.terms == other.terms

    def __hash__(self) -> int:
        return hash(self.terms)

# End class Pauli


def sX(qubit: Qubit, coefficient: complex = 1.0) -> Pauli:
    """Return the Pauli sigma_X operator acting on the given qubit"""
    return Pauli.sigma(qubit, 'X', coefficient)


def sY(qubit: Qubit, coefficient: complex = 1.0) -> Pauli:
    """Return the Pauli sigma_Y operator acting on the given qubit"""
    return Pauli.sigma(qubit, 'Y', coefficient)


def sZ(qubit: Qubit, coefficient: complex = 1.0) -> Pauli:
    """Return the Pauli sigma_Z operator acting on the given qubit"""
    return Pauli.sigma(qubit, 'Z', coefficient)


def sI(qubit: Qubit, coefficient: complex = 1.0) -> Pauli:
    """Return the Pauli sigma_I (identity) operator. The qubit is irrelevant,
    but kept as an  argument for consistency"""
    return Pauli.sigma(qubit, 'I', coefficient)


def pauli_sum(*elements: Pauli) -> Pauli:
    """Return the sum of elements of the Pauli algebra"""
    terms = []

    key = itemgetter(0)
    for term, grp in groupby(heapq.merge(*elements, key=key), key=key):
        coeff = sum(g[1] for g in grp)
        if not isclose(coeff, 0.0):
            terms.append((term, coeff))

    return Pauli(tuple(terms))


def pauli_product(*elements: Pauli) -> Pauli:
    """Return the product of elements of the Pauli algebra"""
    result_terms = []

    for terms in product(*elements):
        coeff = reduce(mul, [term[1] for term in terms])
        ops = (term[0] for term in terms)
        out = []
        key = itemgetter(0)
        for qubit, qops in groupby(heapq.merge(*ops, key=key), key=key):
            res = next(qops)[1]  # Operator: X Y Z
            for op in qops:
                pair = res + op[1]
                res, rescoeff = PAULI_PROD[pair]
                coeff *= rescoeff
            if res != 'I':
                out.append((qubit, res))

        p = Pauli(((tuple(out), coeff),))
        result_terms.append(p)

    return pauli_sum(*result_terms)


def pauli_pow(pauli: Pauli, exponent: int) -> Pauli:
    """
    Raise an element of the Pauli algebra to a non-negative integer power.
    """

    if not isinstance(exponent, int) or exponent < 0:
        raise ValueError("The exponent must be a non-negative integer.")

    if exponent == 0:
        return Pauli.identity()

    if exponent == 1:
        return pauli

    # https://en.wikipedia.org/wiki/Exponentiation_by_squaring
    y = Pauli.identity()
    x = pauli
    n = exponent
    while n > 1:
        if n % 2 == 0:  # Even
            x = x * x
            n = n // 2
        else:           # Odd
            y = x * y
            x = x * x
            n = (n - 1) // 2
    return x * y


def paulis_commute(element0: Pauli, element1: Pauli) -> bool:
    """
    Return true if the two elements of the Pauli algebra commute.
    i.e. if element0 * element1 == element1 * element0

    Derivation similar to arXiv:1405.5749v2 for the check_commutation step in
    the Raesi, Wiebe, Sanders algorithm (arXiv:1108.4318, 2011).
    """

    def _coincident_parity(term0: PauliTerm, term1: PauliTerm) -> bool:
        non_similar = 0
        key = itemgetter(0)

        op0 = term0[0]
        op1 = term1[0]
        for _, qops in groupby(heapq.merge(op0, op1, key=key), key=key):

            listqops = list(qops)
            if len(listqops) == 2 and listqops[0][1] != listqops[1][1]:
                non_similar += 1
        return non_similar % 2 == 0

    for term0, term1 in product(element0, element1):
        if not _coincident_parity(term0, term1):
            return False

    return True


def pauli_commuting_sets(element: Pauli) -> Tuple[Pauli, ...]:
    """Gather the terms of a Pauli polynomial into commuting sets.

    Uses algorithm defined in (Raeisi, Wiebe, Sanders, arXiv:1108.4318, 2011)
    to find commuting sets. Except uses commutation check from
    arXiv:1405.5749v2
    """
    if len(element) < 2:
        return (element,)

    groups: List[Pauli] = []  # typing: List[Pauli]

    for term in element:
        pterm = Pauli((term,))

        assigned = False
        for i, grp in enumerate(groups):
            if paulis_commute(grp, pterm):
                groups[i] = grp + pterm
                assigned = True
                break
        if not assigned:
            groups.append(pterm)

    return tuple(groups)
