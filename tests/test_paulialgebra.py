"""
Unit tests for quantumflow.paulialgebra
"""

from itertools import product

import pytest

import quantumflow as qf
from quantumflow.paulialgebra import PAULI_OPS, sI, sX, sY, sZ


def test_term():
    x = qf.Pauli.term([0], 'X', -1)
    assert x.terms == ((((0, 'X'),), (-1+0j)),)

    x = qf.Pauli.term([1, 0, 5], 'XYZ', 2j)
    assert x.terms == ((((0, 'Y'), (1, 'X'), (5, 'Z')), (0+2j)),)

    with pytest.raises(ValueError):
        qf.Pauli.term([1, 0, 5], 'BYZ', 2j)


def test_pauli_str():
    x = qf.Pauli.term([2], 'X', -1)
    assert str(x) == '+ (-1+0j) X(2)'
    assert repr(x) == "Pauli(((((2, 'X'),), (-1+0j)),))"


def test_pauli_sigma():
    assert qf.Pauli.term([0], 'X', 1) == qf.Pauli.sigma(0, 'X')
    assert qf.Pauli.term([1], 'Y', 2) == qf.Pauli.sigma(1, 'Y', 2)
    assert qf.Pauli.term([2], 'Z') == qf.Pauli.sigma(2, 'Z')
    # assert qf.Pauli.term([3], 'I', 1) == qf.Pauli.sigma('I', 3) # FIXME


def test_sigmas():
    assert qf.sX(0).terms == ((((0, 'X'),), (1+0j)),)
    assert qf.sY(1, 2.).terms == ((((1, 'Y'),), (2+0j)),)
    assert qf.sZ(2).terms == ((((2, 'Z'),), (1+0j)),)


def test_sum():
    x = qf.sX(1)
    y = qf.sY(2, 2.0)
    s = qf.pauli_sum(x, y)
    assert ((((1, 'X'),), (1+0j)), (((2, 'Y'),), (2+0j))) == s.terms

    s2 = qf.pauli_sum(x, x)
    assert ((((1, 'X'),), (2+0j)),) == s2.terms

    s3 = qf.pauli_sum(x, x, y)
    s4 = qf.pauli_sum(y, x, x)

    assert s3 == s4
    assert ((((1, 'X'),), (2+0j)), (((2, 'Y'),), (2+0j))) == s3.terms

    qf.pauli_sum(x, x, x)


def test_add():
    x = sX(1)
    y = sY(2, 2.0)
    s = x + y
    assert ((((1, 'X'),), (1+0j)), (((2, 'Y'),), (2+0j))) == s.terms


def test_sub():
    x = sX(1)
    y = sY(2, 2.0)
    s = x - y
    assert ((((1, 'X'),), (1+0j)), (((2, 'Y'),), (-2+0j))) == s.terms

    s = 2 - y
    assert str(s) == '+ (+2+0j) + (-2+0j) Y(2)'


def test_cmp():
    x = sX(1)
    y = sY(2, 2.0)
    assert x < y
    assert x <= y
    assert x <= x
    assert x == x
    assert y > x
    assert y >= y

    with pytest.raises(TypeError):
        'foo' > y

    assert not 2 == y


def test_hash():
    x = sX(5)
    d = {x: 4}
    assert d[x] == 4


def test_product():
    p = qf.pauli_product(sX(0), sY(0))
    assert p == sZ(0, 1j)

    p = qf.pauli_product(sY(0), sX(0))
    assert p == sZ(0, -1j)

    p = qf.pauli_product(sX(0), sY(1))
    assert p == qf.Pauli.term([0, 1], 'XY', 1)

    p = qf.pauli_product(sX(0), sY(1), sY(0))
    assert p == qf.Pauli.term([0, 1], 'ZY', 1j)

    p = qf.pauli_product(sY(0), sX(0), sY(1))
    assert p == qf.Pauli.term([0, 1], 'ZY', -1j)


def test_mul():
    # TODO CHECK ALL PAULI MULTIPLICATIONS HERE
    assert sX(0) * sY(0) == sZ(0, 1j)


def test_scalar():
    a = qf.Pauli.scalar(1.0)
    assert a.is_scalar()
    assert a.is_identity()

    b = qf.Pauli.scalar(2.0)
    assert b + b == qf.Pauli.scalar(4.0)
    assert b * b == qf.Pauli.scalar(4.0)

    assert -b == qf.Pauli.scalar(-2.0)
    assert +b == qf.Pauli.scalar(2.0)

    assert b * 3 == qf.Pauli.scalar(6.0)
    assert 3 * b == qf.Pauli.scalar(6.0)

    assert sX(0) * 2 == sX(0, 2)

    x = sX(0) + sY(1)
    assert not x.is_scalar()
    assert not x.is_identity()

    c = sX(0) * sY(1)
    assert not c.is_scalar()
    assert not c.is_identity()


def test_zero():
    z = qf.Pauli.scalar(0.0)
    assert z.is_zero()
    assert z.is_scalar()

    z2 = qf.Pauli.zero()
    assert z == z2

    assert sX(0) - sX(0) == qf.Pauli.zero()


def test_merge_sum():
    p = qf.pauli_sum(qf.Pauli(((((1, 'Y'),), (3+0j)),)),
                     qf.Pauli(((((1, 'Y'),), (2+0j)),)))
    assert len(p) == 1
    assert p.terms[0][1] == 5


def test_power():
    p = sX(0) + sY(1) + qf.Pauli.term([2, 3], 'XY')

    assert p**0 == qf.Pauli.identity()
    assert p**1 == p
    assert p * p == p**2
    assert p * p * p == p**3
    assert p * p * p * p * p * p * p * p * p * p == p**10

    with pytest.raises(ValueError):
        p ** -1

    p ** 400


def test_simplify():
    t1 = sZ(0) * sZ(1)
    t2 = sZ(0) * sZ(1)
    assert (t1 + t2) == 2 * sZ(0) * sZ(1)


def test_dont_simplify():
    t1 = sZ(0) * sZ(1)
    t2 = sZ(2) * sZ(3)
    assert (t1 + t2) != 2 * sZ(0) * sZ(1)


def test_zero_term():
    qubit_id = 0
    coefficient = 10
    ps = sI(qubit_id) + sX(qubit_id)
    assert coefficient * qf.Pauli.zero() == qf.Pauli.zero()
    assert qf.Pauli.zero() * coefficient == qf.Pauli.zero()
    assert qf.Pauli.zero() * qf.Pauli.identity() == qf.Pauli.zero()
    assert qf.Pauli.zero() + qf.Pauli.identity() == qf.Pauli.identity()
    assert qf.Pauli.zero() + ps == ps
    assert ps + qf.Pauli.zero() == ps


def test_neg():
    ps = sY(0) - sX(0)
    ps -= sZ(1)
    ps = ps - 3
    ps = 3 + ps


def test_paulisum_iteration():
    term_list = [sX(2), sZ(4)]
    ps = sum(term_list)
    for ii, term in enumerate(ps):
        assert term_list[ii].terms[0] == term


def test_check_commutation():
    term1 = sX(0) * sX(1)
    term2 = sY(0) * sY(1)
    term3 = sY(0) * sZ(2)

    assert qf.paulis_commute(term2, term3)
    assert qf.paulis_commute(term2, term3)
    assert not qf.paulis_commute(term1, term3)


def test_commuting_sets():
    term1 = sX(0) * sX(1)
    term2 = sY(0) * sY(1)
    term3 = sY(0) * sZ(2)
    ps = term1 + term2 + term3
    pcs = qf.pauli_commuting_sets(ps)
    assert len(pcs) == 2

    pcs = qf.pauli_commuting_sets(term1)
    assert len(pcs) == 1


def test_get_qubits():
    term = sZ(0) * sX(1)
    assert term.qubits == [0, 1]

    sum_term = qf.Pauli.term([0], 'X', 0.5) \
        + 0.5j * qf.Pauli.term([10], 'Y') * qf.Pauli.term([0], 'Y', 0.5j)
    assert sum_term.qubits == [0, 10]


def test_check_commutation_rigorous():
    # more rigorous test.  Get all operators in Pauli group
    N = 3
    qubits = list(range(N))
    ps = [qf.Pauli.term(qubits, ops) for ops in product(PAULI_OPS, repeat=N)]

    commuting = []
    non_commuting = []
    for left, right in product(ps, ps):
        if left * right == right * left:
            commuting.append((left, right))
        else:
            non_commuting.append((left, right))

    # now that we have our sets let's check against our code.
    for left, right in non_commuting:
        assert not qf.paulis_commute(left, right)

    for left, right in commuting:
        assert qf.paulis_commute(left, right)
