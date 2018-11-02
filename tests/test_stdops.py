
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


import quantumflow as qf


def test_measure():
    prog = qf.Program()
    prog += qf.Measure(0, ('c', 0))
    prog += qf.Call('X', params=[], qubits=[0])
    prog += qf.Measure(0, ('c', 1))
    ket = prog.run()

    assert ket.qubits == (0,)
    # assert ket.cbits == [('c', 0), ('c', 1)]  # FIXME
    assert ket.memory[('c', 0)] == 0
    assert ket.memory[('c', 1)] == 1


def test_barrier():
    circ = qf.Circuit()
    circ += qf.Barrier(0, 1, 2)
    circ += qf.Barrier(0, 1, 2).H
    circ.run()
    circ.evolve()

    assert str(qf.Barrier(0, 1, 2)) == 'BARRIER 0 1 2'


def test_if():
    circ = qf.Circuit()
    c = qf.Register('c')
    circ += qf.Move(c[0], 0)
    circ += qf.Move(c[1], 1)
    circ += qf.If(qf.X(0), c[1])
    circ += qf.Measure(0, c[0])
    ket = circ.run()
    assert ket.memory[c[0]] == 1
    assert circ.evolve().memory[c[0]] == 1

    circ = qf.Circuit()
    c = qf.Register('c')
    circ += qf.Move(c[0], 0)
    circ += qf.Move(c[1], 0)
    circ += qf.If(qf.X(0), c[1])
    circ += qf.Measure(0, c[0])
    ket = circ.run()
    assert ket.memory[c[0]] == 0
    assert circ.evolve().memory[c[0]] == 0

    circ = qf.Circuit()
    c = qf.Register('c')
    circ += qf.Move(c[0], 0)
    circ += qf.Move(c[1], 0)
    circ += qf.If(qf.X(0), c[1], value=False)
    circ += qf.Measure(0, c[0])
    ket = circ.run()
    assert ket.memory[c[0]] == 1
    assert circ.evolve().memory[c[0]] == 1


def test_neg():
    c = qf.Register('c')
    assert str(qf.Neg(c[10])) == 'NEG c[10]'


def test_logics():
    c = qf.Register('c')

    circ = qf.Circuit([qf.Move(c[0], 0),
                       qf.Move(c[1], 1),
                       qf.And(c[0], c[1])])
    # assert len(circ) == 3     # FIXME
    # assert circ.cbits == [c[0], c[1]] # FIXME

    ket = circ.run()
    assert ket.memory == {c[0]: 0, c[1]: 1}

    circ += qf.Not(c[1])
    circ += qf.And(c[0], c[1])
    ket = circ.run(ket)
    assert ket.memory == {c[0]: 0, c[1]: 0}

    circ = qf.Circuit()
    circ += qf.Move(c[0], 0)
    circ += qf.Move(c[1], 1)
    circ += qf.Ior(c[0], c[1])
    ket = circ.run()
    assert ket.memory == {c[0]: 1, c[1]: 1}

    circ = qf.Circuit()
    circ += qf.Move(c[0], 1)
    circ += qf.Move(c[1], 1)
    circ += qf.Xor(c[0], c[1])
    ket = circ.run()
    assert ket.memory == {c[0]: 0, c[1]: 1}

    circ += qf.Exchange(c[0], c[1])
    ket = circ.run(ket)
    assert ket.memory == {c[0]: 1, c[1]: 0}

    circ += qf.Move(c[0], c[1])
    ket = circ.run(ket)
    assert ket.memory == {c[0]: 0, c[1]: 0}


def test_add():
    ro = qf.Register()
    prog = qf.Program()
    prog += qf.Move(ro[0], 1)
    prog += qf.Move(ro[1], 2)
    prog += qf.Add(ro[0], ro[1])
    prog += qf.Add(ro[0], 4)
    ket = prog.run()
    assert ket.memory[ro[0]] == 7


def test_density_add():
    ro = qf.Register()
    circ = qf.Circuit()
    circ += qf.Move(ro[0], 1)
    circ += qf.Move(ro[1], 2)
    circ += qf.Add(ro[0], ro[1])
    circ += qf.Add(ro[0], 4)
    rho = circ.evolve()
    assert rho.memory[ro[0]] == 7


def test_mul():
    ro = qf.Register()
    prog = qf.Program()
    prog += qf.Move(ro[0], 1)
    prog += qf.Move(ro[1], 2)
    prog += qf.Mul(ro[0], ro[1])
    prog += qf.Mul(ro[0], 4)
    ket = prog.run()
    assert ket.memory[ro[0]] == 8


def test_div():
    ro = qf.Register()
    prog = qf.Program()
    prog += qf.Move(ro[0], 4)
    prog += qf.Move(ro[1], 1)
    prog += qf.Div(ro[0], ro[1])
    prog += qf.Div(ro[0], 2)
    ket = prog.run()
    assert ket.memory[ro[0]] == 2


def test_sub():
    ro = qf.Register()
    prog = qf.Program()
    prog += qf.Move(ro[0], 1)
    prog += qf.Move(ro[1], 2)
    prog += qf.Sub(ro[0], ro[1])
    prog += qf.Sub(ro[0], 4)
    prog += qf.Neg(ro[0])
    ket = prog.run()
    assert ket.memory[ro[0]] == 5


def test_comparisions():
    ro = qf.Register()
    prog = qf.Program()
    prog += qf.Move(ro[0], 1)
    prog += qf.Move(ro[1], 2)
    prog += qf.EQ(('eq', 0), ro[0], ro[1])
    prog += qf.GT(('gt', 0), ro[0], ro[1])
    prog += qf.GE(('ge', 0), ro[0], ro[1])
    prog += qf.LT(('lt', 0), ro[0], ro[1])
    prog += qf.LE(('le', 0), ro[0], ro[1])
    ket = prog.run()
    assert ket.memory[('eq', 0)] == 0
    assert ket.memory[('gt', 0)] == 0
    assert ket.memory[('ge', 0)] == 0
    assert ket.memory[('lt', 0)] == 1
    assert ket.memory[('le', 0)] == 1
