
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import numpy as np

import pytest

import quantumflow as qf
from quantumflow.programs import TARGETS, PC


def test_empty_program():
    prog = qf.Program()
    ket = prog.run()
    assert ket.qubits == ()
    assert ket.qubit_nb == 0


def test_nop():
    prog = qf.Program()
    prog += qf.Nop()
    prog.run()

    for inst in prog:
        assert inst is not None

    assert qf.Nop().qubits == ()
    assert qf.Nop().qubit_nb == 0


def test_nop_evolve():
    prog = qf.Program()
    prog += qf.Nop()
    prog.evolve()


def test_compile_label():
    prog = qf.Program()
    prog += qf.Label('Here')
    prog += qf.Nop()
    prog += qf.Label('There')

    ket = prog.run()

    assert ket.memory[TARGETS] == {'Here': 0, 'There': 2}


def test_jump():
    ro = qf.Register()
    prog = qf.Program()
    prog += qf.Move(ro[0], 0)
    prog += qf.Jump('There')
    prog += qf.Not(ro[0])
    prog += qf.Label('There')
    prog += qf.Not(ro[0])
    ket = prog.run()
    assert ket.memory[ro[0]] == 1

    prog += qf.JumpWhen('There', ro[0])
    ket = prog.run()
    assert ket.memory[ro[0]] == 0

    prog += qf.Not(ro[0])
    prog += qf.JumpUnless('There', ro[0])
    ket = prog.run()
    assert ket.memory[ro[0]] == 1


def test_wait():
    ro = qf.Register()
    prog = qf.Program()
    prog += qf.Move(ro[0], 0)
    prog += qf.Wait()
    prog += qf.Not(ro[0])
    prog += qf.Wait()
    prog += qf.Not(ro[0])
    prog += qf.Wait()
    prog += qf.Not(ro[0])

    ket = prog.run()
    assert ket.memory[ro[0]] == 1


def test_include():
    prog = qf.Program()
    prog += qf.Move(('a', 0), 0)
    instr = qf.Include('somefile.quil', qf.Program())
    assert instr.quil() == 'INCLUDE "somefile.quil"'


def test_halt():
    ro = qf.Register()
    prog = qf.Program()
    prog += qf.Move(ro[0], 0)
    prog += qf.Halt()
    prog += qf.Not(ro[0])

    ket = prog.run()
    assert ket.memory[PC] == -1
    assert ket.memory[ro[0]] == 0


def test_reset():
    ro = qf.Register()
    prog = qf.Program()
    prog += qf.Move(ro[0], 1)
    prog += qf.Call('X', params=[], qubits=[0])
    prog += qf.Reset()
    prog += qf.Measure(0, ro[1])
    ket = prog.run()

    assert ket.qubits == (0,)
    assert ket.memory[ro[0]] == 1
    assert ket.memory[ro[1]] == 0


def test_reset_one():
    prog = qf.Program()
    prog += qf.Call('X', params=[], qubits=[0])
    prog += qf.Call('X', params=[], qubits=[1])
    prog += qf.Reset(0)
    prog += qf.Measure(0, ('b', 0))
    prog += qf.Measure(1, ('b', 1))
    ket = prog.run()

    assert ket.memory[('b', 0)] == 0
    assert ket.memory[('b', 1)] == 1


def test_xgate():
    prog = qf.Program()
    prog += qf.Call('X', params=[], qubits=[0])
    ket = prog.run()

    assert ket.qubits == (0,)
    # assert prog.cbits == []

    qf.print_state(ket)


def test_call():
    prog = qf.Program()
    prog += qf.Call('BELL', params=[], qubits=[])

    assert str(prog) == 'BELL\n'


# FIXME: ref-qvm has do_until option? From pyquil?
def test_measure_until():

    prog = qf.Program()
    prog += qf.Move(('c', 2), 1)
    prog += qf.Label('redo')
    prog += qf.Call('X', [], [0])
    prog += qf.Call('H', [], [0])
    prog += qf.Measure(0, ('c', 2))
    prog += qf.JumpUnless('redo', ('c', 2))

    ket = prog.run()
    assert ket.memory[('c', 2)] == 1


def test_belltest():
    prog = qf.Program()
    prog += qf.Call('H', [], [0])
    prog += qf.Call('CNOT', [], [0, 1])
    ket = prog.run()

    assert qf.states_close(ket, qf.ghz_state(2))
    # qf.print_state(prog.state)


def test_occupation_basis():
    prog = qf.Program()
    prog += qf.Call('X', [], [0])
    prog += qf.Call('X', [], [1])
    prog += qf.Call('I', [], [2])
    prog += qf.Call('I', [], [3])

    ket = prog.run()

    assert ket.qubits == (0, 1, 2, 3)
    probs = qf.asarray(ket.probabilities())
    assert probs[1, 1, 0, 0] == 1.0
    assert probs[1, 1, 0, 1] == 0.0

# TODO: TEST EXP CIRCUIT from test_qvm


def test_qaoa_circuit():
    wf_true = [0.00167784 + 1.00210180e-05*1j, 0.50000000 - 4.99997185e-01*1j,
               0.50000000 - 4.99997185e-01*1j, 0.00167784 + 1.00210180e-05*1j]
    prog = qf.Program()
    prog += qf.Call('RY', [np.pi/2], [0])
    prog += qf.Call('RX', [np.pi], [0])
    prog += qf.Call('RY', [np.pi/2], [1])
    prog += qf.Call('RX', [np.pi], [1])
    prog += qf.Call('CNOT', [], [0, 1])
    prog += qf.Call('RX', [-np.pi/2], [1])
    prog += qf.Call('RY', [4.71572463191], [1])
    prog += qf.Call('RX', [np.pi/2], [1])
    prog += qf.Call('CNOT', [], [0, 1])
    prog += qf.Call('RX', [-2*2.74973750579], [0])
    prog += qf.Call('RX', [-2*2.74973750579], [1])

    test_state = prog.run()
    true_state = qf.State(wf_true)
    assert qf.states_close(test_state, true_state)


QUILPROG = """RY(pi/2) 0
RX(pi) 0
RY(pi/2) 1
RX(pi) 1
CNOT 0 1
RX(-pi/2) 1
RY(4.71572463191) 1
RX(pi/2) 1
CNOT 0 1
RX(-5.49947501158) 0
RX(-5.49947501158) 1
"""


# HADAMARD = """
# DEFGATE HADAMARD:
#     1/sqrt(2), 1/sqrt(2)
#     1/sqrt(2), -1/sqrt(2)
# HADAMARD 0
# """


# def test_defgate():
#     prog = qf.forest.quil_to_program(HADAMARD)
#     ket = prog.run()
#     qf.print_state(ket)
#     assert qf.states_close(ket, qf.ghz_state(1))


# CP = """
# DEFGATE CP(%theta):
#     1, 0, 0, 0
#     0, 1, 0, 0
#     0, 0, 1, 0
#     0, 0, 0, cis(pi+%theta)

# X 0
# X 1
# CP(1.0) 0 1
# """


# def test_defgate_param():
#     prog = qf.forest.quil_to_program(CP)
#     # ket0 = prog.compile()
#     # qf.print_state(ket0)
#     ket1 = prog.run()
#     qf.print_state(ket1)

#     ket = qf.zero_state(2)
#     ket = qf.X(0).run(ket)
#     ket = qf.X(1).run(ket)
#     ket = qf.CPHASE(1.0, 0, 1).run(ket)
#     qf.print_state(ket)

#     assert qf.states_close(ket1, ket)


CIRC0 = """DEFCIRCUIT CIRCX:
    X 0

CIRCX 0
"""


def test_defcircuit():
    prog = qf.Program()

    circ = qf.DefCircuit('CIRCX', {})
    circ += qf.Call('X', params=[], qubits=[0])

    prog += circ
    prog += qf.Call('CIRCX', params=[], qubits=[0])

    assert str(prog) == CIRC0

    # prog.compile()
    assert prog.qubits == [0]

    # FIXME: Not implemented
    # prog.run()
    # qf.print_state(prog.state)


CIRC1 = """DEFCIRCUIT CIRCX(this):
    NOP

"""


def test_defcircuit_param():
    prog = qf.Program()
    circ = qf.DefCircuit('CIRCX', {'this': None})
    circ += qf.Nop()
    prog += circ
    assert str(prog) == CIRC1


def test_exceptions():
    prog = qf.Program()
    prog += qf.Call('NOT_A_GATE', [], [0])
    with pytest.raises(RuntimeError):
        prog.run()
