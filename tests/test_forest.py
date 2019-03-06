
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Unittests for quantumflow.quil"""

import shutil
from math import pi

import numpy as np

import pytest

import quantumflow as qf
from quantumflow.forest import pyquil

from . import skip_unless_pdflatex


def dependancies_installed():
    if shutil.which('qvm') is None:
        return False
    if shutil.which('quilc') is None:
        return False
    return True


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


def test_circuit_to_pyquil():
    circ = qf.Circuit()
    circ += qf.X(0)

    prog = qf.forest.circuit_to_pyquil(circ)
    assert str(prog) == "X 0\n"

    circ = qf.Circuit()
    circ1 = qf.Circuit()
    circ2 = qf.Circuit()
    circ1 += qf.RY(pi/2, 0)
    circ1 += qf.RX(pi, 0)
    circ1 += qf.RY(pi/2, 1)
    circ1 += qf.RX(pi, 1)
    circ1 += qf.CNOT(0, 1)
    circ2 += qf.RX(-pi/2, 1)
    circ2 += qf.RY(4.71572463191, 1)
    circ2 += qf.RX(pi/2, 1)
    circ2 += qf.CNOT(0, 1)
    circ2 += qf.RX(-2*2.74973750579, 0)
    circ2 += qf.RX(-2*2.74973750579, 1)
    circ.extend(circ1)
    circ.extend(circ2)

    prog = qf.forest.circuit_to_pyquil(circ)

    print(prog)

    assert QUILPROG == str(prog)


def test_pyquil_to_circuit():

    prog = pyquil.Program(QUILPROG)
    circ = qf.forest.pyquil_to_circuit(prog)
    prog_new = qf.forest.circuit_to_pyquil(circ)
    print(prog_new)
    assert str(prog_new) == QUILPROG


TRIALS = 100

BELL_STATE = pyquil.Program(pyquil.H(0), pyquil.CNOT(0, 1))

BELL_STATE_MEASURE = """DECLARE ro BIT[2]
H 0
CNOT 0 1
MEASURE 0 ro[0]
MEASURE 1 ro[1]
PRAGMA not_a_pragma
HALT
"""


def test_pyquil_to_circuit_more():
    # Check will ignore or convert DECLARE, MEASURE
    prog = pyquil.Program(BELL_STATE_MEASURE)
    circ = qf.forest.pyquil_to_circuit(prog)
    prog_new = qf.forest.circuit_to_pyquil(circ)

    print(circ.qubits)
    print(prog_new)
    # assert str(prog_new) == QUILPROG


@skip_unless_pdflatex
def test_pyquil_to_latex():
    prog = pyquil.Program(BELL_STATE_MEASURE)
    circ = qf.forest.pyquil_to_circuit(prog)
    latex = qf.circuit_to_latex(circ)
    img = qf.render_latex(latex)
    assert img is not None
    # img.show()


def test_qvm_run():
    qvm = qf.forest.QuantumFlowQVM()
    for _ in range(TRIALS):
        qvm.load(BELL_STATE_MEASURE).run().wait()
        res = qvm.read_from_memory_region(region_name='ro')
        assert res[0] == res[1]


def test_wavefunction():
    wf_program = pyquil.Program(pyquil.Declare('ro', 'BIT', 1),
                                pyquil.H(0),
                                pyquil.CNOT(0, 1),
                                pyquil.MEASURE(0, ('ro', 0)),
                                pyquil.H(0))

    wf_expected1 = np.array([0. + 0.j, 0. + 0.j,
                             0.70710678 + 0.j, -0.70710678 + 0.j])
    wf_expected0 = np.array([0.70710678 + 0.j, 0.70710678 + 0.j,
                             0. + 0.j, 0. + 0.j, ])

    qvm = qf.forest.QuantumFlowQVM()
    qvm.load(wf_program).run().wait()
    wf = qvm.wavefunction()
    res = qvm.read_from_memory_region(region_name='ro')[0]

    for _ in range(TRIALS):
        if res == 0:
            assert np.all(np.isclose(wf.amplitudes, wf_expected0))
        else:
            assert np.all(np.isclose(wf.amplitudes, wf_expected1))


def test_qaoa_circuit():
    circ = qf.Circuit()
    circ += qf.RY(pi/2, 0)
    circ += qf.RX(pi, 0)
    circ += qf.RY(pi/2, 1)
    circ += qf.RX(pi, 1)
    circ += qf.CNOT(0, 1)
    circ += qf.RX(-pi/2, 1)
    circ += qf.RY(4.71572463191, 1)
    circ += qf.RX(pi/2, 1)
    circ += qf.CNOT(0, 1)
    circ += qf.RX(-2*2.74973750579, 0)
    circ += qf.RX(-2*2.74973750579, 1)

    ket = qf.zero_state(2)
    ket = circ.run(ket)

    prog = qf.forest.circuit_to_pyquil(circ)

    qvm = qf.forest.QuantumFlowQVM()
    wf = qvm.load(prog).run().wait().wavefunction()

    state = qf.forest.wavefunction_to_state(wf)
    assert qf.states_close(ket, state)


def test_exceptions():
    circ = qf.Circuit()
    circ += qf.T_H(0)
    with pytest.raises(ValueError):
        qf.forest.circuit_to_pyquil(circ)

    quil = "XOR c[0] c[1]"
    prog = pyquil.Program(quil)
    with pytest.raises(ValueError):  # Not protoquil
        circ = qf.forest.pyquil_to_circuit(prog)

    qvm = qf.forest.QuantumFlowQVM()
    qvm.load(BELL_STATE_MEASURE)
    with pytest.raises(NotImplementedError):
        qvm.write_memory(region_name='ro')

    with pytest.raises(NotImplementedError):
        qvm.run().wait()
        qvm.read_from_memory_region(region_name='ro', offsets=[1])


def test_null_complier():
    quil = "XOR c[0] c[1]"
    prog0 = pyquil.Program(quil)
    nc = qf.forest.NullCompiler()
    nc.get_version_info()
    prog1 = nc.quil_to_native_quil(prog0)
    assert prog1 == prog0
    prog2 = nc.native_quil_to_executable(prog0)
    assert prog2 == prog0


@pytest.mark.skipif(not dependancies_installed(),
                    reason='Necessary external dependencies not installed')
def test_qvm():
    circ = qf.ghz_circuit([0, 1, 2, 3])
    with qf.pyquil.local_qvm():
        qf.forest.qvm_run_and_measure(circ, 1)


# def test_get_virtual_qc():
#     qf.forest.get_virtual_qc(4)
#     qf.forest.get_virtual_qc(4, noisy=True)


# def test_get_compiler():
#     qf.forest.get_compiler(4)
