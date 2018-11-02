# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Unit tests for quantumflow.visualization
"""


from math import pi
import shutil

import pytest

import quantumflow as qf


def test_circuit_to_latex():
    qf.circuit_to_latex(qf.ghz_circuit(range(15)))


def test_circuit_to_latex_error():
    circ = qf.Circuit([qf.CPHASE01(0.4, 0, 1)])
    with pytest.raises(NotImplementedError):
        qf.circuit_to_latex(circ)


def test_gates_to_latex():
    circ = qf.Circuit()

    circ += qf.I(7)
    circ += qf.X(0)
    circ += qf.Y(1)
    circ += qf.Z(2)
    circ += qf.H(3)
    circ += qf.S(4)
    circ += qf.T(5)
    circ += qf.S_H(6)
    circ += qf.T_H(7)

    circ += qf.RX(-0.5*pi, 0)
    circ += qf.RY(0.5*pi, 1)
    circ += qf.RZ((1/3)*pi, 1)
    circ += qf.RY(0.222, 1)

    circ += qf.TX(0.5, 0)
    circ += qf.TY(0.5, 1)
    circ += qf.TZ(0.4, 1)

    circ += qf.TZ(0.47276, 1)
    # Gate with cunning hack
    gate = qf.RZ(0.4, 1)
    gate.params['theta'] = qf.Parameter('\\theta')
    circ += gate

    circ += qf.CNOT(1, 2)
    circ += qf.CNOT(2, 1)
    circ += qf.CZ(1, 3)
    circ += qf.SWAP(1, 5)
    circ += qf.ISWAP(4, 2)

    # circ += qf.Barrier(0, 1, 2, 3, 4, 5, 6)  # Not yet supported

    circ += qf.CCNOT(1, 2, 3)
    circ += qf.CSWAP(4, 5, 6)

    circ += qf.P0(0)
    circ += qf.P1(1)

    circ += qf.Reset(2)
    circ += qf.Reset(4, 5, 6)

    circ += qf.H(4)
    # circ += qf.Reset()    # FIXME. Should fail with clear error message

    circ += qf.XX(0.25, 1, 3)
    circ += qf.YY(0.75, 1, 3)
    circ += qf.ZZ(1/3, 3, 1)

    circ += qf.Measure(0)

    latex = qf.circuit_to_latex(circ)

    print(latex)
    # qf.render_latex(latex).show()

    qf.circuit_to_image(circ)


def render_dependancies_installed():
    if shutil.which('pdflatex') is None:
        return False
    if shutil.which('pdftocairo') is None:
        return False
    return True


@pytest.mark.skipif(not render_dependancies_installed(),
                    reason='Necessary external dependencies not installed')
def test_render_latex():
    # TODO: Double check this circuit is correct
    circ = qf.addition_circuit(['a[0]', 'a[1]', 'a[2]', 'a[3]'],
                               ['b[0]', 'b[1]', 'b[2]', 'b[3]'],
                               ['cin', 'cout'])
    order = ['cin', 'a[0]', 'b[0]', 'a[1]', 'b[1]', 'a[2]', 'b[2]', 'a[3]',
             'b[3]', 'cout']
    # order = ['cin', 'a0', 'a1', 'a2', 'a3', 'b0', 'b1', 'b2', 'b3','cout']

    latex = qf.circuit_to_latex(circ, order)

    qf.render_latex(latex)
    # qf.render_latex(latex).show()
