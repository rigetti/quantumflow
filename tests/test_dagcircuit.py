# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Unit tests for quantumflow.dagcircuit
"""


import numpy as np

import quantumflow as qf


# TODO Refactor in test_circuit
def _test_circ():
    # Adapted from referenceQVM
    circ = qf.Circuit()
    circ += qf.TY(1/2, 0)
    circ += qf.TX(1, 0)
    circ += qf.TY(1/2, 1)
    circ += qf.TX(1, 1)
    circ += qf.CNOT(0, 1)
    circ += qf.TX(-1/2, 1)
    circ += qf.TY(4.71572463191 / np.pi, 1)
    circ += qf.TX(1/2, 1)
    circ += qf.CNOT(0, 1)
    circ += qf.TX(-2 * 2.74973750579 / np.pi, 0)
    circ += qf.TX(-2 * 2.74973750579 / np.pi, 1)
    return circ


def _true_ket():
    # Adapted from referenceQVM
    wf_true = np.array(
        [0.00167784 + 1.00210180e-05*1j, 0.50000000 - 4.99997185e-01*1j,
         0.50000000 - 4.99997185e-01*1j, 0.00167784 + 1.00210180e-05*1j])
    return qf.State(wf_true.reshape((2, 2)))


def test_init():
    dag = qf.DAGCircuit([])
    assert dag.size() == 0


def test_inverse():
    dag = qf.DAGCircuit(_test_circ())
    inv_dag = dag.H

    ket0 = qf.random_state(2)
    ket1 = dag.run(ket0)
    ket2 = inv_dag.run(ket1)

    assert qf.states_close(ket0, ket2)


def test_ascircuit():
    circ0 = qf.ghz_circuit(range(5))
    dag = qf.DAGCircuit(circ0)
    circ1 = qf.Circuit(dag)

    assert tuple(circ1.qubits) == (0, 1, 2, 3, 4)
    assert dag.qubits == circ0.qubits
    assert dag.qubit_nb == 5


def test_asgate():
    gate0 = qf.ZYZ(0.1, 2.2, 0.5)
    circ0 = qf.zyz_circuit(0.1, 2.2, 0.5, 0)
    dag0 = qf.DAGCircuit(circ0)
    gate1 = dag0.asgate()
    assert qf.gates_close(gate0, gate1)


def test_evolve():
    rho0 = qf.random_state(3).asdensity()
    rho1 = qf.CCNOT(0, 1, 2).evolve(rho0)

    dag = qf.DAGCircuit(qf.ccnot_circuit([0, 1, 2]))
    rho2 = dag.evolve(rho0)

    assert qf.densities_close(rho1, rho2)


def test_aschannel():
    rho0 = qf.random_state(3).asdensity()
    rho1 = qf.CCNOT(0, 1, 2).evolve(rho0)

    dag = qf.DAGCircuit(qf.ccnot_circuit([0, 1, 2]))
    chan = dag.aschannel()
    rho2 = chan.evolve(rho0)

    assert qf.densities_close(rho1, rho2)


def test_depth():
    circ = qf.qft_circuit([0, 1, 2, 3])
    dag = qf.DAGCircuit(circ)
    assert dag.depth() == 8

    circ = qf.ghz_circuit(range(5))
    dag = qf.DAGCircuit(circ)
    assert dag.depth() == 5

    assert dag.depth(local=False) == 4


def test_layers():
    circ0 = qf.ghz_circuit(range(5))
    dag = qf.DAGCircuit(circ0)
    layers = dag.layers()
    assert len(layers.elements) == dag.depth()


def test_components():
    circ = qf.Circuit()
    circ += qf.H(0)
    circ += qf.H(1)
    dag = qf.DAGCircuit(circ)
    assert dag.component_nb() == 2

    circ += qf.CNOT(0, 1)
    dag = qf.DAGCircuit(circ)
    assert dag.component_nb() == 1

    circ0 = qf.ghz_circuit([0, 2, 4, 6, 8])
    circ1 = qf.ghz_circuit([1, 3, 5, 7, 9])

    circ = qf.Circuit()
    circ.extend(circ0)
    circ.extend(circ1)
    dag = qf.DAGCircuit(circ)
    comps = dag.components()
    assert dag.component_nb() == 2

    circ0 = qf.qft_circuit([0, 2, 4, 6])
    circ1 = qf.ghz_circuit([1, 3, 5, 7])
    circ.extend(circ0)
    circ.extend(circ1)
    circ += qf.H(10)
    dag = qf.DAGCircuit(circ)
    comps = dag.components()
    assert dag.component_nb() == 3
    assert len(comps) == 3


def test_next():
    circ = qf.ghz_circuit([0, 2, 4, 6, 8])
    elem = circ.elements[3]
    dag = qf.DAGCircuit(circ)

    assert dag.next_element(elem, elem.qubits[1]) == circ.elements[4]
    assert dag.prev_element(elem, elem.qubits[0]) == circ.elements[2]

    # FIXME: out and in nodes should also be Operation's ?
    assert dag.next_element(elem, elem.qubits[0]) == ('out', 4)
    assert dag.prev_element(elem, elem.qubits[1]) == ('in', 6)
