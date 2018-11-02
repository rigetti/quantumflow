
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: QAOA (Quantum Approximate Optimization Algorithm)
"""

from typing import Sequence

import numpy as np
import networkx as nx

from .circuits import Circuit
from .stdgates import H, RX, RZ, ZZ


# DOCME
# TODO: Make Hamiltonian explicit
def qubo_circuit(
        graph: nx.Graph,
        steps: int,
        beta: Sequence,
        gamma: Sequence) -> Circuit:
    """
    A QAOA circuit for the Quadratic Unconstrained Binary Optimization
    problem (i.e. an Ising model).

    Args:
        graph : a networkx graph instance with optional edge and node weights
        steps : number of QAOA steps
        beta  : driver parameters (One per step)
        gamma : cost parameters (One per step)

    """

    qubits = list(graph.nodes())

    # Initialization
    circ = Circuit()
    for q0 in qubits:
        circ += H(q0)

    # Run for given number of QAOA steps
    for p in range(0, steps):

        # Cost
        for q0, q1 in graph.edges():
            weight = graph[q0][q1].get('weight', 1.0)
            # Note factor of pi due to parameterization of ZZ gate
            circ += ZZ(-weight * gamma[p] / np.pi, q0, q1)

        for q0 in qubits:
            node_weight = graph.nodes[q0].get('weight', None)
            if node_weight is not None:
                circ += RZ(node_weight, q0)

        # Drive
        for q0 in qubits:
            circ += RX(beta[p], q0)

    return circ


# TODO: DOCME. Explain what's going on here.
def graph_cuts(graph: nx.Graph) -> np.ndarray:
    """For the given graph, return the cut value for all binary assignments
    of the graph.
    """

    N = len(graph)
    diag_hamiltonian = np.zeros(shape=([2]*N), dtype=np.double)
    for q0, q1 in graph.edges():
        for index, _ in np.ndenumerate(diag_hamiltonian):
            if index[q0] != index[q1]:
                weight = graph[q0][q1].get('weight', 1)
                diag_hamiltonian[index] += weight

    return diag_hamiltonian
